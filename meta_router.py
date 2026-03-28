"""
BrainStacks Meta-Router — Reasoning-Aware Outcome Router

============

Author: Mohammad R. Abu Ayyash — Brains Build Research, Palestine
"""

import os, sys, json, math, time, random, warnings, gc
from pathlib import Path
from typing import List, Tuple, Dict
from itertools import combinations

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _ensure(pkg, cmd=None):
    try: __import__(pkg)
    except ImportError:
        os.system(cmd or f"{sys.executable} -m pip install -q {pkg}")

_ensure("datasets", f"{sys.executable} -m pip install -q datasets")

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

torch.manual_seed(42)
random.seed(42)


# ════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════

MODEL_NAME     = "google/gemma-3-12b-it"
MAX_SEQ_LEN    = 512
LORA_R         = 16
LORA_ALPHA     = 16.0
USE_RSLORA     = True
NUM_EXPERTS    = 4
TOP_K          = 2
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_LORA_SCALE    = LORA_ALPHA / math.sqrt(LORA_R) if USE_RSLORA else LORA_ALPHA / LORA_R

COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

# Paths
SAVE_DIR       = "./BrainStacks_gemma3"
MANIFEST_PATH  = os.path.join(SAVE_DIR, "manifest.json")
ORACLE_CACHE_DIR = os.path.join(SAVE_DIR, "oracle_cache")

# only recompute reasoning oracle, use cache for the other 4
ORACLE_RECOMPUTE_DOMAINS = {"reasoning"}

# Router hyperparameters
ROUTER_SEQ_LEN             = 256    
ROUTER_BATCH_SIZE          = 16
ROUTER_EPOCHS              = 8       
ROUTER_LR                  = 8e-4
ROUTER_WEIGHT_DECAY        = 0.01
SAMPLES_PER_DOMAIN         = 2000
OUTCOME_SAMPLES_PER_DOMAIN = 200
CHAT_FLOOR                 = 0.20
GATE_THRESHOLD             = 0.12
GEN_MAX_TOKENS             = 250

# Domain datasets for router training data
DOMAIN_DATA = {
    "chat":      {"dataset": "HuggingFaceH4/ultrafeedback_binarized", "split": "train_sft",
                  "format": "ultrafeedback"},
    "code":      {"dataset": "iamtarun/python_code_instructions_18k_alpaca", "split": "train",
                  "format": "alpaca"},
    "math":      {"dataset": "openai/gsm8k", "config": "main", "split": "train",
                  "format": "gsm8k"},
    "medical":   {"dataset": "medalpaca/medical_meadow_medical_flashcards", "split": "train",
                  "format": "alpaca"},
    "reasoning": {"dataset": "open-thoughts/OpenThoughts-114k",
                    "split": "train", "format": "openthoughts"},
}

# External mixed-domain sources (cross-domain prompts)
EXTERNAL_MIXED_SOURCES = [
    {"pair": ("code", "math"),    "dataset": "MathLLMs/MathCodeInstruct",
     "split": "train", "kind": "messages_mathcode", "take": 400, "buffer_size": 2000},
    {"pair": ("medical", "math"), "dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
     "config": "en_mix", "split": "train", "kind": "medical_reasoning", "take": 400, "buffer_size": 2000},
    {"pair": ("chat", "code"),    "dataset": "iamtarun/code_instructions_120k_alpaca",
     "split": "train", "kind": "alpaca_code", "take": 400, "buffer_size": 2000},
    {"pair": ("chat", "math"),    "dataset": "openai/gsm8k", "config": "main",
     "split": "train", "kind": "gsm8k", "take": 400, "buffer_size": 2000},
    {"pair": ("chat", "medical"), "dataset": "medalpaca/medical_meadow_medical_flashcards",
     "split": "train", "kind": "alpaca", "take": 400, "buffer_size": 2000},

    # reasoning pairs — verbal reasoning only, NO code-like data ──
    {"pair": ("chat", "reasoning"),    "dataset": "lucasmccabe/logiqa", "revision": "refs/convert/parquet",
    "split": "train", "kind": "logiqa", "take": 400, "buffer_size": 2000},
    {"pair": ("reasoning", "math"),    "dataset": "deepmind/aqua_rat", "config": "raw",
     "split": "train", "kind": "aqua_rat", "take": 400, "buffer_size": 2000},
    {"pair": ("reasoning", "code"),    "dataset": "open-r1/codeforces",
     "split": "train", "kind": "codeforces_reasoning", "take": 400, "buffer_size": 2000},
    {"pair": ("reasoning", "medical"), "dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
     "config": "en_mix", "split": "train", "kind": "medical_reasoning", "take": 400, "buffer_size": 2000},
]

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
)


# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

def safe_text(x):
    return str(x).strip() if x is not None else ""


def get_embed_dim(model):
    """Get hidden dimension from model config."""
    if hasattr(model, 'config'):
        for attr in ['hidden_size', 'd_model', 'n_embd']:
            if hasattr(model.config, attr):
                return getattr(model.config, attr)
    # Fallback: find embedding layer
    for mod in model.modules():
        if isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 30000:
            return mod.weight.shape[1]
    raise ValueError("Could not detect hidden_size from model. Set it manually.")

# ════════════════════════════════════════════════════════════════════════
#  MOE-LORA COMPONENTS — matches SFT script exactly
# ════════════════════════════════════════════════════════════════════════

class LoRAExpert(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.A = nn.Linear(in_f, LORA_R, bias=False)
        self.B = nn.Linear(LORA_R, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(self.A(x)) * _LORA_SCALE


class MoELoRADelta(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.num_experts = NUM_EXPERTS; self.top_k = TOP_K
        self.experts = nn.ModuleList([LoRAExpert(in_f, out_f) for _ in range(NUM_EXPERTS)])
        self.router = nn.Linear(in_f, NUM_EXPERTS, bias=False)
        self.noise_linear = nn.Linear(in_f, NUM_EXPERTS, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.constant_(self.noise_linear.weight, 0.0)
        self.aux_loss = torch.tensor(0.0)
    def forward(self, x):
        flat = x.view(-1, x.size(-1))
        logits = self.router(flat)
        if self.training:
            logits = logits + F.softplus(self.noise_linear(flat)) * torch.randn_like(logits)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        sparse = torch.full_like(logits, float('-inf'))
        sparse.scatter_(-1, topk_idx, topk_vals)
        gates = F.softmax(sparse, dim=-1)
        if self.training:
            probs = F.softmax(logits.detach(), dim=-1)
            self.aux_loss = self.num_experts * (probs.mean(0) * gates.gt(0).float().mean(0)).sum()
        else:
            self.aux_loss = torch.tensor(0.0, device=flat.device)
        A = torch.stack([e.A.weight for e in self.experts])
        B = torch.stack([e.B.weight for e in self.experts])
        mid = torch.einsum("tf,erf->ter", flat, A)
        all_deltas = torch.einsum("ter,eor->teo", mid, B) * _LORA_SCALE
        delta = (gates.unsqueeze(-1) * all_deltas).sum(dim=1)
        return delta.view(*x.shape[:-1], -1)


class StackedMoELoRALayer(nn.Module):
    """Extended with domain weights for routing + CPU offload shuttle."""
    def __init__(self, frozen_linear):
        super().__init__()
        self.frozen = frozen_linear
        self.frozen_stacks = nn.ModuleList()
        self.active_stack = None
        # Routing support
        self._domain_weights = None
        self._domain_stack_counts = None
        self._router_base_only = False

    @property
    def weight(self): return self.frozen.weight
    @property
    def bias(self): return self.frozen.bias

    def forward(self, x):
        out = self.frozen(x)
        out_dtype = out.dtype

        # Base-only mode: skip all stacks (for encoding prompts for router)
        if self._router_base_only:
            return out

        if self.frozen_stacks:
            with torch.no_grad(), torch.amp.autocast(x.device.type, enabled=x.is_cuda):
                if self._domain_weights is not None and self._domain_stack_counts is not None:
                    # ROUTED: weight each domain's stacks by router output
                    start = 0
                    for d_idx, count in enumerate(self._domain_stack_counts):
                        if count <= 0: continue
                        w = self._domain_weights[d_idx]
                        w_val = float(w.float().item()) if torch.is_tensor(w) else float(w)
                        end = start + count
                        if w_val > GATE_THRESHOLD:
                            w_t = torch.tensor(w_val, device=x.device, dtype=out_dtype)
                            for stack in self.frozen_stacks[start:end]:
                                was_cpu = not next(stack.parameters()).is_cuda
                                if was_cpu: stack.to(x.device)
                                out = out + w_t * stack(x).to(dtype=out_dtype)
                                if was_cpu: stack.cpu()
                        start = end
                else:
                    # UNGATED: all stacks fire equally
                    for stack in self.frozen_stacks:
                        was_cpu = not next(stack.parameters()).is_cuda
                        if was_cpu: stack.to(x.device)
                        out = out + stack(x).to(dtype=out_dtype)
                        if was_cpu: stack.cpu()

        if self.active_stack is not None:
            out = out + self.active_stack(x).to(dtype=out_dtype)
        return out


# ════════════════════════════════════════════════════════════════════════
#  MODEL LOADING — Gemma 3 12B 4-bit + Stacked MoE-LoRA replacement
# ════════════════════════════════════════════════════════════════════════

def inject_stacked_layers(model):
    stacked_layers = []
    for name, mod in list(model.named_modules()):
        for target in TARGET_MODULES:
            if name.endswith(target) and isinstance(mod, (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                layer = StackedMoELoRALayer(mod)
                setattr(parent, child_name, layer)
                stacked_layers.append(layer)
                break
    for p in model.parameters(): p.requires_grad_(False)
    return model, stacked_layers


def load_stack_as_frozen(model, stacked_layers, stack_path, device):
    """Load a saved stack .pt file as a frozen stack."""
    # Add temporary active stack
    for layer in stacked_layers:
        in_f = layer.frozen.in_features
        out_f = layer.frozen.out_features
        layer.active_stack = MoELoRADelta(in_f, out_f).to(device)

    state = torch.load(stack_path, map_location=device, weights_only=False)
    for name, mod in model.named_modules():
        if isinstance(mod, StackedMoELoRALayer) and mod.active_stack is not None:
            for pname, p in mod.active_stack.named_parameters():
                key = f"{name}.active_stack.{pname}"
                if key in state:
                    p.data.copy_(state[key].to(device=device, dtype=p.dtype))

    # Freeze + offload to CPU
    for layer in stacked_layers:
        if layer.active_stack is not None:
            for p in layer.active_stack.parameters():
                p.requires_grad_(False)
            layer.active_stack.half()
            layer.active_stack.cpu()
            layer.frozen_stacks.append(layer.active_stack)
            layer.active_stack = None


def load_all_stacks_from_manifest(model, stacked_layers, device):
    """Load all domain stacks from manifest.json. Returns domain_names and stacks_per_domain."""
    if not os.path.exists(MANIFEST_PATH):
        print(f"  [Error] No manifest at {MANIFEST_PATH}")
        return [], {}

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    domain_names = []
    stacks_per_domain = {}

    for block in manifest["domains"]:
        name = block["name"]
        stack_files = [sf for sf in block["stack_files"] if os.path.exists(sf)]
        if not stack_files:
            print(f"  [Skip] {name}: no stack files found")
            continue

        for sf in stack_files:
            load_stack_as_frozen(model, stacked_layers, sf, device)
            print(f"  [Load] {sf}")

        stacks_per_domain[name] = len(stack_files)
        domain_names.append(name)

    # Set domain stack counts on all layers
    counts = [stacks_per_domain.get(d, 0) for d in domain_names]
    for layer in stacked_layers:
        layer._domain_stack_counts = counts

    n = sum(len(l.frozen_stacks) for l in stacked_layers) // max(len(stacked_layers), 1)
    print(f"\n  [Ready] {n} stacks/layer  |  Domains: {domain_names}  |  Counts: {counts}")
    return domain_names, stacks_per_domain


def set_domain_weights(model, weights):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer):
            m._domain_weights = weights

def clear_domain_weights(model):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer):
            m._domain_weights = None

def set_router_base_only(model, flag):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer):
            m._router_base_only = flag


# ════════════════════════════════════════════════════════════════════════
#  META-ROUTER — Deep Semantic + Sigmoid + Outcome-Based
# ════════════════════════════════════════════════════════════════════════

class MetaRouter(nn.Module):
    """
    Deep features (mid+last hidden) + cross-attention domain queries
    + SIGMOID output (independent per domain, true cross-domain).
    """
    def __init__(self, token_dim, n_domains, hidden=512, dropout=0.10):
        super().__init__()
        self.n_domains = n_domains

        # Token projection
        self.token_proj = nn.Linear(token_dim, hidden)
        self.token_ln = nn.LayerNorm(hidden)

        # Domain-specific queries
        self.domain_queries = nn.Parameter(torch.randn(n_domains, hidden) * 0.02)
        self.global_query = nn.Parameter(torch.randn(hidden) * 0.02)

        # Normalization
        self.ctx_ln = nn.LayerNorm(hidden)
        self.global_ln = nn.LayerNorm(hidden)

        # Fusion MLP
        self.ff = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # SIGMOID output head — each domain independent
        self.out = nn.Linear(hidden, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def masked_softmax(self, scores, mask, dim=-1):
        scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=dim)

    def forward(self, token_states, attention_mask):
        """Returns raw logits per domain — apply sigmoid outside."""
        x = self.token_proj(token_states.float())
        x = self.token_ln(x)

        # Global context via attention
        g_scores = torch.einsum("bsh,h->bs", x, self.global_query)
        g_attn = self.masked_softmax(g_scores, attention_mask, dim=-1)
        g_ctx = torch.einsum("bs,bsh->bh", g_attn, x)
        g_ctx = self.global_ln(g_ctx)

        # Per-domain context via cross-attention
        d_scores = torch.einsum("bsh,dh->bds", x, self.domain_queries)
        d_mask = attention_mask.unsqueeze(1).expand(-1, d_scores.size(1), -1)
        d_attn = self.masked_softmax(d_scores, d_mask, dim=-1)
        d_ctx = torch.einsum("bds,bsh->bdh", d_attn, x)
        d_ctx = self.ctx_ln(d_ctx)

        # Fuse global + domain context
        g_expanded = g_ctx.unsqueeze(1).expand(-1, d_ctx.size(1), -1)
        fused = torch.cat([d_ctx, g_expanded], dim=-1)
        fused = self.ff(fused)

        # Per-domain logits
        temperature = torch.exp(self.log_temperature).clamp(min=0.3, max=3.0)
        logits = self.out(fused).squeeze(-1) / temperature
        return logits

    def predict(self, token_states, attention_mask):
        """Returns sigmoid probabilities per domain."""
        return torch.sigmoid(self.forward(token_states, attention_mask))


# ════════════════════════════════════════════════════════════════════════
#  SEMANTIC ENCODING — deep features for router input
# ════════════════════════════════════════════════════════════════════════

def encode_semantic_batch(model, tokenizer, prompts, device, seq_len=ROUTER_SEQ_LEN):
    """Get mid+last hidden states as deep semantic features for routing."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=seq_len, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids)

    set_router_base_only(model, True)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        set_router_base_only(model, False)

    hs = out.hidden_states
    mid = hs[len(hs) // 2].float()
    last = hs[-1].float()
    return 0.45 * mid + 0.55 * last, attention_mask


# ════════════════════════════════════════════════════════════════════════
#  DATA BUILDING — content-only prompts + external mixed-domain
# ════════════════════════════════════════════════════════════════════════

def extract_prompt_answer(record, fmt):
    """Extract (prompt_text, answer_text) from a dataset record."""
    if fmt == "alpaca":
        inst = safe_text(record.get("instruction", record.get("prompt", record.get("question", ""))))
        inp = safe_text(record.get("input", record.get("context", "")))
        out = safe_text(record.get("output", record.get("response", record.get("answer", ""))))
        prompt = f"{inst}\n{inp}".strip() if inp else inst
        return prompt, out
    elif fmt == "gsm8k":
        return safe_text(record.get("question", "")), safe_text(record.get("answer", ""))
    elif fmt == "gsm8k_reasoning":
        q = safe_text(record.get("question", ""))
        a = safe_text(record.get("answer", ""))
        return f"Think step by step: {q}", a
    elif fmt == "ultrafeedback":
        chosen = record.get("chosen", [])
        if isinstance(chosen, list) and len(chosen) >= 2:
            user_text, asst_text = None, None
            for msg in chosen:
                if isinstance(msg, dict):
                    role = safe_text(msg.get("role", "")).lower()
                    content = safe_text(msg.get("content", ""))
                    if role == "user" and not user_text: user_text = content
                    elif role == "assistant" and user_text and not asst_text: asst_text = content; break
            if user_text and asst_text:
                return user_text, asst_text
        return "", ""
    elif fmt == "openthoughts":
        msgs = record.get("conversations", record.get("messages", []))
        user_text, asst_text = "", ""
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict):
                    role = safe_text(m.get("role", m.get("from", ""))).lower()
                    content = safe_text(m.get("content", m.get("value", "")))
                    if role in ("user", "human") and not user_text:
                        user_text = content
                    elif role in ("assistant", "gpt") and user_text and not asst_text:
                        asst_text = content
                        break
        return user_text, asst_text
    elif fmt == "logiqa":
        ctx = safe_text(record.get("context", ""))
        query = safe_text(record.get("query", record.get("question", "")))
        opts = record.get("options", [])
        q = ctx + "\n" + query
        if isinstance(opts, list) and opts:
            q += "\n" + "\n".join(f"{chr(65+i)}. {safe_text(o)}" for i, o in enumerate(opts))
        # correct_option is int 0-3, return the actual answer text
        correct_idx = record.get("correct_option", 0)
        if isinstance(correct_idx, int) and isinstance(opts, list) and correct_idx < len(opts):
            answer = f"{chr(65+correct_idx)}. {safe_text(opts[correct_idx])}"
        else:
            answer = ""
        return q, answer
    return "", ""


def extract_mixed_prompt(row, kind):
    """Extract prompts from external mixed-domain datasets."""
    if kind == "messages_mathcode":
        msgs = row.get("messages", row.get("conversations", []))
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and str(m.get("role", "")).lower() == "user":
                    c = m.get("content", "")
                    if isinstance(c, list):
                        return "\n".join(safe_text(i.get("content", "")) if isinstance(i, dict) else safe_text(i) for i in c)
                    return safe_text(c)
    elif kind == "medical_reasoning":
        q = safe_text(row.get("Question", row.get("question", "")))
        return q
    elif kind in ("alpaca", "alpaca_code"):
        inst = safe_text(row.get("instruction", row.get("prompt", "")))
        inp = safe_text(row.get("input", row.get("context", "")))
        return f"{inst}\n{inp}".strip() if inp else inst
    elif kind == "gsm8k":
        return safe_text(row.get("question", ""))
    elif kind == "gsm8k_reasoning":
        q = safe_text(row.get("question", ""))
        return f"Think step by step and explain your reasoning: {q}" if q else ""
    elif kind == "openthoughts_chat":
        msgs = row.get("conversations", row.get("messages", []))
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict):
                    role = safe_text(m.get("role", m.get("from", ""))).lower()
                    content = safe_text(m.get("content", m.get("value", "")))
                    if role in ("user", "human") and content:
                        return content
        return ""
    elif kind == "logiqa":
        q = safe_text(row.get("context", "")) + "\n" + safe_text(row.get("query", row.get("question", "")))
        opts = row.get("options", [])
        if isinstance(opts, list) and opts:
            q += "\n" + "\n".join(f"{chr(65+i)}. {safe_text(o)}" for i, o in enumerate(opts))
        return q
    elif kind == "aqua_rat":
        q = safe_text(row.get("question", ""))
        opts = row.get("options", [])
        if isinstance(opts, list) and opts:
            q += "\n" + "\n".join(safe_text(o) for o in opts)
        return q
    elif kind == "codeforces_reasoning":
        return safe_text(row.get("description", ""))
    return ""


def build_all_examples(domain_names):
    """Build single-domain + mixed-domain examples with prompt+answer pairs."""
    examples = []
    n_domains = len(domain_names)

    # Single-domain
    print(f"\n  [Data] Building: {SAMPLES_PER_DOMAIN}/domain × {n_domains} domains")
    for domain in domain_names:
        if domain not in DOMAIN_DATA:
            print(f"    [Skip] {domain}: no dataset configured")
            continue
        cfg = DOMAIN_DATA[domain]
        print(f"    Loading {domain}...", end=" ", flush=True)
        c = cfg.get("config")
        rev = cfg.get("revision")
        if c:
            ds = load_dataset(cfg["dataset"], c, split=cfg["split"], revision=rev) if rev else load_dataset(cfg["dataset"], c, split=cfg["split"])
        else:
            ds = load_dataset(cfg["dataset"], split=cfg["split"], revision=rev) if rev else load_dataset(cfg["dataset"], split=cfg["split"])
        rows = list(ds); random.shuffle(rows)
        count = 0
        for row in rows:
            if count >= SAMPLES_PER_DOMAIN: break
            prompt, answer = extract_prompt_answer(row, cfg["format"])
            if not prompt or len(prompt) < 15: continue
            target = torch.zeros(n_domains)
            target[domain_names.index(domain)] = 1.0
            examples.append({
                "prompt": prompt, "answer": answer if answer and len(answer) > 5 else None,
                "target": target, "kind": "single", "domains": [domain],
            })
            count += 1
        print(f"{count}")

    # Mixed-domain from external datasets
    print("  [Data] Loading external mixed-domain prompts...")
    for src in EXTERNAL_MIXED_SOURCES:
        d1, d2 = src["pair"]
        if d1 not in domain_names or d2 not in domain_names: continue
        i1, i2 = domain_names.index(d1), domain_names.index(d2)
        c = src.get("config")
        for attempt in range(3):
            try:
                rev = src.get("revision")
                if c:
                    ds = load_dataset(src["dataset"], c, split=src["split"], streaming=True, revision=rev) if rev else load_dataset(src["dataset"], c, split=src["split"], streaming=True)
                else:
                    ds = load_dataset(src["dataset"], split=src["split"], streaming=True, revision=rev) if rev else load_dataset(src["dataset"], split=src["split"], streaming=True)
                ds = ds.shuffle(seed=42, buffer_size=src.get("buffer_size", 2000))
                added = 0
                for row in ds:
                    if added >= src.get("take", 400): break
                    prompt = extract_mixed_prompt(row, src["kind"])
                    if len(prompt) < 15: continue
                    target = torch.zeros(n_domains)
                    target[i1] = 1.0; target[i2] = 1.0  # Multi-hot for cross-domain
                    examples.append({
                        "prompt": prompt, "answer": None,
                        "target": target, "kind": "mixed", "domains": [d1, d2],
                    })
                    added += 1
                print(f"    {d1}+{d2}: {added} from {src['dataset']}")
                break  # success
            except Exception as e:
                if attempt < 2:
                    print(f"    [Retry {attempt+1}/3] {d1}+{d2}: {e}")
                    time.sleep(5)
                else:
                    print(f"    [Skip] {d1}+{d2}: {e}")

    print(f"  [Data] Total: {len(examples)}")
    return examples


# ════════════════════════════════════════════════════════════════════════
#  OUTCOME-BASED TARGET DISCOVERY 
# ════════════════════════════════════════════════════════════════════════
#
# Instead of "this prompt is medical" (label) or "medical stack reduced
# loss by 0.3" (single-domain utility), we discover:
#
#   "For THIS prompt, the combination chat+medical+math produced the
#    lowest loss. Therefore the ideal routing is [1.0, 0.0, 0.8, 1.0, 0.0]"
#
# We use GREEDY combo search to keep cost at O(n²) per prompt:
#   1. Start with base-only loss
#   2. Try adding each domain alone → keep the best
#   3. Try adding a second domain on top → keep if it helps
#   4. Continue until no domain helps
#   5. The discovered combo = training target

def build_lm_batch(tokenizer, prompts, answers, device, max_length=256):
    """Build input_ids + labels for loss computation."""
    prompt_texts = [ALPACA_PROMPT.format(p, "", "") for p in prompts]
    full_texts = [ALPACA_PROMPT.format(p, "", a) for p, a in zip(prompts, answers)]
    prompt_enc = tokenizer(prompt_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
    full_enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length)
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids)
    labels = input_ids.clone()
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
    for i in range(labels.size(0)):
        labels[i, :int(prompt_lens[i].item())] = -100
        labels[i, full_enc["attention_mask"][i] == 0] = -100
    return input_ids, attention_mask, token_type_ids, labels


def compute_loss_with_routing(model, input_ids, attention_mask, token_type_ids,
                              labels, device, weight_vec=None, base_only=False):
    """Compute per-example CE loss with a specific routing configuration."""
    clear_domain_weights(model)
    set_router_base_only(model, base_only)
    if not base_only and weight_vec is not None:
        set_domain_weights(model, weight_vec)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        use_cache=False, return_dict=True)
    finally:
        clear_domain_weights(model)
        set_router_base_only(model, False)

    logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1),
                              ignore_index=-100, reduction="none").view(shift_labels.size())
    mask = (shift_labels != -100).float()
    return (per_tok * mask).sum(1) / mask.sum(1).clamp_min(1.0)


def discover_optimal_routing(model, tokenizer, examples, domain_names, device):
    """
    OUTCOME-BASED: For each prompt+answer, greedily discover which domain
    combination produces the lowest loss. This is the router's training signal.

    CHECKPOINTED: saves per-domain results to disk after each domain.
    If cache exists, loads from disk and skips recomputation.
    """
    n_domains = len(domain_names)
    os.makedirs(ORACLE_CACHE_DIR, exist_ok=True)
    print(f"\n  [Outcome] Discovering optimal routing for {OUTCOME_SAMPLES_PER_DOMAIN}×{n_domains} samples...")
    print(f"  [Outcome] Cache dir: {ORACLE_CACHE_DIR}")

    # Select examples with answers for teacher measurement
    by_domain = {d: [] for d in domain_names}
    for ex in examples:
        if ex["kind"] == "single" and ex["answer"] is not None:
            by_domain[ex["domains"][0]].append(ex)

    # Build per-domain pools
    domain_pools = {}
    for d in domain_names:
        chosen = by_domain[d][:OUTCOME_SAMPLES_PER_DOMAIN]
        domain_pools[d] = chosen
        print(f"    {d}: {len(chosen)} samples for outcome discovery")

    if not any(domain_pools.values()):
        print("  [Outcome] No samples with answers. Falling back to label targets.")
        return

    # Process each domain with checkpoint save/load
    total_updated = 0
    batch_size = 8

    for domain in domain_names:
        pool = domain_pools[domain]
        if not pool:
            continue

        cache_path = os.path.join(ORACLE_CACHE_DIR, f"oracle_{domain}.pt")

        # Check for cached results
        if os.path.exists(cache_path) and domain not in ORACLE_RECOMPUTE_DOMAINS:
            print(f"\n  [Cache ✓] Loading {domain} oracle from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Apply cached targets back to examples
            cache_map = {entry["prompt"]: entry["target"] for entry in cached}
            applied = 0
            for ex in pool:
                if ex["prompt"] in cache_map:
                    ex["target"] = cache_map[ex["prompt"]]
                    applied += 1
            print(f"  [Cache ✓] Applied {applied}/{len(pool)} cached targets for {domain}")
            total_updated += applied
            continue

        # Compute fresh oracle for this domain
        print(f"\n  [Compute] Running oracle discovery for {domain} ({len(pool)} samples)...")
        domain_results = []  # will save to disk
        domain_updated = 0

        for start in range(0, len(pool), batch_size):
            batch = pool[start:start + batch_size]
            prompts = [x["prompt"] for x in batch]
            answers = [x["answer"] for x in batch]
            input_ids, attn_mask, token_type_ids, labels = build_lm_batch(
                tokenizer, prompts, answers, device)

            # Step 1: Base-only loss
            base_loss = compute_loss_with_routing(
                model, input_ids, attn_mask, token_type_ids, labels, device, base_only=True)

            # Step 2: Single-domain losses
            single_losses = {}
            eye = torch.eye(n_domains, dtype=COMPUTE_DTYPE, device=device)
            for d_idx in range(n_domains):
                w = eye[d_idx]
                if "chat" in domain_names:
                    chat_idx = domain_names.index("chat")
                    w = w.clone()
                    w[chat_idx] = max(float(w[chat_idx]), CHAT_FLOOR)
                losses = compute_loss_with_routing(
                    model, input_ids, attn_mask, token_type_ids, labels, device, weight_vec=w)
                single_losses[d_idx] = losses

            # Step 3: Greedy combo search per example
            for i, ex in enumerate(batch):
                b_loss = base_loss[i].item()

                best_combo = set()
                best_loss = b_loss

                for d_idx in range(n_domains):
                    d_loss = single_losses[d_idx][i].item()
                    if d_loss < best_loss:
                        best_loss = d_loss
                        best_combo = {d_idx}

                improved = True
                while improved and len(best_combo) < n_domains:
                    improved = False
                    remaining = set(range(n_domains)) - best_combo
                    for d_idx in remaining:
                        test_combo = best_combo | {d_idx}
                        w = torch.zeros(n_domains, dtype=COMPUTE_DTYPE, device=device)
                        for di in test_combo:
                            w[di] = 1.0
                        if "chat" in domain_names:
                            chat_idx = domain_names.index("chat")
                            w[chat_idx] = max(float(w[chat_idx]), CHAT_FLOOR)

                        combo_loss = compute_loss_with_routing(
                            model, input_ids, attn_mask, token_type_ids, labels, device, weight_vec=w)
                        c_loss = combo_loss[i].item()

                        if c_loss < best_loss - 0.01:  # kept at 0.01
                            best_loss = c_loss
                            best_combo = test_combo
                            improved = True

                # Reasoning soft-boost
                # Reasoning never wins greedy race (subtle gains). After greedy,
                # check if reasoning helps AT ALL — if so, soft target 0.5.
                reasoning_idx = domain_names.index("reasoning") if "reasoning" in domain_names else -1
                _reasoning_soft = False
                if reasoning_idx >= 0 and reasoning_idx not in best_combo:
                    test_with_r = best_combo | {reasoning_idx}
                    w = torch.zeros(n_domains, dtype=COMPUTE_DTYPE, device=device)
                    for di in test_with_r:
                        w[di] = 1.0
                    if "chat" in domain_names:
                        chat_idx = domain_names.index("chat")
                        w[chat_idx] = max(float(w[chat_idx]), CHAT_FLOOR)
                    r_loss = compute_loss_with_routing(
                        model, input_ids, attn_mask, token_type_ids, labels, device, weight_vec=w)
                    if r_loss[i].item() < best_loss:  # ANY improvement
                        best_combo = test_with_r
                        best_loss = r_loss[i].item()
                        _reasoning_soft = True

                target = torch.zeros(n_domains)
                for d_idx in best_combo:
                    if d_idx == reasoning_idx and _reasoning_soft:
                        target[d_idx] = 0.5  # soft — not full 1.0
                    else:
                        target[d_idx] = 1.0
                if "chat" in domain_names:
                    chat_idx = domain_names.index("chat")
                    target[chat_idx] = max(float(target[chat_idx]), CHAT_FLOOR)

                prior = ex["target"].clone()
                ex["target"] = (0.80 * target + 0.20 * prior).clamp(0, 1)
                domain_updated += 1
                total_updated += 1

                # Save for checkpoint
                domain_results.append({
                    "prompt": ex["prompt"],
                    "target": ex["target"].clone(),
                    "base_loss": b_loss,
                    "best_loss": best_loss,
                    "route": sorted(best_combo),
                })

                combo_names = [domain_names[d] for d in sorted(best_combo)]
                gain = b_loss - best_loss
                print(f"    [{total_updated:>3}] [{ex['domains'][0]:>10}] "
                      f"base={b_loss:.3f} → best={best_loss:.3f} (Δ={gain:+.3f}) "
                      f"route={combo_names}")

            if (start // batch_size) % 10 == 0 and start > 0:
                print(f"    ... {domain_updated}/{len(pool)} {domain} samples processed")

        # Save domain checkpoint to disk
        torch.save(domain_results, cache_path)
        print(f"  [Save ✓] {domain}: {domain_updated} targets → {cache_path} "
              f"({os.path.getsize(cache_path)/1024:.1f} KB)")

    print(f"\n  [Outcome] Total: {total_updated} targets updated across {n_domains} domains")


# ════════════════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════════════════

def router_loss(logits, targets):
    """BCE + confidence margin penalty."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    # Confidence penalty: push toward clear yes/no, not mushy middle
    probs = torch.sigmoid(logits)
    confidence = (probs - 0.5).abs()
    margin_penalty = (0.3 - confidence).clamp_min(0).mean()

    return bce + 0.05 * margin_penalty


def split_no_leakage(examples, val_frac=0.10):
    """Split by unique prompts to prevent leakage."""
    grouped = {}
    for ex in examples:
        grouped.setdefault(ex["prompt"].strip()[:100], []).append(ex)
    keys = list(grouped.keys())
    random.Random(42).shuffle(keys)
    n_val = max(int(len(keys) * val_frac), 200)
    val_keys = set(keys[:n_val])
    train, val = [], []
    for k, group in grouped.items():
        (val if k in val_keys else train).extend(group)
    return train, val


def evaluate_router(router, model, tokenizer, val_examples, device, domain_names):
    """Evaluate router on validation set."""
    router.eval()
    total_bce, n_b = 0.0, 0
    single_correct, single_total = 0, 0
    mixed_match, mixed_total = 0, 0

    with torch.no_grad():
        for i in range(0, len(val_examples), ROUTER_BATCH_SIZE):
            batch = val_examples[i:i + ROUTER_BATCH_SIZE]
            prompts = [x["prompt"] for x in batch]
            target = torch.stack([x["target"] for x in batch]).to(device)
            states, mask = encode_semantic_batch(model, tokenizer, prompts, device)
            logits = router(states.to(device), mask.to(device))
            total_bce += F.binary_cross_entropy_with_logits(logits, target).item()
            n_b += 1

            probs = torch.sigmoid(logits)
            for j, ex in enumerate(batch):
                if ex["kind"] == "single":
                    pred = int(torch.argmax(probs[j]).item())
                    true = int(torch.argmax(target[j]).item())
                    single_correct += int(pred == true)
                    single_total += 1
                else:
                    pred_top2 = set(torch.topk(probs[j], k=2).indices.cpu().tolist())
                    true_active = set(torch.nonzero(target[j] > 0.3, as_tuple=False).view(-1).cpu().tolist())
                    if true_active and pred_top2 == true_active:
                        mixed_match += 1
                    mixed_total += 1

    return {
        "val_bce": total_bce / max(n_b, 1),
        "single_top1": single_correct / max(single_total, 1),
        "mixed_set": mixed_match / max(mixed_total, 1),
    }


def train_meta_router(model, tokenizer, domain_names, device):
    """Full training pipeline: data → outcome discovery → train router."""
    # 1. Build examples
    examples = build_all_examples(domain_names)

    # 2. OUTCOME-BASED: discover optimal routing combos
    discover_optimal_routing(model, tokenizer, examples, domain_names, device)

    # 3. Split
    train_ex, val_ex = split_no_leakage(examples)
    print(f"  [Split] train={len(train_ex)}  val={len(val_ex)}")

    # 4. Create router
    token_dim = get_embed_dim(model)
    router = MetaRouter(token_dim=token_dim, n_domains=len(domain_names)).to(device)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"\n  [Router] {n_params/1e6:.3f}M params  |  token_dim={token_dim}  |  domains={domain_names}")

    # 5. Train
    opt = torch.optim.AdamW(router.parameters(), lr=ROUTER_LR, weight_decay=ROUTER_WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ROUTER_EPOCHS)
    best_score, best_state = -1e9, None
    t0 = time.time()

    for epoch in range(ROUTER_EPOCHS):
        router.train()
        random.shuffle(train_ex)
        total_loss, n_b = 0.0, 0

        for i in range(0, len(train_ex), ROUTER_BATCH_SIZE):
            batch = train_ex[i:i + ROUTER_BATCH_SIZE]
            prompts = [x["prompt"] for x in batch]
            target = torch.stack([x["target"] for x in batch]).to(device)
            states, mask = encode_semantic_batch(model, tokenizer, prompts, device)
            logits = router(states.to(device), mask.to(device))
            loss = router_loss(logits, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_b += 1

        sch.step()
        m = evaluate_router(router, model, tokenizer, val_ex, device, domain_names)
        score = 0.50 * m["single_top1"] + 0.35 * m["mixed_set"] - 0.15 * m["val_bce"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}

        print(f"    Epoch {epoch+1:>3}/{ROUTER_EPOCHS}  loss={total_loss/max(n_b,1):.4f}  "
              f"bce={m['val_bce']:.4f}  single={m['single_top1']:.4f}  mixed={m['mixed_set']:.4f}")

    if best_state: router.load_state_dict(best_state)
    print(f"\n  [Router] Best score: {best_score:.4f}  |  {time.time()-t0:.0f}s")

    # 6. Save
    save_path = Path(SAVE_DIR) / "meta_router.pt"
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": router.state_dict(),
        "domain_names": domain_names,
        "token_dim": token_dim,
        "n_domains": len(domain_names),
        "version": "reasoning_aware",
        "chat_floor": CHAT_FLOOR,
        "gate_threshold": GATE_THRESHOLD,
    }, save_path)
    print(f"  [Save] {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)")
    return router


# ════════════════════════════════════════════════════════════════════════
#  ROUTING + GENERATION
# ════════════════════════════════════════════════════════════════════════

def route_prompt(router, model, tokenizer, prompt, device, domain_names):
    """Route on RAW prompt → returns per-domain weights."""
    states, mask = encode_semantic_batch(model, tokenizer, [prompt], device, seq_len=ROUTER_SEQ_LEN)
    with torch.no_grad():
        probs = router.predict(states.to(device), mask.to(device)).squeeze(0)

    # Chat floor — always keep chat active for formatting
    if "chat" in domain_names:
        chat_idx = domain_names.index("chat")
        probs[chat_idx] = torch.max(probs[chat_idx], torch.tensor(CHAT_FLOOR, device=probs.device))

    active = [f"{domain_names[i]}={float(probs[i]):.2f}"
              for i in range(len(domain_names)) if float(probs[i]) >= GATE_THRESHOLD]
    route_str = ", ".join(active) if active else "all domains"

    return probs.to(device=device, dtype=COMPUTE_DTYPE), route_str


def routed_generate(model, router, tokenizer, prompt, device, domain_names, max_tokens=GEN_MAX_TOKENS):
    """Generate with routing applied."""
    weights, route_str = route_prompt(router, model, tokenizer, prompt, device, domain_names)
    set_domain_weights(model, weights)

    full = ALPACA_PROMPT.format(prompt, "", "") + "\n"
    ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    token_type_ids = torch.zeros_like(ids)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                             repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                             token_type_ids=token_type_ids)

    resp = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
    clear_domain_weights(model)
    return resp, route_str


def ungated_generate(model, tokenizer, prompt, device, max_tokens=GEN_MAX_TOKENS):
    """Generate without routing — all stacks fire."""
    clear_domain_weights(model)

    full = ALPACA_PROMPT.format(prompt, "", "") + "\n"
    ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    token_type_ids = torch.zeros_like(ids)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                             repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                             token_type_ids=token_type_ids)

    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BrainStacks Meta-Router — Reasoning-Aware Outcome Router                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  OUTCOME-BASED: discovers optimal domain combos from actual loss         ║
║  Deep features (mid+last hidden) + domain cross-attention                ║
║  Sigmoid output (true cross-domain) + chat floor ({CHAT_FLOOR})          ║
║  Greedy combo search + BCE loss + confidence margin                      ║
║  Base:   {MODEL_NAME:<62}║
║  Device: {str(device):<62}║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # 1. Load model
    print("  [1/5] Loading Gemma 3 12B IT (4-bit) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=COMPUTE_DTYPE,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 2. Inject + load stacks
    print("  [2/5] Injecting StackedMoELoRALayer ...")
    model, stacked_layers = inject_stacked_layers(model)
    model._hf_peft_config_loaded = True

    print("\n  [3/5] Loading domain stacks from manifest ...")
    domain_names, stacks_per_domain = load_all_stacks_from_manifest(model, stacked_layers, device)
    if not domain_names:
        print("  [Error] No stacks found. Run SFT first."); return

    # 3. Train router
    print(f"\n  [4/5] Training outcome-based meta-router...")
    router = train_meta_router(model, tokenizer, domain_names, device)

    # 4. Compare: ungated vs routed 
    test_prompts = [
        "Explain what a neural network is in simple terms.",
        "Write a Python function to check if a number is prime.",
        "What are the common symptoms of influenza?",
        "If a train travels 120km in 2 hours, what is its speed?",
        "Write a Python function to calculate BMI given weight and height.",
        "Explain the difference between type 1 and type 2 diabetes.",
        "Write a Python class for a stack data structure with push, pop, and peek.",
        "A patient needs 500mg of medication per day split into 3 doses. How many mg per dose?",
        "How does gravity work?",
        "Write Python code to calculate medication dosage based on patient weight.",
        "Explain how to solve 3x + 7 = 22 step by step.",
        "What is the chain rule in calculus and why is it important?",
        # reasoning-targeted prompts (standalone + cross-domain)
        "If all mammals produce milk and all whales are mammals, prove that whales produce milk.",
        "Think step by step: A farmer has 17 sheep. All but 9 die. How many are left?",
        "Prove by contradiction that the square root of 2 is irrational.",
        "A 72-year-old diabetic with eGFR 28 is on metformin. Explain the clinical reasoning for why renal function affects drug safety.",
    ]

    print(f"\n{'='*75}")
    print(f"  [5/5] GENERATION: UNGATED vs ROUTED")
    print(f"{'='*75}")

    for prompt in test_prompts:
        resp_u = ungated_generate(model, tokenizer, prompt, device)
        resp_r, route = routed_generate(model, router, tokenizer, prompt, device, domain_names)
        print(f"\n  > {prompt[:70]}...")
        print(f"  Route: [{route}]")
        print(f"  UNGATED: {resp_u[:250]}")
        print(f"  ROUTED:  {resp_r[:250]}")

    # Router decision summary
    print(f"\n{'='*75}")
    print(f"  ROUTER DECISIONS (outcome-based sigmoid)")
    print(f"{'='*75}")

    for prompt in test_prompts:
        weights, route = route_prompt(router, model, tokenizer, prompt, device, domain_names)
        decisions = {domain_names[i]: f"{float(weights[i].float()):.3f}" for i in range(len(domain_names))}
        active = [f"{k}={v}" for k, v in decisions.items() if float(v) > GATE_THRESHOLD]
        print(f"  {prompt[:60]}...")
        print(f"    Active: {', '.join(active)}")
        print(f"    All:    {decisions}")

    print(f"\n{'='*75}")
    print(f"  DONE. Router saved to: {SAVE_DIR}/meta_router.pt")
    print(f"{'='*75}")

    return model, router, domain_names


if __name__ == "__main__":
    main()