"""
BrainStacks Evaluation
======================
BENCHMARKS:

General: 1.HellaSwag 2.ARC-Easy 3.ARC-Challenge 4.TruthfulQA 5.MMLU
Code: 6.HumanEval
Math: 7.GSM8K 8.MATH-500   9.AIME-2024
Medical: 10. MedQA 11. MedMCQA
Science: 12. GPQA-Diamond

=======================
MODES (runs all 3 automatically):

A. Base model only (no stacks)
B. Base + all stacks UNGATED (all fire)
C. Base + stacks + meta-router (selective gating)

=======================

Author: Mohammad R. Abu Ayyash — Brains Build Research, Palestine

"""

import os, sys, json, math, re, warnings, gc, time, subprocess, tempfile, random
from pathlib import Path
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ════════════════════════════════════════════════════════════════════════
#  ★★★ CONFIGURE THESE IN YOUR COLAB CELL ★★★
# ════════════════════════════════════════════════════════════════════════

LIMIT           = 200          # samples per benchmark (None = full, 200 = quick test)
SAVE_DIR        = "./BrainStacks_gemma3"  # where stacks are saved (for loading and evaluation)
ROUTER_PATH     = "./BrainStacks_gemma3/meta_router.pt"  # set None to skip routed mode
EVAL_OUTPUT_DIR = "./BrainStacks_gemma3/evaluation"
SKIP_BENCHMARKS = [] # e.g. ["humaneval"] to skip slow ones
#SKIP_BENCHMARKS = ["hellaswag", "arc_easy", "arc_challenge", "truthfulqa", "mmlu", "humaneval", "gsm8k", "medqa", "medmcqa", "math500", "aime2024", "gpqa_diamond"]
RUN_BASE_ONLY   = False         # test base model without any stacks
RUN_UNGATED     = False         # test with all stacks firing
RUN_ROUTED      = True         # test with meta-router (if available)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bitsandbytes as bnb

try:
    from tqdm import tqdm
except ImportError:
    os.system(f"{sys.executable} -m pip install -q tqdm")
    from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_PLT = True
except ImportError:
    os.system(f"{sys.executable} -m pip install -q matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_PLT = True

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

# Config matching SFT script
MODEL_NAME     = "google/gemma-3-12b-it"
MAX_SEQ_LEN    = 512
LORA_R         = 16
LORA_ALPHA     = 16.0
USE_RSLORA     = True
NUM_EXPERTS    = 4
TOP_K          = 2
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CHAT_FLOOR     = 0.20
GATE_THRESHOLD = 0.12
_LORA_SCALE    = LORA_ALPHA / math.sqrt(LORA_R) if USE_RSLORA else LORA_ALPHA / LORA_R


# ════════════════════════════════════════════════════════════════════════
#  MOE-LORA COMPONENTS — exact match with SFT script
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
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        sparse = torch.full_like(logits, float('-inf'))
        sparse.scatter_(-1, topk_idx, topk_vals)
        gates = F.softmax(sparse, dim=-1)
        self.aux_loss = torch.tensor(0.0, device=flat.device)
        A = torch.stack([e.A.weight for e in self.experts])
        B = torch.stack([e.B.weight for e in self.experts])
        mid = torch.einsum("tf,erf->ter", flat, A)
        all_deltas = torch.einsum("ter,eor->teo", mid, B) * _LORA_SCALE
        delta = (gates.unsqueeze(-1) * all_deltas).sum(dim=1)
        return delta.view(*x.shape[:-1], -1)

class StackedMoELoRALayer(nn.Module):
    def __init__(self, frozen_linear):
        super().__init__()
        self.frozen = frozen_linear
        self.frozen_stacks = nn.ModuleList()
        self.active_stack = None
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
        if self._router_base_only:
            return out
        if self.frozen_stacks:
            with torch.no_grad(), torch.amp.autocast(x.device.type, enabled=x.is_cuda):
                if self._domain_weights is not None and self._domain_stack_counts is not None:
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
                    for stack in self.frozen_stacks:
                        was_cpu = not next(stack.parameters()).is_cuda
                        if was_cpu: stack.to(x.device)
                        out = out + stack(x).to(dtype=out_dtype)
                        if was_cpu: stack.cpu()
        if self.active_stack is not None:
            out = out + self.active_stack(x).to(dtype=out_dtype)
        return out

# ════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
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
    for layer in stacked_layers:
        if layer.active_stack is not None:
            for p in layer.active_stack.parameters(): p.requires_grad_(False)
            layer.active_stack.half()
            layer.active_stack.cpu()
            layer.frozen_stacks.append(layer.active_stack)
            layer.active_stack = None

def load_model_and_stacks(device):
    """Base Model + all stacks from manifest."""
    print(f"[Model] Loading {MODEL_NAME} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map="auto", attn_implementation="sdpa", torch_dtype=COMPUTE_DTYPE,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("[Model] Injecting StackedMoELoRALayer ...")
    model, stacked_layers = inject_stacked_layers(model)
    model._hf_peft_config_loaded = True

    # Load stacks from manifest
    manifest_path = Path(SAVE_DIR) / "manifest.json"
    domain_names = []
    stacks_per_domain = {}

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for block in manifest["domains"]:
            name = block["name"]
            stack_files = [sf for sf in block["stack_files"] if os.path.exists(sf)]
            if not stack_files: continue
            for sf in stack_files:
                load_stack_as_frozen(model, stacked_layers, sf, device)
            stacks_per_domain[name] = len(stack_files)
            domain_names.append(name)
            print(f"  [Load] {name}: {len(stack_files)} stacks")

        counts = [stacks_per_domain.get(d, 0) for d in domain_names]
        for layer in stacked_layers:
            layer._domain_stack_counts = counts
    else:
        print("  [Warn] No manifest found — running base model only")

    n = sum(len(l.frozen_stacks) for l in stacked_layers) // max(len(stacked_layers), 1)
    print(f"  [Ready] {n} stacks/layer  |  Domains: {domain_names}")

    model.eval()
    return model, tokenizer, stacked_layers, domain_names

def set_domain_weights(model, weights):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer): m._domain_weights = weights

def clear_domain_weights(model):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer): m._domain_weights = None

def set_base_only(model, flag):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer): m._router_base_only = flag

def disable_all_stacks(model):
    """Temporarily disable stacks for base-only evaluation."""
    set_base_only(model, True)

def enable_all_stacks(model):
    """Re-enable stacks (ungated — all fire)."""
    set_base_only(model, False)
    clear_domain_weights(model)

# ════════════════════════════════════════════════════════════════════════
#  META-ROUTER LOADING
# ════════════════════════════════════════════════════════════════════════

class MetaRouter(nn.Module):
    """Meta-router for dynamic stack gating. Exact architecture match with SFT script."""
    def __init__(self, token_dim, n_domains, hidden=512, dropout=0.10):
        super().__init__()
        self.n_domains = n_domains
        self.token_proj = nn.Linear(token_dim, hidden)
        self.token_ln = nn.LayerNorm(hidden)
        self.domain_queries = nn.Parameter(torch.randn(n_domains, hidden) * 0.02)
        self.global_query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.ctx_ln = nn.LayerNorm(hidden)
        self.global_ln = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def masked_softmax(self, scores, mask, dim=-1):
        scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=dim)

    def forward(self, token_states, attention_mask):
        x = self.token_proj(token_states.float())
        x = self.token_ln(x)
        g_scores = torch.einsum("bsh,h->bs", x, self.global_query)
        g_attn = self.masked_softmax(g_scores, attention_mask, dim=-1)
        g_ctx = torch.einsum("bs,bsh->bh", g_attn, x)
        g_ctx = self.global_ln(g_ctx)
        d_scores = torch.einsum("bsh,dh->bds", x, self.domain_queries)
        d_mask = attention_mask.unsqueeze(1).expand(-1, d_scores.size(1), -1)
        d_attn = self.masked_softmax(d_scores, d_mask, dim=-1)
        d_ctx = torch.einsum("bds,bsh->bdh", d_attn, x)
        d_ctx = self.ctx_ln(d_ctx)
        g_expanded = g_ctx.unsqueeze(1).expand(-1, d_ctx.size(1), -1)
        fused = torch.cat([d_ctx, g_expanded], dim=-1)
        fused = self.ff(fused)
        temperature = torch.exp(self.log_temperature).clamp(min=0.3, max=3.0)
        return self.out(fused).squeeze(-1) / temperature

    def predict(self, token_states, attention_mask):
        return torch.sigmoid(self.forward(token_states, attention_mask))

def load_meta_router(device):
    """Load trained meta-router if available."""
    if ROUTER_PATH is None or not os.path.exists(ROUTER_PATH):
        print("  [Router] No meta-router found — skipping routed mode")
        return None

    data = torch.load(ROUTER_PATH, map_location=device, weights_only=False)
    router = MetaRouter(
        token_dim=data["token_dim"],
        n_domains=data["n_domains"],
    ).to(device)
    router.load_state_dict(data["state_dict"])
    router.eval()
    print(f"  [Router] Loaded {ROUTER_PATH}  |  v={data.get('version', '?')}  |  domains={data.get('domain_names', '?')}")
    return router

def route_and_set_weights(model, router, tokenizer, prompt, device, domain_names):
    """Route a prompt through the meta-router and set domain weights."""
    enc = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True,
                    max_length=96, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids)

    set_base_only(model, True)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True, use_cache=False, return_dict=True)
    set_base_only(model, False)

    hs = out.hidden_states
    states = (0.45 * hs[len(hs)//2].float() + 0.55 * hs[-1].float())

    with torch.no_grad():
        probs = router.predict(states, attention_mask).squeeze(0)

    if "chat" in domain_names:
        chat_idx = domain_names.index("chat")
        probs[chat_idx] = torch.max(probs[chat_idx], torch.tensor(CHAT_FLOOR, device=device))

    set_domain_weights(model, probs.to(dtype=COMPUTE_DTYPE))

# ════════════════════════════════════════════════════════════════════════
#  GLOBAL ROUTING CONTEXT — auto-routes every model call when active
# ════════════════════════════════════════════════════════════════════════

_ROUTING_CTX = {
    "active": False,
    "router": None,
    "model": None,
    "tokenizer": None,
    "domain_names": [],
    "device": None,
    "last_prompt": None,
}

def activate_routing(router, model, tokenizer, domain_names, device):
    _ROUTING_CTX.update(active=True, router=router, model=model,
                        tokenizer=tokenizer, domain_names=domain_names,
                        device=device, last_prompt=None)

def deactivate_routing():
    _ROUTING_CTX.update(active=False, last_prompt=None)
    if _ROUTING_CTX["model"]:
        clear_domain_weights(_ROUTING_CTX["model"])

def maybe_route(prompt_text):
    """If routing is active, route this prompt and set domain weights.
    Caches — won't re-route if same prompt as last call."""
    ctx = _ROUTING_CTX
    if not ctx["active"] or ctx["router"] is None:
        return
    if prompt_text == ctx["last_prompt"]:
        return
    route_and_set_weights(ctx["model"], ctx["router"], ctx["tokenizer"],
                          prompt_text, ctx["device"], ctx["domain_names"])
    ctx["last_prompt"] = prompt_text


# ════════════════════════════════════════════════════════════════════════
#  SCORING HELPERS — auto-route via maybe_route() when routed mode is on
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_loglikelihood(model, tokenizer, context, continuation, device):
    """Compute log-likelihood of continuation given context. Auto-routes if active."""
    maybe_route(context)  # routes on the question, not the choice
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=True).input_ids
    cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = torch.cat([ctx_ids, cont_ids], dim=1).to(device)
    attention_mask = torch.ones_like(full_ids)
    token_type_ids = torch.zeros_like(full_ids)

    if full_ids.shape[1] > MAX_SEQ_LEN:
        full_ids = full_ids[:, -MAX_SEQ_LEN:]
        attention_mask = attention_mask[:, -MAX_SEQ_LEN:]
        token_type_ids = token_type_ids[:, -MAX_SEQ_LEN:]

    with torch.amp.autocast("cuda"):
        outputs = model(full_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    logits = logits[0]

    ctx_len = ctx_ids.shape[1]
    if full_ids.shape[1] < ctx_ids.shape[1] + cont_ids.shape[1]:
        ctx_len = full_ids.shape[1] - cont_ids.shape[1]
    ctx_len = max(ctx_len, 0)

    log_probs = F.log_softmax(logits, dim=-1)
    total, count = 0.0, 0
    for i in range(ctx_len, full_ids.shape[1] - 1):
        total += log_probs[i, full_ids[0, i+1]].item()
        count += 1
    return total / max(count, 1)

@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_tokens=256):
    """Generate text with Base Model compatibility. Auto-routes if active."""
    maybe_route(prompt)  # routes GSM8K/HumanEval prompts too
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(device)
    token_type_ids = torch.zeros_like(ids)
    with torch.amp.autocast("cuda"):
        out = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2, token_type_ids=token_type_ids)
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

def mc_accuracy_streaming(model, tokenizer, dataset_name, config, split,
                          format_fn, device, limit, label):
    """Stream dataset → score → delete. Returns (accuracy, n_samples)."""
    print(f"\n  [{label}] Loading (streaming) ...")
    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"  [Skip] {dataset_name}: {e}")
        return None, 0

    correct, total = 0, 0
    pbar = tqdm(desc=f"  {label}", total=limit, ncols=80)

    for item in ds:
        if limit and total >= limit:
            break
        result = format_fn(item)
        if result is None:
            continue
        context, choices, answer_idx = result
        if context is None:
            continue

        best_score, best_idx = float("-inf"), 0
        for j, choice in enumerate(choices):
            score = compute_loglikelihood(model, tokenizer, context, choice, device)
            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx == answer_idx:
            correct += 1
        total += 1
        pbar.update(1)
        pbar.set_postfix(acc=f"{correct/max(total,1):.3f}")

    pbar.close()

    acc = correct / max(total, 1)
    print(f"  {label}: {acc:.4f} ({total} samples)")

    # Clean up
    del ds; gc.collect()
    return acc, total

# ════════════════════════════════════════════════════════════════════════
#  BENCHMARK DEFINITIONS
# ════════════════════════════════════════════════════════════════════════

def bench_hellaswag(model, tokenizer, device, limit):
    def fmt(item):
        return item["ctx"], item["endings"], int(item["label"])
    acc, n = mc_accuracy_streaming(
        model, tokenizer, "Rowan/hellaswag", None, "validation", fmt, device, limit, "HellaSwag")
    return {"benchmark": "HellaSwag", "metric": "accuracy", "score": acc, "n": n}


def bench_arc_easy(model, tokenizer, device, limit):
    def fmt(item):
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        try: idx = labels.index(item["answerKey"])
        except ValueError: return None
        return f"Question: {item['question']}\nAnswer:", [f" {c}" for c in choices], idx
    acc, n = mc_accuracy_streaming(
        model, tokenizer, "allenai/ai2_arc", "ARC-Easy", "test", fmt, device, limit, "ARC-Easy")
    return {"benchmark": "ARC-Easy", "metric": "accuracy", "score": acc, "n": n}


def bench_arc_challenge(model, tokenizer, device, limit):
    def fmt(item):
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        try: idx = labels.index(item["answerKey"])
        except ValueError: return None
        return f"Question: {item['question']}\nAnswer:", [f" {c}" for c in choices], idx
    acc, n = mc_accuracy_streaming(
        model, tokenizer, "allenai/ai2_arc", "ARC-Challenge", "test", fmt, device, limit, "ARC-Challenge")
    return {"benchmark": "ARC-Challenge", "metric": "accuracy", "score": acc, "n": n}


def bench_truthfulqa(model, tokenizer, device, limit):
    """TruthfulQA MC1 — pick the single best truthful answer."""
    def fmt(item):
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]
        idx = labels.index(1) if 1 in labels else 0
        return f"Q: {item['question']}\nA:", [f" {c}" for c in choices], idx
    # Try original dataset first (parquet-based, works with modern datasets)
    acc, n = mc_accuracy_streaming(
        model, tokenizer, "truthfulqa/truthful_qa", "multiple_choice", "validation",
        fmt, device, limit, "TruthfulQA")
    return {"benchmark": "TruthfulQA", "metric": "accuracy", "score": acc, "n": n}


def bench_mmlu(model, tokenizer, device, limit):
    """MMLU — massive multitask language understanding."""
    LABELS = ["A", "B", "C", "D"]
    def fmt(item):
        q = item["question"]
        choices = item["choices"]           # cais/mmlu: list of 4 strings
        answer = int(item["answer"])        # cais/mmlu: int 0-3
        ctx = f"Question: {q}\n"
        for i, c in enumerate(choices):
            ctx += f"{LABELS[i]}. {c}\n"
        ctx += "Answer:"
        return ctx, [f" {choices[i]}" for i in range(len(choices))], answer

    acc, n = mc_accuracy_streaming(
        model, tokenizer, "cais/mmlu", "all", "test", fmt, device, limit, "MMLU")
    return {"benchmark": "MMLU", "metric": "accuracy", "score": acc, "n": n}

def extract_number(text):
    match = re.findall(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match[-1].replace(",", "").strip()
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "").strip() if numbers else ""


def bench_gsm8k(model, tokenizer, device, limit):
    """GSM8K — math word problems (exact match)."""
    print(f"\n  [GSM8K] Loading (streaming) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    correct, total = 0, 0
    pbar = tqdm(desc="  GSM8K", total=limit, ncols=80)

    for item in ds:
        if limit and total >= limit: break
        question = item["question"]
        gold = extract_number(item["answer"])
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        response = generate_text(model, tokenizer, prompt, device, max_tokens=256)
        pred = extract_number(response)
        try:
            if gold and pred and abs(float(gold) - float(pred)) < 0.01:
                correct += 1
        except ValueError: pass
        total += 1
        pbar.update(1)
        pbar.set_postfix(em=f"{correct/max(total,1):.3f}")

    pbar.close()
    acc = correct / max(total, 1)
    print(f"  GSM8K exact match: {acc:.4f} ({total} samples)")
    del ds; gc.collect()
    return {"benchmark": "GSM8K", "metric": "exact_match", "score": acc, "n": total}


def bench_humaneval(model, tokenizer, device, limit):
    """HumanEval — code completion (pass@1)."""
    print(f"\n  [HumanEval] Loading ...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit: ds = ds.select(range(min(limit, len(ds))))

    passed, total = 0, 0
    pbar = tqdm(desc="  HumanEval", total=len(ds), ncols=80)

    for item in ds:
        prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]

        # IT-model instruction prompt
        instruct_prompt = (
            f"Complete the following Python function. "
            f"Output ONLY the function body code (indented with 4 spaces), nothing else. "
            f"Do not repeat the function signature or docstring.\n\n{prompt}"
        )
        raw = generate_text(model, tokenizer, instruct_prompt, device, max_tokens=512)

        # Clean IT model output
        raw = raw.replace("<end_of_turn>", "").replace("<eos>", "")

        # Strategy 1: Extract from markdown code fences
        completion = ""
        if "```" in raw:
            blocks = raw.split("```")
            for block in blocks[1::2]:
                blines = block.strip().split("\n")
                # Strip language tag
                if blines and blines[0].strip().lower() in ("python", "python3", "py", ""):
                    blines = blines[1:]
                candidate = "\n".join(blines)
                if candidate.strip():
                    completion = candidate
                    break

        # Strategy 2: If no fences, use raw output
        if not completion.strip():
            completion = raw

        # Remove re-declared function signature + docstring
        # The IT model often repeats the full function
        prompt_lines = prompt.strip().split("\n")
        func_sig = prompt_lines[0].strip() if prompt_lines else ""  # e.g. "def has_close_elements(..."

        comp_lines = completion.split("\n")
        # Find and skip re-declared signature
        for ci, cline in enumerate(comp_lines):
            stripped = cline.strip()
            if stripped.startswith("def ") and ci < 5:
                # Skip the signature line
                comp_lines = comp_lines[ci+1:]
                # Skip docstring if present
                in_docstring = False
                while comp_lines:
                    dl = comp_lines[0]
                    if '"""' in dl or "'''" in dl:
                        if in_docstring:
                            comp_lines.pop(0)
                            break
                        elif dl.strip().count('"""') >= 2 or dl.strip().count("'''") >= 2:
                            comp_lines.pop(0)
                            break
                        else:
                            in_docstring = True
                            comp_lines.pop(0)
                    elif in_docstring:
                        comp_lines.pop(0)
                    else:
                        break
                break
        completion = "\n".join(comp_lines)

        # Trim at next top-level function/class
        lines = completion.split("\n")
        clean_lines = []
        for li, line in enumerate(lines):
            if li > 0 and line.strip().startswith("def ") and not line.startswith("    "):
                break
            if li > 0 and line.strip().startswith("class "):
                break
            clean_lines.append(line)
        completion_clean = "\n".join(clean_lines)

        # Ensure first line is indented (model drops leading indent on line 1)
        body_lines = completion_clean.split("\n")
        if body_lines and body_lines[0] and not body_lines[0].startswith((" ", "\t")):
            body_lines[0] = "    " + body_lines[0]
            completion_clean = "\n".join(body_lines)

        full_code = prompt + completion_clean + "\n\n" + test_code + f"\ncheck({entry_point})\n"
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(full_code); tmp_path = f.name
            result = subprocess.run(["python", tmp_path], capture_output=True, timeout=10, text=True)
            if result.returncode == 0: passed += 1
        except: pass
        finally:
            try: os.unlink(tmp_path)
            except: pass
        total += 1
        pbar.update(1)
        pbar.set_postfix(p1=f"{passed/max(total,1):.3f}")

    pbar.close()
    acc = passed / max(total, 1)
    print(f"  HumanEval pass@1: {acc:.4f} ({total} samples)")
    del ds; gc.collect()
    return {"benchmark": "HumanEval", "metric": "pass@1", "score": acc, "n": total}


def bench_medqa(model, tokenizer, device, limit):
    LABELS = ["A", "B", "C", "D", "E"]
    def fmt(item):
        if "question" not in item or "options" not in item: return None
        q = item["question"]
        opts = item["options"]
        if isinstance(opts, dict):
            choices, keys = list(opts.values()), list(opts.keys())
        elif isinstance(opts, list):
            choices, keys = opts, [str(i) for i in range(len(opts))]
        else: return None
        answer = item.get("answer_idx", item.get("answer", None))
        if answer is None: return None
        if isinstance(answer, str):
            try: answer_idx = keys.index(answer)
            except ValueError:
                try: answer_idx = choices.index(answer)
                except ValueError: return None
        else: answer_idx = int(answer)
        ctx = f"Question: {q}\n"
        for i, c in enumerate(choices):
            ctx += f"{LABELS[i]}. {c}\n"
        ctx += "Answer:"
        return ctx, [f" {choices[i]}" for i in range(len(choices))], answer_idx

    acc, n = mc_accuracy_streaming(
        model, tokenizer, "GBaker/MedQA-USMLE-4-options", None, "test",
        fmt, device, limit, "MedQA")
    return {"benchmark": "MedQA", "metric": "accuracy", "score": acc, "n": n}


def bench_medmcqa(model, tokenizer, device, limit):
    LABELS = ["A", "B", "C", "D"]
    def fmt(item):
        q = item["question"]
        choices = [item["opa"], item["opb"], item["opc"], item["opd"]]
        idx = int(item["cop"]) if item["cop"] is not None else 0
        ctx = f"Question: {q}\n"
        for i, c in enumerate(choices):
            ctx += f"{LABELS[i]}. {c}\n"
        ctx += "Answer:"
        return ctx, [f" {choices[i]}" for i in range(len(choices))], idx

    acc, n = mc_accuracy_streaming(
        model, tokenizer, "openlifescienceai/medmcqa", None, "validation",
        fmt, device, limit, "MedMCQA")
    return {"benchmark": "MedMCQA", "metric": "accuracy", "score": acc, "n": n}


def extract_boxed(text):
    """Extract answer from \\boxed{...} in LaTeX output."""
    # Try \boxed{...} first
    matches = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_math_answer(ans):
    """Normalize LaTeX answer for comparison."""
    if not ans:
        return ""
    ans = ans.strip()
    ans = ans.replace(" ", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace("\\{", "{").replace("\\}", "}")
    ans = ans.replace("\\%", "%")
    ans = ans.replace("$", "")
    ans = ans.replace("\\text{", "").replace("\\mathrm{", "")
    # Remove trailing }
    while ans.endswith("}") and ans.count("{") < ans.count("}"):
        ans = ans[:-1]
    return ans


def math_equiv(pred, gold):
    """Check mathematical equivalence: string match, numeric, then sympy."""
    if not pred or not gold:
        return False
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    # 1. Exact string match
    if p == g:
        return True
    # 2. Numeric comparison
    try:
        if abs(float(p) - float(g)) < 0.001:
            return True
    except (ValueError, TypeError):
        pass
    # 3. Sympy equivalence
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, Rational
        p_expr = parse_latex(pred)
        g_expr = parse_latex(gold)
        if simplify(p_expr - g_expr) == 0:
            return True
    except Exception:
        pass
    return False


MATH500_PROMPT = """Solve the following math problem step by step. Present the final answer as \\boxed{{x}}, where x is the fully simplified solution.

Example:
**Question:** What is the value of $\\frac{{3}}{{4}} + \\frac{{1}}{{4}}$?
**Solution:** $\\frac{{3}}{{4}} + \\frac{{1}}{{4}} = \\frac{{4}}{{4}} = 1$
\\boxed{{1}}

Now solve:
{problem}
"""


def bench_math500(model, tokenizer, device, limit):
    """MATH-500 — competition math (LaTeX exact match + sympy equivalence)."""
    print(f"\n  [MATH-500] Loading ...")
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as e:
        print(f"  [Skip] MATH-500: {e}")
        return {"benchmark": "MATH-500", "metric": "exact_match", "score": 0.0, "n": 0}
    if limit: ds = ds.select(range(min(limit, len(ds))))

    correct, total = 0, 0
    pbar = tqdm(desc="  MATH-500", total=len(ds), ncols=80)

    for item in ds:
        problem = item["problem"]
        gold = item.get("answer", "")
        prompt = MATH500_PROMPT.format(problem=problem)
        response = generate_text(model, tokenizer, prompt, device, max_tokens=512)
        pred_boxed = extract_boxed(response)
        if pred_boxed and math_equiv(pred_boxed, gold):
            correct += 1
        total += 1
        pbar.update(1)
        pbar.set_postfix(em=f"{correct/max(total,1):.3f}")

    pbar.close()
    acc = correct / max(total, 1)
    print(f"  MATH-500 exact match: {acc:.4f} ({total} samples)")
    del ds; gc.collect()
    return {"benchmark": "MATH-500", "metric": "exact_match", "score": acc, "n": total}


def bench_aime2024(model, tokenizer, device, limit):
    """AIME 2024 — olympiad math (integer 0-999 exact match)."""
    print(f"\n  [AIME-2024] Loading ...")
    try:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    except Exception as e:
        print(f"  [Skip] AIME-2024: {e}")
        return {"benchmark": "AIME-2024", "metric": "exact_match", "score": 0.0, "n": 0}
    if limit: ds = ds.select(range(min(limit, len(ds))))

    correct, total = 0, 0
    pbar = tqdm(desc="  AIME-2024", total=len(ds), ncols=80)

    for item in ds:
        problem = item["Problem"]
        gold = str(item["Answer"]).strip()
        prompt = (
            "Solve the following AIME problem step by step. "
            "AIME answers are always integers from 000 to 999. "
            "Present your final answer as \\boxed{N} where N is the integer.\n\n"
            f"Problem: {problem}\n\nSolution:\n"
        )
        response = generate_text(model, tokenizer, prompt, device, max_tokens=512)
        # Try boxed first
        pred_boxed = extract_boxed(response)
        pred = ""
        if pred_boxed:
            nums = re.findall(r'\d+', pred_boxed)
            pred = nums[0] if nums else ""
        # Fallback: find "answer is X" pattern
        if not pred:
            m = re.search(r'(?:answer|Answer|ANSWER)\s*(?:is|=|:)\s*(\d+)', response)
            if m: pred = m.group(1)
        # Fallback: last integer in response
        if not pred:
            nums = re.findall(r'\b(\d{1,3})\b', response)
            pred = nums[-1] if nums else ""
        try:
            if pred and int(pred) == int(gold):
                correct += 1
        except (ValueError, TypeError):
            pass
        total += 1
        pbar.update(1)
        pbar.set_postfix(em=f"{correct/max(total,1):.3f}")

    pbar.close()
    acc = correct / max(total, 1)
    print(f"  AIME-2024 exact match: {acc:.4f} ({total} samples)")
    del ds; gc.collect()
    return {"benchmark": "AIME-2024", "metric": "exact_match", "score": acc, "n": total}


def bench_gpqa_diamond(model, tokenizer, device, limit):
    """GPQA Diamond — graduate-level science MCQ (generation + CoT)."""
    print(f"\n  [GPQA-Diamond] Loading ...")
    try:
        ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    except Exception:
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train",
                              trust_remote_code=True)
        except Exception as e:
            print(f"  [Skip] GPQA-Diamond: {e}")
            return {"benchmark": "GPQA-Diamond", "metric": "accuracy", "score": 0.0, "n": 0}
    if limit: ds = ds.select(range(min(limit, len(ds))))

    LABELS = ["A", "B", "C", "D"]
    correct, total = 0, 0
    pbar = tqdm(desc="  GPQA-Diamond", total=len(ds), ncols=80)

    for item in ds:
        q = item.get("Question", item.get("question", ""))
        correct_ans = item.get("Correct Answer", item.get("correct_answer", ""))
        wrong = [
            item.get("Incorrect Answer 1", item.get("incorrect_answer_1", "")),
            item.get("Incorrect Answer 2", item.get("incorrect_answer_2", "")),
            item.get("Incorrect Answer 3", item.get("incorrect_answer_3", "")),
        ]
        if not q or not correct_ans:
            continue

        # Shuffle choices deterministically
        choices = [correct_ans] + wrong
        random.seed(hash(q) & 0xFFFFFFFF)
        indices = list(range(4))
        random.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        correct_letter = LABELS[indices.index(0)]

        prompt = f"Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
        prompt += f"Question: {q}\n"
        for i, c in enumerate(shuffled):
            prompt += f"({LABELS[i]}) {c}\n"

        response = generate_text(model, tokenizer, prompt, device, max_tokens=512)

        # Extract ANSWER: X from response
        pred_letter = ""
        answer_match = re.findall(r'ANSWER:\s*([A-D])', response, re.IGNORECASE)
        if answer_match:
            pred_letter = answer_match[-1].upper()
        else:
            # Fallback: look for standalone letter at end
            letter_match = re.findall(r'\b([A-D])\b', response[-50:])
            if letter_match:
                pred_letter = letter_match[-1].upper()

        if pred_letter == correct_letter:
            correct += 1
        total += 1
        pbar.update(1)
        pbar.set_postfix(acc=f"{correct/max(total,1):.3f}")

    pbar.close()
    acc = correct / max(total, 1)
    print(f"  GPQA-Diamond: {acc:.4f} ({total} samples)")
    del ds; gc.collect()
    return {"benchmark": "GPQA-Diamond", "metric": "accuracy", "score": acc, "n": total}


# ════════════════════════════════════════════════════════════════════════
#  RUN ALL BENCHMARKS IN ONE MODE
# ════════════════════════════════════════════════════════════════════════

ALL_BENCHMARKS = [
    ("hellaswag",     bench_hellaswag),
    ("arc_easy",      bench_arc_easy),
    ("arc_challenge", bench_arc_challenge),
    ("truthfulqa",    bench_truthfulqa),
    ("mmlu",          bench_mmlu),
    ("humaneval",     bench_humaneval),
    ("gsm8k",         bench_gsm8k),
    ("medqa",         bench_medqa),
    ("medmcqa",       bench_medmcqa),
    ("math500",       bench_math500),
    ("aime2024",      bench_aime2024),
    ("gpqa_diamond",  bench_gpqa_diamond),
]


def run_all_benchmarks(model, tokenizer, device, mode_name):
    """Run all benchmarks for a given mode. Returns list of result dicts."""
    print(f"\n{'━'*65}")
    print(f"  RUNNING: {mode_name}")
    print(f"{'━'*65}")

    results = []
    for name, fn in ALL_BENCHMARKS:
        if name in SKIP_BENCHMARKS:
            print(f"\n  [SKIP] {name}")
            continue
        try:
            t0 = time.time()
            result = fn(model, tokenizer, device, LIMIT)
            result["time_sec"] = round(time.time() - t0, 1)
            result["mode"] = mode_name
            results.append(result)
        except Exception as e:
            print(f"  [Error] {name}: {e}")
            import traceback; traceback.print_exc()
            results.append({"benchmark": name, "score": None, "mode": mode_name, "error": str(e)})

        gc.collect()
        torch.cuda.empty_cache()

    return results


# ════════════════════════════════════════════════════════════════════════
#  REFERENCE SCORES  — for context on how BrainStacks compares to Gemma 3 12B IT and 4B PT
# ════════════════════════════════════════════════════════════════════════

REFERENCE = {
    "Gemma 3 12B IT": {
        "HellaSwag": 0.821, "ARC-Easy": None, "ARC-Challenge": None,
        "TruthfulQA": None, "MMLU": 0.754,
        "HumanEval": 0.652, "GSM8K": 0.785,
        "MedQA": None, "MedMCQA": None,
        "MATH-500": None, "AIME-2024": None, "GPQA-Diamond": None,
    },
    "Gemma 3 4B PT": {
        "HellaSwag": 0.772, "ARC-Easy": 0.824, "ARC-Challenge": 0.562,
        "TruthfulQA": None, "MMLU": 0.596,
        "HumanEval": 0.360, "GSM8K": 0.384,
        "MedQA": None, "MedMCQA": None,
        "MATH-500": None, "AIME-2024": None, "GPQA-Diamond": None,
    },
}


# ════════════════════════════════════════════════════════════════════════
#  PLOTTING — Comprehensive comparison charts
# ════════════════════════════════════════════════════════════════════════

def plot_all(all_results, save_dir):
    """Generate all comparison plots."""
    if not HAS_PLT:
        print("  [Skip] matplotlib not available")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Organize by mode
    by_mode = {}
    for r in all_results:
        mode = r.get("mode", "unknown")
        by_mode.setdefault(mode, {})[r["benchmark"]] = r.get("score")

    modes = list(by_mode.keys())
    bench_names = []
    for name, fn in ALL_BENCHMARKS:
        if name not in SKIP_BENCHMARKS:
            for r in all_results:
                if r.get("benchmark", "").lower().replace("-", "_").replace(" ", "_") == name or \
                   r.get("benchmark", "") in [name]:
                    bench_names.append(r["benchmark"])
                    break
            else:
                bench_names.append(name)

    # De-duplicate bench_names
    seen = set()
    unique_bench_names = []
    for b in bench_names:
        if b not in seen:
            seen.add(b)
            unique_bench_names.append(b)
    bench_names = unique_bench_names

    colors = {"Base Only": "#95a5a6", "Ungated (all stacks)": "#e74c3c", "Routed (meta-router)": "#27ae60"}

    # Plot 1: Grouped bar chart — all modes side by side
    fig, ax = plt.subplots(figsize=(max(14, len(bench_names)*2), 7))
    n_modes = len(modes)
    width = 0.8 / max(n_modes, 1)
    x = np.arange(len(bench_names))

    for i, mode in enumerate(modes):
        scores = [by_mode[mode].get(b, 0) or 0 for b in bench_names]
        offset = (i - n_modes/2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=mode,
                      color=colors.get(mode, f"C{i}"), alpha=0.85, edgecolor="white")
        for bar, s in zip(bars, scores):
            if s and s > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{s:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Add Gemma 3 12B reference line
    ref = REFERENCE.get("Gemma 3 12B IT", {})
    ref_scores = [ref.get(b) for b in bench_names]
    for j, rs in enumerate(ref_scores):
        if rs is not None:
            ax.plot([j - 0.5, j + 0.5], [rs, rs], color="gold", linewidth=2, linestyle="--", alpha=0.8)
    ax.plot([], [], color="gold", linewidth=2, linestyle="--", label="Gemma 3 12B IT (Google)")

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("BrainStacks Evaluation — All Modes Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_all_modes.png", dpi=150)
    plt.close()
    print(f"  [Plot] comparison_all_modes.png")

    # Plot 2: Domain-specific radar chart
    domain_groups = {
        "General": ["HellaSwag", "ARC-Easy", "ARC-Challenge", "TruthfulQA", "MMLU"],
        "Code": ["HumanEval"],
        "Math": ["GSM8K"],
        "Medical": ["MedQA", "MedMCQA"],
    }

    fig, axes = plt.subplots(1, len(domain_groups), figsize=(5*len(domain_groups), 5))
    if len(domain_groups) == 1: axes = [axes]

    for ax, (group_name, group_benchmarks) in zip(axes, domain_groups.items()):
        present = [b for b in group_benchmarks if b in bench_names]
        if not present: continue

        x_pos = np.arange(len(present))
        for i, mode in enumerate(modes):
            scores = [by_mode[mode].get(b, 0) or 0 for b in present]
            ax.bar(x_pos + i*width, scores, width, label=mode if ax == axes[0] else "",
                   color=colors.get(mode, f"C{i}"), alpha=0.85)

        ax.set_xticks(x_pos + width*(n_modes-1)/2)
        ax.set_xticklabels(present, rotation=25, ha="right", fontsize=9)
        ax.set_title(group_name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_by_domain.png", dpi=150)
    plt.close()
    print(f"  [Plot] comparison_by_domain.png")

    # Plot 3: Improvement over base
    if "Base Only" in by_mode and len(modes) > 1:
        fig, ax = plt.subplots(figsize=(max(12, len(bench_names)*1.5), 6))
        base_scores = by_mode["Base Only"]

        for i, mode in enumerate(modes):
            if mode == "Base Only": continue
            improvements = []
            for b in bench_names:
                base_s = base_scores.get(b, 0) or 0
                mode_s = (by_mode[mode].get(b, 0) or 0)
                improvements.append(mode_s - base_s)

            color = colors.get(mode, f"C{i}")
            bars = ax.bar(x + (i-1)*width, improvements, width, label=mode,
                          color=color, alpha=0.85, edgecolor="white")
            for bar, imp in zip(bars, improvements):
                if abs(imp) > 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + (0.003 if imp > 0 else -0.015),
                            f"{imp:+.3f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bench_names, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Score Change vs Base", fontsize=12)
        ax.set_title("BrainStacks — Improvement Over Base Model", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "improvement_over_base.png", dpi=150)
        plt.close()
        print(f"  [Plot] improvement_over_base.png")

    # Plot 4: Per-benchmark individual charts
    n_bench = min(len(bench_names), 12)
    n_rows = (n_bench + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, bench in enumerate(bench_names[:n_bench]):
        ax = axes[idx]
        scores = []
        labels = []
        clrs = []

        for mode in modes:
            s = by_mode[mode].get(bench, 0) or 0
            scores.append(s)
            labels.append(mode.replace(" (all stacks)", "\n(ungated)").replace(" (meta-router)", "\n(routed)"))
            clrs.append(colors.get(mode, "gray"))

        # Add Google reference
        ref_s = REFERENCE.get("Gemma 3 12B IT", {}).get(bench)
        if ref_s is not None:
            scores.append(ref_s)
            labels.append("Gemma 3\n12B IT")
            clrs.append("gold")

        bars = ax.bar(range(len(scores)), scores, color=clrs, alpha=0.85, edgecolor="white")
        for bar, s in zip(bars, scores):
            if s and s > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{s:.3f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(bench, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(len(bench_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("BrainStacks — Per-Benchmark Breakdown", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_dir / "per_benchmark_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] per_benchmark_breakdown.png")

    # Plot 5: Summary table as image
    fig, ax = plt.subplots(figsize=(max(12, len(modes)*3), max(4, len(bench_names)*0.5)))
    ax.axis("off")

    cell_text = []
    for b in bench_names:
        row = [b]
        for mode in modes:
            s = by_mode[mode].get(b)
            row.append(f"{s:.4f}" if s is not None else "—")
        ref_s = REFERENCE.get("Gemma 3 12B IT", {}).get(b)
        row.append(f"{ref_s:.4f}" if ref_s is not None else "—")
        cell_text.append(row)

    col_labels = ["Benchmark"] + modes + ["Gemma 3 12B IT"]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("BrainStacks Evaluation Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] summary_table.png")


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(EVAL_OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BrainStacks Evaluation — 12 Benchmarks · 3 Modes                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Model     : {MODEL_NAME:<58}║
║  Stacks    : {SAVE_DIR:<58}║
║  Router    : {str(ROUTER_PATH or 'None'):<58}║
║  Limit     : {str(LIMIT or 'full'):<58}║
║  Skip      : {str(SKIP_BENCHMARKS or 'none'):<58}║
║  Modes     : {'base' if RUN_BASE_ONLY else ''} {'ungated' if RUN_UNGATED else ''} {'routed' if RUN_ROUTED else '':<40}║
║  Device    : {str(device):<58}║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load model + stacks
    model, tokenizer, stacked_layers, domain_names = load_model_and_stacks(device)

    # Load router if available
    router = None
    if RUN_ROUTED:
        router = load_meta_router(device)

    all_results = []
    total_t0 = time.time()

    # MODE A: Base only
    if RUN_BASE_ONLY:
        disable_all_stacks(model)
        results = run_all_benchmarks(model, tokenizer, device, "Base Only")
        all_results.extend(results)
        enable_all_stacks(model)

    # MODE B: Ungated (all stacks fire)
    if RUN_UNGATED and domain_names:
        clear_domain_weights(model)
        enable_all_stacks(model)
        results = run_all_benchmarks(model, tokenizer, device, "Ungated (all stacks)")
        all_results.extend(results)

    # MODE C: Routed (meta-router)
    if RUN_ROUTED and router and domain_names:
        print(f"\n  [Note] Routed mode: each prompt auto-routed before scoring/generation")
        enable_all_stacks(model)
        activate_routing(router, model, tokenizer, domain_names, device)
        results = run_all_benchmarks(model, tokenizer, device, "Routed (meta-router)")
        all_results.extend(results)
        deactivate_routing()

    total_time = time.time() - total_t0

    # Save results
    results_path = save_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  [Save] {results_path}")

    # Print summary table
    print(f"\n{'═'*85}")
    print(f"  RESULTS SUMMARY  |  Total time: {total_time/60:.1f} min")
    print(f"{'═'*85}")

    # Collect modes and benchmarks
    by_mode = {}
    for r in all_results:
        mode = r.get("mode", "?")
        by_mode.setdefault(mode, {})[r["benchmark"]] = r.get("score")

    bench_order = ["HellaSwag", "ARC-Easy", "ARC-Challenge", "TruthfulQA", "MMLU",
                   "HumanEval", "GSM8K", "MedQA", "MedMCQA",
                   "MATH-500", "AIME-2024", "GPQA-Diamond"]
    modes_present = list(by_mode.keys())

    header = f"  {'Benchmark':<16}"
    for m in modes_present:
        header += f"  {m:>20}"
    header += f"  {'Gemma3 12B IT':>14}"
    print(header)
    print(f"  {'─'*14}" + "".join(f"  {'─'*20}" for _ in modes_present) + f"  {'─'*14}")

    for bench in bench_order:
        row = f"  {bench:<16}"
        for mode in modes_present:
            s = by_mode[mode].get(bench)
            row += f"  {s:>20.4f}" if s is not None else f"  {'—':>20}"
        ref = REFERENCE.get("Gemma 3 12B IT", {}).get(bench)
        row += f"  {ref:>14.4f}" if ref is not None else f"  {'—':>14}"
        print(row)

    print(f"\n  Results saved to: {results_path}")

    # Generate plots
    print(f"\n{'━'*65}")
    print(f"  GENERATING PLOTS")
    print(f"{'━'*65}")
    plot_all(all_results, save_dir)

    # List all output files
    print(f"\n{'━'*65}")
    print(f"  ALL OUTPUTS IN: {save_dir}/")
    print(f"{'━'*65}")
    for f in sorted(save_dir.iterdir()):
        size = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size:>8.1f} KB")

    print(f"\n  Total evaluation time: {total_time/60:.1f} min\n")

    return all_results

if __name__ == "__main__":
    main()
    
    