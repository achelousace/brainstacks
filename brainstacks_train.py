"""
BRAINSTACKS SFT
==============
Domain A (SFT) → freeze → Domain B (SFT) → freeze → ...
=======================================================
Pipeline per domain:
1. Load & decontaminate data.
2. Compute null space projectors.
3. SFT inner loop (residual boosting).
4. Freeze stacks, forgetting check, save checkpoint.
========================================================
Author: Mohammad R. Abu Ayyash — Brains Build Research, Palestine.
"""

import os, sys, json, math, time, warnings, copy, shutil, gc, re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

warnings.filterwarnings("ignore")

# Installions

# Reduce CUDA memory fragmentation — reclaims reserved-but-unallocated blocks
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _ensure(pkg, install_cmd=None):
    try: __import__(pkg)
    except ImportError:
        cmd = install_cmd or f"{sys.executable} -m pip install -q {pkg}"
        print(f"[Setup] Installing {pkg} ...")
        os.system(cmd)

_ensure("trl",      f"{sys.executable} -m pip install -q trl datasets")
_ensure("datasets", f"{sys.executable} -m pip install -q datasets")

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, concatenate_datasets
import subprocess
from transformers import TrainerCallback

COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

#  CONFIG

@dataclass
class Config:
    model_name: str  = "google/gemma-3-12b-it" # or similar variants
    max_seq_len: int = 512          # Gemma 3 supports 128K, we cap for VRAM

    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_r: int          = 16
    lora_alpha: float    = 16.0
    lora_dropout: float  = 0.0
    use_rslora: bool     = True
    num_experts: int     = 4
    top_k: int           = 2
    load_balance_coeff: float = 0.01

    # Training — Colab G4 96GB optimized for 12B 4-bit + CPU offload
    batch_size: int      = 4          
    grad_accum: int      = 4          # effective batch = 16
    max_steps: int       = 400
    warmup_steps: int    = 40
    lr: float            = 1e-4       # lower LR for larger model
    weight_decay: float  = 0.01
    max_grad_norm: float = 1.0
    eval_every: int      = 50
    packing: bool        = True
    seed: int            = 42
    val_ratio: float     = 0.02

    # Inner loop
    max_inner_rounds: int    = 3
    min_loss_delta: float    = 0.002

    # Null space — scaled for Gemma 3840-dim hidden space
    ns_samples: int      = 400
    ns_top_k_dirs: int   = 64

    # Generation
    gen_prompts: List[str] = field(default_factory=lambda: [
        "Explain what a neural network is in simple terms.",
        "Write a short Python function to reverse a string.",
        "What are the symptoms of type 2 diabetes?",
        "If a train travels 120km in 2 hours, what is its speed?",
        "A patient needs 500mg of medication per day split into 3 doses. How many mg per dose?",
    ])
    max_new_tokens: int = 200

    save_dir: str = "./BrainStacks_gemma3"

CFG = Config()

_LORA_SCALE = (CFG.lora_alpha / math.sqrt(CFG.lora_r)
               if CFG.use_rslora else CFG.lora_alpha / CFG.lora_r)


# ════════════════════════════════════════════════════════════════════════
#  CHAT TEMPLATE TOKEN STRIPPING
# ════════════════════════════════════════════════════════════════════════
# Gemma 3 IT has chat template tokens that MUST be stripped from training data so the model learns domain semantics, not formatting artifacts.

CHAT_TEMPLATE_TOKENS = [
    # Gemma 3 specific
    "<start_of_turn>", "<end_of_turn>",
    "<start_of_conversation>", "<end_of_conversation>",
    # Gemma 2 / generic
    "<bos>", "<eos>", "<pad>",
    # Common in instruction datasets
    "<|im_start|>", "<|im_end|>",
    "<|user|>", "<|assistant|>", "<|system|>",
    "<|begin_of_text|>", "<|end_of_text|>",
    "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
    # Llama-style
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
]

# Compile regex for fast stripping
_TEMPLATE_RE = re.compile(
    "|".join(re.escape(tok) for tok in CHAT_TEMPLATE_TOKENS),
    flags=re.IGNORECASE
)


def strip_chat_tokens(text):
    """Remove ALL chat template tokens from text. Collapse excess whitespace."""
    if not text:
        return ""
    cleaned = _TEMPLATE_RE.sub("", text)
    # Collapse runs of whitespace/newlines left by token removal
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    return cleaned.strip()


# ════════════════════════════════════════════════════════════════════════
#  DOMAINS — 5 domains, high-quality datasets
# ════════════════════════════════════════════════════════════════════════

DOMAINS = [
    # ══════════════════════════════════════════════════════════════
    # DOMAIN 1: CHAT — Instruction following, conversation, alignment
    # Train FIRST: provides output formatting scaffolding
    # that all other domains depend on (the "universal formatter")
    # ══════════════════════════════════════════════════════════════
    {
        "name": "chat",
        "sources": [
            # 1. NVIDIA Nemotron v2 chat split — high quality, messages format
            {"dataset": "nvidia/Nemotron-Post-Training-Dataset-v2",
             "split": "chat",
             "format": "nemotron_v2", "take": 15000},
            # 2. UltraFeedback SFT split — diverse instructions, curated
            {"dataset": "HuggingFaceH4/ultrafeedback_binarized",
             "split": "train_sft",
             "format": "ultrafeedback_sft", "take": 15000},
            # 3. NVIDIA Daring-Anteater — complex multi-constraint
            {"dataset": "nvidia/Daring-Anteater",
             "split": "train",
             "format": "openassistant", "take": 10000},
        ],
        "max_inner_rounds": 2,
        "max_steps": 500,
    },

    # ══════════════════════════════════════════════════════════════
    # DOMAIN 2: CODE — Programming, debugging, explanation
    # Train SECOND: code benefits from chat formatting scaffolding
    # ══════════════════════════════════════════════════════════════
    {
        "name": "code",
        "sources": [
            # 1. Clean Python instruction-output pairs (foundation)
            {"dataset": "iamtarun/python_code_instructions_18k_alpaca",
             "split": "train", "format": "alpaca"},
            # 2. NVIDIA Nemotron v2 code split — multi-language
            {"dataset": "nvidia/Nemotron-Post-Training-Dataset-v2",
             "split": "code",
             "format": "nemotron_v2", "take": 15000},
            # 3. OpenCodeReasoning — code with reasoning traces
            {"dataset": "nvidia/OpenCodeReasoning",
             "split": "split_0",
             "format": "opencodereasoning", "take": 10000},
            # 4. OpenThoughts (code-filtered) — think-then-code
            {"dataset": "open-thoughts/OpenThoughts-114k",
             "split": "train",
             "format": "openthoughts", "take": 5000,
             "filter": "code"},
        ],
        "max_inner_rounds": 2,
        "max_steps": 500,
    },

    # ══════════════════════════════════════════════════════════════
    # DOMAIN 3: MATH — Arithmetic, algebra, word problems, proofs
    # Train THIRD: math benefits from code (computation thinking)
    # and chat (explanation structure)
    # ══════════════════════════════════════════════════════════════
    {
        "name": "math",
        "sources": [
            # 1. GSM8K — clean grade-school math with step-by-step
            {"dataset": "openai/gsm8k", "config": "main",
             "split": "train", "format": "gsm8k"},
            # 2. NVIDIA OpenMathReasoning CoT — AIMO-2 winning traces
            {"dataset": "nvidia/OpenMathReasoning",
             "split": "cot", "format": "openmath_cot", "take": 20000},
            # 3. NuminaMath — competition-level with CoT solutions
            {"dataset": "AI-MO/NuminaMath-CoT",
             "split": "train", "format": "numina_cot", "take": 15000},
            # 4. NVIDIA Nemotron v2 math split
            {"dataset": "nvidia/Nemotron-Post-Training-Dataset-v2",
             "split": "math",
             "format": "nemotron_v2", "take": 10000},
        ],
        "max_inner_rounds": 2,
        "max_steps": 500,
    },

    # ══════════════════════════════════════════════════════════════
    # DOMAIN 4: MEDICAL — Clinical knowledge, diagnosis, pharmacology
    # Train FOURTH: benefits from math (dosage calc), chat (comm),
    # and reasoning (differential dx)
    # ══════════════════════════════════════════════════════════════
    {
        "name": "medical",
        "sources": [
            {"dataset": "GBaker/MedQA-USMLE-4-options",
             "split": "train", "format": "medqa", "take": 10000},
            {"dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
             "config": "en_mix", "split": "train",
             "format": "medical_reasoning", "take": 10000},
            {"dataset": "qiaojin/PubMedQA",
             "config": "pqa_labeled", "split": "train",
             "format": "pubmedqa"},
        ],
        "max_inner_rounds": 2,
        "max_steps": 500,
    },

    # ══════════════════════════════════════════════════════════════
    # DOMAIN 5: REASONING — General CoT, multi-step logic, planning
    # Train LAST: the "meta-skill" that composes all other domains.
    # Teaches <think>...</think> traces across all problem types.
    # ══════════════════════════════════════════════════════════════
    {
        "name": "reasoning",
        "sources": [
            # 1. OpenThoughts — diverse CoT across STEM, logic, common sense
            {"dataset": "open-thoughts/OpenThoughts-114k",
             "split": "train",
             "format": "openthoughts", "take": 20000},
            # 2. NVIDIA Nemotron v2 STEM split — science/STEM reasoning
            {"dataset": "nvidia/Nemotron-Post-Training-Dataset-v2",
             "split": "stem",
             "format": "nemotron_v2", "take": 10000},
            # 3. Sky-T1 — R1-distilled reasoning traces
            {"dataset": "NovaSky-AI/Sky-T1_data_17k",
             "split": "train",
             "format": "sky_t1", "take": 15000},
            # 4. OpenMathReasoning tool_integrated (cross-domain)
            {"dataset": "nvidia/OpenMathReasoning",
             "split": "tool_integrated", "format": "openmath_tool",
             "take": 5000},
        ],
        "max_inner_rounds": 2,
        "max_steps": 600,  # slightly more — reasoning needs more
    },
]

# ════════════════════════════════════════════════════════════════════════
#  DECONTAMINATION — Remove cross-domain leaks
# ════════════════════════════════════════════════════════════════════════

DOMAIN_KEYWORDS = {
    "code": {
        "python", "function", "def ", "import ", "class ", "return ",
        "variable", "loop", "array", "string", "algorithm", "compile",
        "debug", "syntax", "javascript", "html", "css", "sql",
        "code", "programming", "script", "library", "api", "json",
        "for i in", "while ", "if __name__", "print(", ".append(",
    },
    "medical": {
        "symptom", "disease", "patient", "medication", "dose", "diagnosis",
        "treatment", "clinical", "diabetes", "blood", "surgery", "vaccine",
        "infection", "therapy", "prescription", "bmi", "cholesterol",
        "tumor", "cancer", "cardiac", "respiratory", "antibiotic",
        "pharmaceutical", "pathology", "prognosis", "chronic", "acute",
        "influenza", "hypertension", "insulin", "immune",
    },
    "math": {
        "solve", "equation", "calculate", "formula", "algebra", "geometry",
        "derivative", "integral", "probability", "theorem", "proof",
        "fraction", "percentage", "coefficient", "quadratic", "polynomial",
        "logarithm", "trigonometry", "matrix", "vector", "factorial",
        "x =", "y =", "π", "sqrt", "sum of",
    },
}

def detect_domain(text):
    """Detect which specialist domain a text belongs to. Returns domain name or 'chat'."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for k in keywords if k.lower() in text_lower)
        scores[domain] = score
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    return "chat"


def decontaminate_dataset(examples, source_domain, all_domain_names):
    """Remove examples from source_domain that belong to OTHER domains."""
    kept = []
    reassigned = {d: [] for d in all_domain_names if d != source_domain}
    n_moved = 0

    for ex in examples:
        text = ex.get("text", "")
        detected = detect_domain(text)

        if source_domain == "chat" and detected != "chat" and detected in reassigned:
            reassigned[detected].append(ex)
            n_moved += 1
        else:
            kept.append(ex)

    if n_moved > 0:
        print(f"    [Decontam] Moved {n_moved} examples from {source_domain} to other domains")
        for d, exs in reassigned.items():
            if exs:
                print(f"      → {d}: +{len(exs)}")

    return kept, reassigned


# ════════════════════════════════════════════════════════════════════════
#  MOE-LORA COMPONENTS
# ════════════════════════════════════════════════════════════════════════

class LoRAExpert(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.A = nn.Linear(in_f, CFG.lora_r, bias=False)
        self.B = nn.Linear(CFG.lora_r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(self.A(x)) * _LORA_SCALE


class MoELoRADelta(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.num_experts = CFG.num_experts
        self.top_k = CFG.top_k
        self.experts = nn.ModuleList([LoRAExpert(in_f, out_f) for _ in range(CFG.num_experts)])
        self.router = nn.Linear(in_f, CFG.num_experts, bias=False)
        self.noise_linear = nn.Linear(in_f, CFG.num_experts, bias=False)
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
        mid = torch.einsum('tf,erf->ter', flat, A)
        all_deltas = torch.einsum('ter,eor->teo', mid, B) * _LORA_SCALE
        delta = (gates.unsqueeze(-1) * all_deltas).sum(dim=1)
        return delta.view(*x.shape[:-1], -1)


class StackedMoELoRALayer(nn.Module):
    def __init__(self, frozen_linear):
        super().__init__()
        self.frozen = frozen_linear
        self.frozen_stacks = nn.ModuleList()
        self.active_stack = None
        self.null_projector = None

    @property
    def weight(self): return self.frozen.weight
    @property
    def bias(self): return self.frozen.bias

    def forward(self, x):
        out = self.frozen(x)
        if self.frozen_stacks:
            with torch.no_grad(), torch.amp.autocast(x.device.type, enabled=x.is_cuda):
                for stack in self.frozen_stacks:
                    # CPU offload shuttle: bring to GPU → compute → send back
                    was_cpu = not next(stack.parameters()).is_cuda
                    if was_cpu:
                        stack.to(x.device)
                    out = out + stack(x)
                    if was_cpu:
                        stack.cpu()
        if self.active_stack is not None:
            active_delta = self.active_stack(x)
            if self.training and self.null_projector is not None:
                shape = active_delta.shape
                flat = active_delta.reshape(-1, shape[-1])
                flat = flat - flat @ self.null_projector
                active_delta = flat.reshape(shape)
            out = out + active_delta
        return out


# ════════════════════════════════════════════════════════════════════════
#  INJECTION / FREEZE / STACK UTILITIES
# ════════════════════════════════════════════════════════════════════════

def inject_stacked_layers(model):
    stacked_layers = []
    replaced = 0
    for name, mod in list(model.named_modules()):
        for target in CFG.target_modules:
            if name.endswith(target) and isinstance(mod, (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                layer = StackedMoELoRALayer(mod)
                setattr(parent, child_name, layer)
                stacked_layers.append(layer)
                replaced += 1
                break
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"  [Inject] StackedMoELoRALayer into {replaced} layers")
    return model, stacked_layers


def add_new_stack(model, stacked_layers, device):
    for layer in stacked_layers:
        in_f = layer.frozen.in_features
        out_f = layer.frozen.out_features
        stack = MoELoRADelta(in_f, out_f).to(device)
        layer.active_stack = stack
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in stacked_layers:
        if layer.active_stack is not None:
            for p in layer.active_stack.parameters():
                p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(len(l.frozen_stacks) for l in stacked_layers) // max(len(stacked_layers), 1)
    print(f"  [Stack] New active  |  Frozen: {n_frozen}  |  Trainable: {trainable/1e6:.1f}M")


def freeze_active_stack(model, stacked_layers):
    for layer in stacked_layers:
        if layer.active_stack is not None:
            for p in layer.active_stack.parameters():
                p.requires_grad_(False)
            layer.active_stack.half()      # fp16 to halve memory
            layer.active_stack.cpu()       # offload to CPU RAM
            layer.frozen_stacks.append(layer.active_stack)
            layer.active_stack = None
    n_frozen = sum(len(l.frozen_stacks) for l in stacked_layers) // max(len(stacked_layers), 1)
    print(f"  [Stack] Frozen  |  Total stacks/layer: {n_frozen}  |  (offloaded to CPU)")


def get_all_moe_deltas(stacked_layers):
    return [l.active_stack for l in stacked_layers if l.active_stack is not None]


def save_stack(model, stacked_layers, save_path):
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, StackedMoELoRALayer) and mod.active_stack is not None:
            for pname, p in mod.active_stack.named_parameters():
                state[f"{name}.active_stack.{pname}"] = p.data.cpu()
    torch.save(state, save_path)
    print(f"  [Save] {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)")


def load_stack_as_frozen(model, stacked_layers, stack_path, device):
    add_new_stack(model, stacked_layers, device)
    state = torch.load(stack_path, map_location=device, weights_only=False)
    for name, mod in model.named_modules():
        if isinstance(mod, StackedMoELoRALayer) and mod.active_stack is not None:
            for pname, p in mod.active_stack.named_parameters():
                key = f"{name}.active_stack.{pname}"
                if key in state:
                    p.data.copy_(state[key].to(device=device, dtype=p.dtype))
    dtype = next(model.parameters()).dtype
    for layer in stacked_layers:
        if layer.active_stack is not None:
            layer.active_stack.to(dtype=dtype)
    freeze_active_stack(model, stacked_layers)
    print(f"  [Load] {stack_path}")


# ════════════════════════════════════════════════════════════════════════
#  NULL SPACE — Randomized SVD
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_null_projectors(model, stacked_layers, tokenizer, val_ds, device,
                            n_samples=200, top_k_dirs=32):
    print(f"  [NullSpace] {n_samples} samples, top-{top_k_dirs} dirs...")
    model.eval()
    layer_deltas = {i: [] for i in range(len(stacked_layers))}

    for idx in range(min(n_samples, len(val_ds))):
        text = val_ds[idx]["text"]
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=CFG.max_seq_len).to(device)
        hooks = []
        def make_hook(layer_idx):
            def hook_fn(mod, inp, out):
                x = inp[0]
                if mod.frozen_stacks:
                    with torch.amp.autocast(x.device.type, enabled=x.is_cuda):
                        frozen_delta = None
                        for stack in mod.frozen_stacks:
                            was_cpu = not next(stack.parameters()).is_cuda
                            if was_cpu:
                                stack.to(x.device)
                            d = stack(x)
                            frozen_delta = d if frozen_delta is None else frozen_delta + d
                            if was_cpu:
                                stack.cpu()
                else:
                    frozen_delta = torch.zeros_like(mod.frozen(x))
                layer_deltas[layer_idx].append(frozen_delta.mean(dim=1).squeeze(0).cpu())
            return hook_fn
        for i, layer in enumerate(stacked_layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))
        with torch.amp.autocast("cuda"):
            model(enc["input_ids"], token_type_ids=torch.zeros_like(enc["input_ids"]))
        for h in hooks: h.remove()

    for i, layer in enumerate(stacked_layers):
        if not layer_deltas[i] or not layer.frozen_stacks:
            layer.null_projector = None; continue
        D = torch.stack(layer_deltas[i]).float()
        if D.shape[0] > top_k_dirs * 2:
            U, S, V = torch.svd_lowrank(D, q=top_k_dirs)
        else:
            _, _, Vt = torch.linalg.svd(D, full_matrices=False)
            V = Vt[:min(top_k_dirs, Vt.shape[0])].T
        layer.null_projector = (V @ V.T).to(device=device, dtype=next(model.parameters()).dtype)

    n_set = sum(1 for l in stacked_layers if l.null_projector is not None)
    print(f"  [NullSpace] {n_set}/{len(stacked_layers)} layers set")
    model.train()


# ════════════════════════════════════════════════════════════════════════
#  DATA FORMATTING — Multi-format support (ALL chat tokens stripped)
# ════════════════════════════════════════════════════════════════════════

# We use Alpaca prompt format for ALL training data regardless of source.
# This gives the model ONE consistent format to learn, and the router will later operate on raw content before wrapping.

ALPACA_PROMPT = """\
Below is an instruction that describes a task, paired with an input that \
provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def safe_text(x):
    """Safely convert to string, strip whitespace."""
    return str(x).strip() if x is not None else ""


def _extract_messages_pair(row, msg_key="messages"):
    """
    Extract first user+assistant turn from a messages-format dataset.
    Works for: Nemotron v2, UltraFeedback, OpenAssistant, Sky-T1, etc.

    Messages format: [{"role": "user"|"assistant"|..., "content": "..."}]
    Some datasets use "from"/"value" instead of "role"/"content".
    Some have nested content (list of dicts with "type"/"text").
    """
    msgs = row.get(msg_key, row.get("conversations", row.get("messages", [])))
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None, None

    user_text, asst_text = None, None
    for msg in msgs:
        if not isinstance(msg, dict):
            continue

        role = safe_text(msg.get("role", msg.get("from", ""))).lower()
        content = msg.get("content", msg.get("value", ""))

        # Handle nested content (Nemotron v2 sometimes has [{type, text}])
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    parts.append(safe_text(c.get("text", c.get("content", ""))))
                else:
                    parts.append(safe_text(c))
            content = "\n".join(p for p in parts if p)
        else:
            content = safe_text(content)

        # Strip chat template tokens from content
        content = strip_chat_tokens(content)

        if role in ("user", "human") and not user_text:
            user_text = content
        elif role in ("assistant", "gpt", "model") and user_text and not asst_text:
            asst_text = content
            break

    return user_text, asst_text


def _load_dataset_safe(dataset_name, config=None, split="train", take=None):
    """Load dataset. Uses streaming for large datasets to avoid downloading everything."""
    try:
        if take and take < 50000:
            if config:
                ds = load_dataset(dataset_name, config, split=split, streaming=True)
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True)
            rows = []
            for row in ds:
                rows.append(row)
                if len(rows) >= take * 2:  # grab 2x for shuffle margin
                    break
            from datasets import Dataset as HFDataset
            if not rows: return None
            keys = rows[0].keys()
            return HFDataset.from_dict({k: [r[k] for r in rows] for k in keys})
        else:
            if config:
                return load_dataset(dataset_name, config, split=split)
            return load_dataset(dataset_name, split=split)
    except Exception as e:
        for fallback in ["train", "test", "validation"]:
            if fallback == split: continue
            try:
                print(f"    [Fallback] {dataset_name}: trying '{fallback}'")
                if config:
                    return load_dataset(dataset_name, config, split=fallback)
                return load_dataset(dataset_name, split=fallback)
            except: continue
        print(f"    [Skip] {dataset_name}: {e}")
        return None


def format_dataset(ds, fmt, tokenizer, take=None, filter_tag=None):
    """
    Convert any dataset format → HF Dataset with 'text' column.
    ALL text is stripped of chat template tokens before Alpaca wrapping.
    """
    EOS = tokenizer.eos_token
    if ds is None: return None

    if take and take < len(ds):
        import random
        indices = list(range(len(ds)))
        random.shuffle(indices)
        ds = ds.select(indices[:take])

    rows = list(ds)
    texts = []

    # Alpaca format (medalpaca flashcards, python_code_instructions) 
    if fmt == "alpaca":
        inst_col = next((c for c in ["instruction", "prompt", "question"] if c in ds.column_names), None)
        inp_col = next((c for c in ["input", "context"] if c in ds.column_names), None)
        out_col = next((c for c in ["output", "response", "answer"] if c in ds.column_names), None)
        if inst_col and out_col:
            for r in rows:
                inst = strip_chat_tokens(safe_text(r.get(inst_col, "")))
                inp = strip_chat_tokens(safe_text(r.get(inp_col, ""))) if inp_col else ""
                out = strip_chat_tokens(safe_text(r.get(out_col, "")))
                if inst and out:
                    texts.append(ALPACA_PROMPT.format(inst, inp, out) + EOS)

    # GSM8K (question / answer)
    elif fmt == "gsm8k":
        for r in rows:
            q = strip_chat_tokens(safe_text(r.get("question", "")))
            a = strip_chat_tokens(safe_text(r.get("answer", "")))
            if q and a:
                texts.append(ALPACA_PROMPT.format(q, "", a) + EOS)

    # Nemotron v2 — messages format (chat/code/math/stem splits)
    elif fmt == "nemotron_v2":
        for r in rows:
            user_text, asst_text = _extract_messages_pair(r)
            if user_text and asst_text and len(user_text) > 10 and len(asst_text) > 10:
                texts.append(ALPACA_PROMPT.format(user_text, "", asst_text[:2000]) + EOS)

    # UltraFeedback SFT (chosen messages)
    elif fmt == "ultrafeedback_sft":
        for r in rows:
            # UltraFeedback binarized has "chosen" and "rejected"
            chosen = r.get("chosen", [])
            if isinstance(chosen, list) and len(chosen) >= 2:
                user_text, asst_text = None, None
                for msg in chosen:
                    if isinstance(msg, dict):
                        role = safe_text(msg.get("role", "")).lower()
                        content = strip_chat_tokens(safe_text(msg.get("content", "")))
                        if role == "user" and not user_text:
                            user_text = content
                        elif role == "assistant" and user_text and not asst_text:
                            asst_text = content
                            break
                if user_text and asst_text and len(user_text) > 10 and len(asst_text) > 10:
                    texts.append(ALPACA_PROMPT.format(user_text, "", asst_text[:2000]) + EOS)

    # OpenAssistant / Daring-Anteater (messages format)
    elif fmt == "openassistant":
        for r in rows:
            user_text, asst_text = _extract_messages_pair(r)
            if user_text and asst_text and len(user_text) > 10 and len(asst_text) > 10:
                texts.append(ALPACA_PROMPT.format(user_text, "", asst_text[:2000]) + EOS)

    # OpenCodeReasoning (input / output / solution)
    elif fmt == "opencodereasoning":
        for r in rows:
            inp = strip_chat_tokens(safe_text(r.get("input", "")))
            sol = strip_chat_tokens(safe_text(r.get("solution", r.get("output", ""))))
            if inp and sol and len(inp) > 10:
                texts.append(ALPACA_PROMPT.format(inp, "", sol[:2000]) + EOS)

    # OpenThoughts-114k (conversations with optional code filter)
    elif fmt == "openthoughts":
        for r in rows:
            # OpenThoughts has conversations format
            user_text, asst_text = _extract_messages_pair(r)
            if not user_text or not asst_text:
                continue

            # Optional domain filter
            if filter_tag:
                combined = (user_text + " " + asst_text).lower()
                if filter_tag == "code":
                    code_signals = ["def ", "function", "class ", "import ", "return ",
                                    "python", "javascript", "code", "algorithm", "program"]
                    if not any(s in combined for s in code_signals):
                        continue

            if len(user_text) > 10 and len(asst_text) > 10:
                texts.append(ALPACA_PROMPT.format(user_text, "", asst_text[:2000]) + EOS)

    # OpenMathReasoning CoT (problem / generated_solution)
    elif fmt == "openmath_cot":
        for r in rows:
            prob = strip_chat_tokens(safe_text(r.get("problem", r.get("question", ""))))
            sol = strip_chat_tokens(safe_text(
                r.get("generated_solution", r.get("solution", r.get("answer", "")))
            ))
            if prob and sol and len(prob) > 10:
                texts.append(ALPACA_PROMPT.format(prob, "", sol[:2000]) + EOS)

    # OpenMathReasoning tool_integrated
    elif fmt == "openmath_tool":
        for r in rows:
            prob = strip_chat_tokens(safe_text(r.get("problem", r.get("question", ""))))
            sol = strip_chat_tokens(safe_text(
                r.get("generated_solution", r.get("solution", r.get("answer", "")))
            ))
            if prob and sol and len(prob) > 10:
                texts.append(ALPACA_PROMPT.format(prob, "", sol[:2000]) + EOS)

    # NuminaMath-CoT (problem / solution, also has messages)
    elif fmt == "numina_cot":
        for r in rows:
            prob = strip_chat_tokens(safe_text(r.get("problem", "")))
            sol = strip_chat_tokens(safe_text(r.get("solution", "")))
            if not prob or not sol:
                # Fallback to messages format
                user_text, asst_text = _extract_messages_pair(r)
                if user_text and asst_text:
                    prob, sol = user_text, asst_text
            if prob and sol and len(prob) > 10:
                texts.append(ALPACA_PROMPT.format(prob, "", sol[:2000]) + EOS)

    # MedQA USMLE (question / options / answer)
    elif fmt == "medqa":
        for r in rows:
            q = strip_chat_tokens(safe_text(r.get("question", r.get("sent1", ""))))
            # Options may be dict or list
            options = r.get("options", r.get("ending0", None))
            opts_text = ""
            if isinstance(options, dict):
                opts_text = "\n".join(f"{k}) {v}" for k, v in options.items())
            elif isinstance(options, list):
                opts_text = "\n".join(options)

            # Answer key
            answer = safe_text(r.get("answer", r.get("answer_idx", r.get("label", ""))))
            # Some MedQA versions have explanation
            exp = safe_text(r.get("exp", r.get("explanation", "")))

            if exp:
                answer_text = f"Answer: {answer}\nExplanation: {exp}"
            elif opts_text:
                answer_text = f"Options:\n{opts_text}\n\nAnswer: {answer}"
            else:
                answer_text = f"Answer: {answer}"

            if q and answer_text:
                texts.append(ALPACA_PROMPT.format(q, "", strip_chat_tokens(answer_text)) + EOS)

    # Medical reasoning (Question / Complex_CoT / Response)
    elif fmt == "medical_reasoning":
        for r in rows:
            q = strip_chat_tokens(safe_text(
                r.get("Question", r.get("question", r.get("instruction", "")))
            ))
            cot = strip_chat_tokens(safe_text(
                r.get("Complex_CoT", r.get("complex_cot", r.get("reasoning", "")))
            ))
            resp = strip_chat_tokens(safe_text(
                r.get("Response", r.get("response", r.get("output", "")))
            ))
            answer = resp if resp else cot
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            if q and answer and len(q) > 10:
                texts.append(ALPACA_PROMPT.format(q, "", answer[:2000]) + EOS)

    # PubMedQA (question / context / long_answer)
    elif fmt == "pubmedqa":
        for r in rows:
            q = strip_chat_tokens(safe_text(r.get("question", "")))
            ctx = r.get("context", r.get("CONTEXTS", ""))
            if isinstance(ctx, list):
                ctx = " ".join(str(c) for c in ctx)
            ctx = strip_chat_tokens(safe_text(ctx))
            ans = strip_chat_tokens(safe_text(
                r.get("long_answer", r.get("LONG_ANSWER", r.get("final_decision", "")))
            ))
            if q and ans:
                texts.append(ALPACA_PROMPT.format(q, ctx[:300], ans) + EOS)

    # Sky-T1 (messages format with thinking traces)
    elif fmt == "sky_t1":
        for r in rows:
            user_text, asst_text = _extract_messages_pair(r)
            if user_text and asst_text and len(user_text) > 10 and len(asst_text) > 10:
                texts.append(ALPACA_PROMPT.format(user_text, "", asst_text[:2000]) + EOS)

    # Fallback: try messages, then text column
    else:
        print(f"    [Warn] Unknown format '{fmt}', trying messages → text fallback")
        for r in rows:
            user_text, asst_text = _extract_messages_pair(r)
            if user_text and asst_text:
                texts.append(ALPACA_PROMPT.format(user_text, "", asst_text[:2000]) + EOS)
            elif "text" in r:
                text = strip_chat_tokens(safe_text(r["text"]))
                if text and len(text) > 20:
                    texts.append(text + EOS)

    if not texts:
        return None
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({"text": texts})


def build_domain_dataset(domain_cfg, tokenizer, extra_examples=None):
    """Load all sources for a domain, format, decontaminate, split."""
    domain_name = domain_cfg["name"]
    all_domains = [d["name"] for d in DOMAINS]

    print(f"  [Data] Loading '{domain_name}' sources ...")
    parts = []
    for src in domain_cfg.get("sources", []):
        ds = _load_dataset_safe(src["dataset"], src.get("config"), src["split"], src.get("take"))
        formatted = format_dataset(
            ds, src["format"], tokenizer, src.get("take"),
            filter_tag=src.get("filter")
        )
        if formatted:
            parts.append(formatted)
            print(f"    {src['dataset']}: {len(formatted)} examples")

    if not parts:
        raise ValueError(f"No data loaded for domain '{domain_name}'")

    combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    # Add extra examples from decontamination reassignment
    if extra_examples:
        from datasets import Dataset as HFDataset
        extra_ds = HFDataset.from_dict({"text": [e["text"] for e in extra_examples]})
        combined = concatenate_datasets([combined, extra_ds])
        print(f"    + {len(extra_examples)} reassigned from chat decontamination")

    # Decontaminate remove cross-domain leaks from chat
    domain_cfg["_reassigned"] = {}

    # Split
    split = combined.train_test_split(test_size=CFG.val_ratio, seed=CFG.seed)
    train_ds, val_ds = split["train"], split["test"]
    print(f"  [Data] {domain_name}: Train {len(train_ds):,}  |  Val {len(val_ds):,}")
    return train_ds, val_ds


# ════════════════════════════════════════════════════════════════════════
#  EVALUATE
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_ds, tokenizer, device, max_samples=200):
    model.eval()
    total_loss, count = 0.0, 0
    for i in range(0, min(len(val_ds), max_samples), CFG.batch_size):
        batch_texts = val_ds[i:i+CFG.batch_size]["text"]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=CFG.max_seq_len).to(device)
        enc["labels"] = enc["input_ids"].clone()
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
        with torch.amp.autocast("cuda"):          # ← add this
            outputs = model(**enc)
        total_loss += outputs.loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ════════════════════════════════════════════════════════════════════════
#  TRAINER — Custom loss with aux load balancing
# ════════════════════════════════════════════════════════════════════════

class BRAINSTACKSTrainer(SFTTrainer):
    def __init__(self, *args, moe_deltas=None, load_balance_coeff=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_deltas = moe_deltas or []
        self.load_balance_coeff = load_balance_coeff

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Gemma 3 requires token_type_ids (multimodal model, 0=text)
        if "token_type_ids" not in inputs and "input_ids" in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        outputs = model(**inputs)
        task_loss = outputs.loss
        aux_parts = []
        for delta in self.moe_deltas:
            a = delta.aux_loss
            if isinstance(a, torch.Tensor) and a.requires_grad:
                aux_parts.append(a)
        aux_loss = torch.stack(aux_parts).sum() if aux_parts else torch.tensor(0.0, device=task_loss.device)
        loss = task_loss + self.load_balance_coeff * aux_loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        # save_strategy="no" means this shouldn't be called,
        # but just in case — save only active stack weights
        os.makedirs(output_dir, exist_ok=True)
        state = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, StackedMoELoRALayer) and mod.active_stack is not None:
                for pname, p in mod.active_stack.named_parameters():
                    state[f"{name}.active_stack.{pname}"] = p.data.cpu()
        torch.save(state, os.path.join(output_dir, "adapters.pt"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


# ════════════════════════════════════════════════════════════════════════
#  DOMAIN BLOCK + MANAGER — Full resume support
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DomainBlock:
    name: str
    dataset: str
    num_inner_stacks: int
    stack_files: List[str]
    final_val_loss: float
    val_losses_per_round: List[float]
    time_min: float


class BRAINSTACKSManager:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.domain_blocks: List[DomainBlock] = []
        self.manifest_path = self.save_dir / "manifest.json"

    def save_manifest(self):
        data = {
            "model": CFG.model_name,
            "domains": [asdict(b) for b in self.domain_blocks],
            "total_stacks": sum(b.num_inner_stacks for b in self.domain_blocks),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            valid_fields = {f.name for f in __import__('dataclasses').fields(DomainBlock)}
            self.domain_blocks = [DomainBlock(**{k: v for k, v in d.items() if k in valid_fields}) for d in data["domains"]]
            print(f"  [Manifest] Loaded {len(self.domain_blocks)} domain blocks")

    def add_domain_block(self, block):
        self.domain_blocks.append(block)
        self.save_manifest()

    def get_completed_names(self):
        return {b.name for b in self.domain_blocks}

    def resume_stacks(self, model, stacked_layers, device):
        """Resume: reload ALL stacks from completed domains."""
        if not self.manifest_path.exists():
            return set()
        self.load_manifest()
        completed = set()
        for block in self.domain_blocks:
            for sf in block.stack_files:
                if os.path.exists(sf):
                    load_stack_as_frozen(model, stacked_layers, sf, device)
            completed.add(block.name)
        if completed:
            n = sum(len(l.frozen_stacks) for l in stacked_layers) // max(len(stacked_layers), 1)
            print(f"  [Resume] {n} stacks/layer  |  Skip: {', '.join(completed)}")
        gc.collect(); torch.cuda.empty_cache()
        return completed


# ════════════════════════════════════════════════════════════════════════
#  EARLY STOP + BEST WEIGHT SNAPSHOT CALLBACK
# ════════════════════════════════════════════════════════════════════════

class BestStackCallback(TrainerCallback):
    """
    Tracks best val loss during training. Snapshots active stack weights
    when val loss improves. Stops training if val loss spikes > patience evals.
    """
    def __init__(self, model, stacked_layers, patience=3, spike_threshold=0.05):
        self.model = model
        self.stacked_layers = stacked_layers
        self.patience = patience
        self.spike_threshold = spike_threshold
        self.best_val_loss = float('inf')
        self.best_state = None
        self.bad_evals = 0

    def _snapshot_active_stack(self):
        """Save a CPU copy of the current active stack weights."""
        state = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, StackedMoELoRALayer) and mod.active_stack is not None:
                for pname, p in mod.active_stack.named_parameters():
                    state[f"{name}.active_stack.{pname}"] = p.data.detach().cpu().clone()
        return state

    def _restore_active_stack(self, device):
        """Restore the best snapshot back into the active stack."""
        if self.best_state is None:
            return
        for name, mod in self.model.named_modules():
            if isinstance(mod, StackedMoELoRALayer) and mod.active_stack is not None:
                for pname, p in mod.active_stack.named_parameters():
                    key = f"{name}.active_stack.{pname}"
                    if key in self.best_state:
                        p.data.copy_(self.best_state[key].to(device=device, dtype=p.dtype))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_state = self._snapshot_active_stack()
            self.bad_evals = 0
            print(f"  [Best] ★ New best val loss: {val_loss:.6f} at step {state.global_step}")
        else:
            self.bad_evals += 1
            jump = val_loss - self.best_val_loss

            if jump > self.spike_threshold:
                print(f"  [EarlyStop] Val loss spiked {self.best_val_loss:.4f} → {val_loss:.4f} "
                      f"(+{jump:.4f}). Stopping & restoring best weights.")
                control.should_training_stop = True
                return

            if self.bad_evals >= self.patience:
                print(f"  [EarlyStop] No improvement for {self.patience} evals. "
                      f"Best: {self.best_val_loss:.6f}, current: {val_loss:.6f}. Stopping.")
                control.should_training_stop = True


# ════════════════════════════════════════════════════════════════════════
#  INNER LOOP
# ════════════════════════════════════════════════════════════════════════

def domain_inner_loop(model, stacked_layers, tokenizer, train_ds, val_ds,
                      device, domain_name, domain_dir, max_rounds, max_steps):
    domain_dir = Path(domain_dir)
    domain_dir.mkdir(parents=True, exist_ok=True)

    stack_files, val_losses = [], []
    prev_val_loss = evaluate(model, val_ds, tokenizer, device)
    val_losses.append(prev_val_loss)
    print(f"\n  [Inner {domain_name}] Baseline val loss: {prev_val_loss:.6f}")

    total_t0 = time.time()

    for round_num in range(1, max_rounds + 1):
        print(f"\n   {domain_name} inner round {round_num}/{max_rounds} ")
        add_new_stack(model, stacked_layers, device)
        moe_deltas = get_all_moe_deltas(stacked_layers)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        try:
            optimizer = bnb.optim.PagedAdamW8bit(
                trainable_params, lr=CFG.lr, weight_decay=CFG.weight_decay, betas=(0.9, 0.95))
        except:
            from torch.optim import AdamW
            optimizer = AdamW(trainable_params, lr=CFG.lr, weight_decay=CFG.weight_decay)

        round_dir = str(domain_dir / f"round_{round_num}")
        
        sft_args = SFTConfig(
            output_dir=round_dir,
            per_device_train_batch_size=CFG.batch_size,
            gradient_accumulation_steps=CFG.grad_accum,
            max_steps=max_steps,
            warmup_steps=CFG.warmup_steps,
            learning_rate=CFG.lr,
            weight_decay=CFG.weight_decay,
            max_grad_norm=CFG.max_grad_norm,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_steps=CFG.eval_every,
            eval_strategy="steps",
            # no checkpoint saves during training
            # Saves waste disk and time on 12B. We save stacks manually.
            save_strategy="no",
            load_best_model_at_end=False,
            seed=CFG.seed + round_num + len(stack_files) * 7,
            dataset_text_field="text",
            max_length=CFG.max_seq_len,
            packing=CFG.packing,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        t0 = time.time()

        # Callback: snapshot best weights + early stop on spike/plateau
        best_cb = BestStackCallback(
            model, stacked_layers,
            patience=4,
            spike_threshold=0.1,
        )

        trainer = BRAINSTACKSTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=train_ds, eval_dataset=val_ds,
            args=sft_args, optimizers=(optimizer, None),
            moe_deltas=moe_deltas, load_balance_coeff=CFG.load_balance_coeff,
            callbacks=[best_cb],
        )

        # Patch dataloader for pin_memory
        orig_get_train_dl = trainer.get_train_dataloader
        def patched_train_dl():
            dl = orig_get_train_dl()
            dl.pin_memory = True
            return dl
        
        trainer.get_train_dataloader = patched_train_dl
        train_result = trainer.train()
        elapsed = time.time() - t0
        print(f"  [Train] Done in {elapsed/60:.1f} min  |  Loss: {train_result.training_loss:.4f}")

        # Restore best weights if callback captured them
        if best_cb.best_state is not None:
            best_cb._restore_active_stack(device)
            val_loss = best_cb.best_val_loss
            print(f"  [Restore] Best weights restored  |  val_loss: {val_loss:.6f}")
        else:
            val_loss = None
            for entry in reversed(trainer.state.log_history):
                if "eval_loss" in entry:
                    val_loss = entry["eval_loss"]; break
            if val_loss is None:
                val_loss = evaluate(model, val_ds, tokenizer, device)

        improvement = prev_val_loss - val_loss
        print(f"  [Eval] Val loss: {val_loss:.6f}  |  Δ: {improvement:.6f}")
        val_losses.append(val_loss)

        stack_path = domain_dir / f"stack_{round_num}.pt"
        save_stack(model, stacked_layers, stack_path)
        stack_files.append(str(stack_path))

        # Generation test after SFT stack
        show_samples(model, tokenizer, f"After {domain_name} SFT stack {round_num}", device)

        freeze_active_stack(model, stacked_layers)

        if improvement < CFG.min_loss_delta and round_num > 1:
            print(f"  [Stop] Plateau after {round_num} rounds")
            break
        prev_val_loss = val_loss

        # Clean up trainer to free VRAM
        del trainer, optimizer, moe_deltas, trainable_params
        gc.collect(); torch.cuda.empty_cache()

    total_time = time.time() - total_t0
    return stack_files, val_losses, total_time


# ════════════════════════════════════════════════════════════════════════
#  GENERATION
# ════════════════════════════════════════════════════════════════════════

def show_samples(model, tokenizer, label, device):
    model.eval()
    print(f"\n{'='*65}\n  {label}\n{'='*65}")
    for prompt in CFG.gen_prompts:
        full = ALPACA_PROMPT.format(prompt, "", "") + "\n"
        ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model.generate(ids, max_new_tokens=CFG.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                                 token_type_ids=torch.zeros_like(ids))
        resp = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        print(f"\n> {prompt}\n  {resp.strip()[:400]}")
    print(f"{'='*65}\n")
    model.train()


# ════════════════════════════════════════════════════════════════════════
#  MAIN — OUTER LOOP
# ════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  BRAINSTACKS SFT                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Model     : {CFG.model_name:<50}║
║  Quant     : {'4-bit NF4':<50}║
║  Domains   : {len(DOMAINS):<50}║
║  Device    : {str(device):<50}║
║  Batch     : {CFG.batch_size}×{CFG.grad_accum} = {CFG.batch_size*CFG.grad_accum} effective{' '*(37-len(str(CFG.batch_size*CFG.grad_accum)))}║
║  Seq len   : {CFG.max_seq_len:<50}║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load base model (4-bit via BitsAndBytes) 
    print("[Model] Loading Gemma 3 12B IT (4-bit) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=COMPUTE_DTYPE,
    )
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.enable_input_require_grads()
    if hasattr(F, 'scaled_dot_product_attention'):
        print("  [Optim] SDPA attention backend active")

    # Inject stacked layers 
    model, stacked_layers = inject_stacked_layers(model)
    model._hf_peft_config_loaded = True

    # Manager + Resume 
    manager = BRAINSTACKSManager(save_dir=CFG.save_dir)
    completed_domains = manager.resume_stacks(model, stacked_layers, device)

    # Pre-load val sets for forgetting checks 
    domain_val_sets = {}

    # Baseline 
    show_samples(model, tokenizer, "Before training (base + resumed stacks)", device)

    # Outer loop 
    all_results = {}
    forgetting_matrix = {}
    total_t0 = time.time()
    chat_reassigned = {}

    for domain_idx, domain_cfg in enumerate(DOMAINS):
        domain_name = domain_cfg["name"]

        if domain_name in completed_domains:
            print(f"\n{'━'*65}\n  DOMAIN {domain_idx+1}/{len(DOMAINS)}: {domain_name.upper()}  |  SKIP (resumed)\n{'━'*65}")
            try:
                train_ds, val_ds = build_domain_dataset(domain_cfg, tokenizer)
                domain_val_sets[domain_name] = val_ds
            except: pass
            continue

        print(f"\n{'━'*65}")
        print(f"  OUTER LOOP — DOMAIN {domain_idx+1}/{len(DOMAINS)}: {domain_name.upper()}")
        print(f"{'━'*65}")

        extra = []

        # Build dataset (with decontamination for chat)
        train_ds, val_ds = build_domain_dataset(domain_cfg, tokenizer, extra_examples=extra if extra else None)
        domain_val_sets[domain_name] = val_ds
        
        # Null space
        if any(len(l.frozen_stacks) > 0 for l in stacked_layers):
            compute_null_projectors(model, stacked_layers, tokenizer, val_ds, device,
                                    CFG.ns_samples, CFG.ns_top_k_dirs)

        # Inner loop (SFT)
        domain_dir = Path(CFG.save_dir) / domain_name
        stack_files, val_losses, domain_time = domain_inner_loop(
            model, stacked_layers, tokenizer, train_ds, val_ds, device,
            domain_name=domain_name, domain_dir=domain_dir,
            max_rounds=domain_cfg.get("max_inner_rounds", CFG.max_inner_rounds),
            max_steps=domain_cfg.get("max_steps", CFG.max_steps),
        )

        # Clear null projectors
        for l in stacked_layers: l.null_projector = None

        # Create domain block
        block = DomainBlock(
            name=domain_name,
            dataset=str([s["dataset"] for s in domain_cfg.get("sources", [])]),
            num_inner_stacks=len(stack_files),
            stack_files=stack_files,
            final_val_loss=val_losses[-1],
            val_losses_per_round=val_losses,
            time_min=round(domain_time / 60, 2),
        )
        manager.add_domain_block(block)

        all_results[domain_name] = {
            "val_losses": val_losses,
            "num_stacks": len(stack_files),
            "time_min": round(domain_time / 60, 2),
        }

        # Forgetting check
        print(f"\n   Forgetting check after {domain_name} ")
        forgetting_matrix[domain_name] = {}
        for prev_name, prev_val_ds in domain_val_sets.items():
            prev_loss = evaluate(model, prev_val_ds, tokenizer, device)
            forgetting_matrix[domain_name][prev_name] = prev_loss
            marker = " ✓" if prev_name == domain_name else ""
            print(f"    {prev_name:<15} val loss: {prev_loss:.6f}{marker}")

        # Show generation
        show_samples(model, tokenizer, f"After domain: {domain_name}", device)

        # Save results
        with open(Path(CFG.save_dir) / "forgetting_matrix.json", "w", encoding="utf-8") as f:
            json.dump(forgetting_matrix, f, indent=2)

        # Free ALL domain data aggressively
        del train_ds
        for src in domain_cfg.get("sources", []):
            cache_name = src["dataset"].replace("/", "___")
            cache_dirs = list(Path.home().glob(f".cache/huggingface/datasets/{cache_name}*"))
            for cd in cache_dirs:
                try: shutil.rmtree(cd); print(f"  [Free] {cd.name}")
                except: pass
        gc.collect(); torch.cuda.empty_cache()

    #  7. Final summary 
    total_elapsed = time.time() - total_t0
    n_stacks = sum(len(l.frozen_stacks) for l in stacked_layers) // max(len(stacked_layers), 1)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'━'*65}")
    print(f"  BRAINSTACKS SFT — OUTER LOOP COMPLETE")
    print(f"{'━'*65}")
    print(f"  Base model         : {CFG.model_name}")
    print(f"  Quantization       : 4-bit NF4")
    print(f"  Domains trained    : {len(manager.domain_blocks)}")
    print(f"  Total stacks/layer : {n_stacks}")
    print(f"  Total params       : {total_params/1e6:.1f}M")
    print(f"  Total time         : {total_elapsed/60:.1f} min")
    print(f"  >> Then: run meta_router to train the meta-router <<")

    for block in manager.domain_blocks:
        print(f"\n  [{block.name}]  stacks: {block.num_inner_stacks}  |  "
              f"val loss: {block.final_val_loss:.6f}  |  time: {block.time_min:.1f} min")

    # Forgetting matrix
    if forgetting_matrix:
        domains_so_far = list(forgetting_matrix.keys())
        print(f"\n  {'Forgetting':>15}", end="")
        for d in domains_so_far: print(f"  {d:>12}", end="")
        print()
        for after_d in domains_so_far:
            print(f"  After {after_d:<8}", end="")
            for eval_d in domains_so_far:
                val = forgetting_matrix[after_d].get(eval_d, float('nan'))
                print(f"  {val:>12.4f}", end="")
            print()

    print(f"\n{'━'*65}")
    print(f"  All outputs in: {CFG.save_dir}/")
    print(f"{'━'*65}\n")

    return model, tokenizer, manager

if __name__ == "__main__":
    main()
    
    