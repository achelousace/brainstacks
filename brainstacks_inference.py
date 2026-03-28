"""
BrainStacks Inference — Gemma 3 12B IT · Meta-Routed Generation       
================================================================
   
Disk-offloaded inference:                                               
• Base model (4-bit NF4) + meta-router always in GPU                  
• ALL domain stacks live on DISK until needed                         
• Per prompt: router classifies → load stacks → generate → free       

==========================================================================

Modes:                                                                  
1. Interactive chat loop (default — run cell, start typing)           
2. Benchmark test prompts (set MODE = "benchmark")                    
3. Compare ungated vs routed (set MODE = "compare")                   
4. Single prompt (set MODE = "single")               
                
==========================================================================

Author: Mohammad R. Abu Ayyash — Brains Build Research, Palestine
"""

import os, sys, json, math, time, gc, warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


#  ★★★ CONFIG — EDIT THESE FOR YOUR COLAB ★★★

MODEL_NAME     = "google/gemma-3-12b-it"
MAX_SEQ_LEN    = 512
SAVE_DIR       = "./BrainStacks_gemma3"
ROUTER_PATH    = "./BrainStacks_gemma3/meta_router.pt"

# Run mode: "interactive", "benchmark", "compare", "single"
MODE           = "interactive"
SINGLE_PROMPT  = "Write a Python function to check if a number is prime."

# Generation
GEN_MAX_TOKENS = 256
GREEDY         = True
TEMPERATURE    = 0.7

#  ARCHITECTURE CONFIG — must match BrainStacksSFT script exactly

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

# Router config
ROUTER_SEQ_LEN = 96
CHAT_FLOOR     = 0.20
GATE_THRESHOLD = 0.12


# ════════════════════════════════════════════════════════════════════════
#  MOE-LORA COMPONENTS — exact match with BrainStacksSFT script
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
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K
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
    """Soft-gated: each domain weighted independently via sigmoid router."""
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
                    # UNGATED: all stacks fire
                    for stack in self.frozen_stacks:
                        was_cpu = not next(stack.parameters()).is_cuda
                        if was_cpu: stack.to(x.device)
                        out = out + stack(x).to(dtype=out_dtype)
                        if was_cpu: stack.cpu()

        if self.active_stack is not None:
            out = out + self.active_stack(x).to(dtype=out_dtype)
        return out


# ════════════════════════════════════════════════════════════════════════
#  META-ROUTER — Must match router training architecture
# ════════════════════════════════════════════════════════════════════════

class MetaRouter(nn.Module):
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
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
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
        g_ctx = self.global_ln(torch.einsum("bs,bsh->bh", g_attn, x))
        d_scores = torch.einsum("bsh,dh->bds", x, self.domain_queries)
        d_mask = attention_mask.unsqueeze(1).expand(-1, d_scores.size(1), -1)
        d_attn = self.masked_softmax(d_scores, d_mask, dim=-1)
        d_ctx = self.ctx_ln(torch.einsum("bds,bsh->bdh", d_attn, x))
        g_expanded = g_ctx.unsqueeze(1).expand(-1, d_ctx.size(1), -1)
        fused = self.ff(torch.cat([d_ctx, g_expanded], dim=-1))
        temperature = torch.exp(self.log_temperature).clamp(min=0.3, max=3.0)
        return self.out(fused).squeeze(-1) / temperature

    def predict(self, token_states, attention_mask):
        return torch.sigmoid(self.forward(token_states, attention_mask))


# ════════════════════════════════════════════════════════════════════════
#  MODEL SETUP
# ════════════════════════════════════════════════════════════════════════

def inject_stacked_layers(model):
    """Replace target linear layers with StackedMoELoRALayer wrappers."""
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model, stacked_layers


def load_single_stack(model, stacked_layers, stack_path, device):
    """Load one .pt stack file → freeze into frozen_stacks."""
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
            for p in layer.active_stack.parameters():
                p.requires_grad_(False)
            layer.active_stack.half()
            layer.active_stack.cpu()
            layer.frozen_stacks.append(layer.active_stack)
            layer.active_stack = None
    del state


def set_domain_weights(model, weights):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer): m._domain_weights = weights

def clear_domain_weights(model):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer): m._domain_weights = None

def set_base_only(model, flag):
    for m in model.modules():
        if isinstance(m, StackedMoELoRALayer): m._router_base_only = flag


# ════════════════════════════════════════════════════════════════════════
#  DISK OFFLOAD ENGINE
# ════════════════════════════════════════════════════════════════════════

class DiskOffloadEngine:
    """
    Manages stack loading/unloading from disk per prompt.

    GPU at rest:  base model (4-bit) + router (~11MB)
    GPU at peak:  + active domain stacks (~50-150MB per domain)
    """

    def __init__(self, model, stacked_layers, router, tokenizer, domain_names,
                 domain_stack_paths, device):
        self.model = model
        self.stacked_layers = stacked_layers
        self.router = router
        self.tokenizer = tokenizer
        self.domain_names = domain_names
        self.device = device

        self.domain_stack_paths = domain_stack_paths  # {domain: [path1, path2, ...]}
        self.domain_stack_counts = [len(domain_stack_paths.get(d, [])) for d in domain_names]
        self._loaded_domains = set()

        total = sum(self.domain_stack_counts)
        disk_mb = sum(
            os.path.getsize(p) for paths in domain_stack_paths.values()
            for p in paths if os.path.exists(p)
        ) / 1e6
        print(f"  [Disk] {total} stacks across {len(domain_names)} domains  |  {disk_mb:.0f} MB on disk")
        for d in domain_names:
            n = len(domain_stack_paths.get(d, []))
            print(f"    {d}: {n} stacks")

    def _load_single_stack(self, stack_path):
        """Load one .pt file into frozen_stacks."""
        for layer in self.stacked_layers:
            layer.active_stack = MoELoRADelta(
                layer.frozen.in_features, layer.frozen.out_features
            ).to(device=self.device, dtype=COMPUTE_DTYPE)

        state = torch.load(stack_path, map_location=self.device, weights_only=False)
        for name, mod in self.model.named_modules():
            if isinstance(mod, StackedMoELoRALayer) and mod.active_stack is not None:
                for pname, p in mod.active_stack.named_parameters():
                    key = f"{name}.active_stack.{pname}"
                    if key in state:
                        p.data.copy_(state[key].to(device=self.device, dtype=p.dtype))

        for layer in self.stacked_layers:
            if layer.active_stack is not None:
                for p in layer.active_stack.parameters():
                    p.requires_grad_(False)
                layer.active_stack.half()
                layer.active_stack.cpu()
                layer.frozen_stacks.append(layer.active_stack)
                layer.active_stack = None
        del state

    def load_domains(self, domains_to_load):
        """Load specific domains' stacks from disk to GPU."""
        for domain in domains_to_load:
            if domain in self._loaded_domains:
                continue
            for path in self.domain_stack_paths.get(domain, []):
                self._load_single_stack(path)
            self._loaded_domains.add(domain)

        # Update stack counts
        counts = []
        for d in self.domain_names:
            if d in self._loaded_domains:
                counts.append(len(self.domain_stack_paths.get(d, [])))
            else:
                counts.append(0)
        for layer in self.stacked_layers:
            layer._domain_stack_counts = counts

    def _unload_domains(self, domains_to_remove):
        """Selectively remove specific domains' stacks, keep the rest.

        Stacks are stored sequentially in frozen_stacks by domain order.
        E.g. if domains=[chat(2), code(1), math(2)] → frozen_stacks has 5 entries:
          [chat_s1, chat_s2, code_s1, math_s1, math_s2]
        To remove code: delete index 2, update counts.
        """
        if not domains_to_remove:
            return

        for layer in self.stacked_layers:
            new_stacks = nn.ModuleList()
            start = 0
            for d_idx, d in enumerate(self.domain_names):
                count = len(self.domain_stack_paths.get(d, [])) if d in self._loaded_domains else 0
                if d not in domains_to_remove:
                    # Keep these stacks
                    for s in layer.frozen_stacks[start:start + count]:
                        new_stacks.append(s)
                start += count
            layer.frozen_stacks = new_stacks

        self._loaded_domains -= set(domains_to_remove)

        # Update counts
        counts = []
        for d in self.domain_names:
            if d in self._loaded_domains:
                counts.append(len(self.domain_stack_paths.get(d, [])))
            else:
                counts.append(0)
        for layer in self.stacked_layers:
            layer._domain_stack_counts = counts

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self):
        """Remove ALL stacks from GPU."""
        for layer in self.stacked_layers:
            layer.frozen_stacks = nn.ModuleList()
            layer.active_stack = None
            layer._domain_weights = None
            layer._domain_stack_counts = None
        self._loaded_domains.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_all_domains(self):
        """Load every domain — for ungated comparison."""
        self.unload_all()
        self.load_domains(list(self.domain_names))

    def route(self, prompt):
        """Classify prompt → domain weights (sigmoid, independent per domain)."""
        enc = self.tokenizer(
            [prompt], return_tensors="pt", padding=True,
            truncation=True, max_length=ROUTER_SEQ_LEN, add_special_tokens=False
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        token_type_ids = torch.zeros_like(input_ids)

        set_base_only(self.model, True)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True, use_cache=False, return_dict=True
            )
        set_base_only(self.model, False)

        hs = out.hidden_states
        features = (0.45 * hs[len(hs)//2].float() + 0.55 * hs[-1].float())

        with torch.no_grad():
            probs = self.router.predict(features, attention_mask).squeeze(0)

        # Chat floor
        if "chat" in self.domain_names:
            chat_idx = self.domain_names.index("chat")
            probs[chat_idx] = torch.max(probs[chat_idx], torch.tensor(CHAT_FLOOR, device=probs.device))

        return probs.to(device=self.device, dtype=COMPUTE_DTYPE)

    def get_active_domains(self, weights):
        return [d for i, d in enumerate(self.domain_names)
                if float(weights[i].float().item()) > GATE_THRESHOLD]

    def format_route(self, weights):
        parts = []
        for i, d in enumerate(self.domain_names):
            w = float(weights[i].float().item())
            if w > GATE_THRESHOLD:
                parts.append(f"{d}={w:.2f}")
        return ", ".join(parts) if parts else "base only"

    def routed_generate(self, prompt, max_tokens=GEN_MAX_TOKENS,
                        temperature=0.7, greedy=True):
        """Full inference cycle with smart domain caching.

        Only loads/unloads domains that CHANGED between prompts:
          - If last prompt used [chat, code] and this one needs [chat, math],
            it keeps chat, unloads code, loads math.
          - If same domains → zero disk I/O.
        """
        t0 = time.time()

        # 1. Route
        weights = self.route(prompt)
        active_domains = set(self.get_active_domains(weights))
        route_str = self.format_route(weights)
        t_route = time.time() - t0

        # 2. Smart swap — only change what's different
        to_unload = self._loaded_domains - active_domains
        to_load = active_domains - self._loaded_domains

        if to_unload:
            self._unload_domains(to_unload)
        if to_load:
            self.load_domains(to_load)
        t_load = time.time() - t0 - t_route

        # 3. Set routing weights and generate
        set_domain_weights(self.model, weights)

        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=MAX_SEQ_LEN).input_ids.to(self.device)
        token_type_ids = torch.zeros_like(ids)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            token_type_ids=token_type_ids,
        )
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = self.model.generate(ids, **gen_kwargs)

        response = self.tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        t_gen = time.time() - t0 - t_route - t_load

        # 4. Clear weights but KEEP stacks loaded for next prompt
        clear_domain_weights(self.model)

        stats = {
            "route": route_str,
            "domains_loaded": list(active_domains),
            "swapped_in": list(to_load),
            "swapped_out": list(to_unload),
            "t_route": t_route, "t_load": t_load,
            "t_gen": t_gen, "t_total": time.time() - t0,
        }
        return response, stats

    def ungated_generate(self, prompt, max_tokens=GEN_MAX_TOKENS,
                         temperature=0.7, greedy=True):
        """Load ALL stacks, no routing — for comparison."""
        t0 = time.time()
        self.load_all_domains()
        clear_domain_weights(self.model)

        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=MAX_SEQ_LEN).input_ids.to(self.device)
        token_type_ids = torch.zeros_like(ids)

        gen_kwargs = dict(
            max_new_tokens=max_tokens, repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            token_type_ids=token_type_ids,
        )
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = self.model.generate(ids, **gen_kwargs)

        response = self.tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        self.unload_all()
        return response, time.time() - t0


# ════════════════════════════════════════════════════════════════════════
#  BENCHMARK PROMPTS — covers all 5 domains + cross-domain
# ════════════════════════════════════════════════════════════════════════

BENCHMARK_PROMPTS = [
    # Chat / general
    ("Explain what a neural network is in simple terms.", "chat"),
    ("What is the difference between civil and criminal law?", "chat"),
    ("How does gravity work?", "chat/reasoning"),
    # Code
    ("Write a Python function to check if a number is prime.", "code"),
    ("Write a Python class for a stack data structure with push, pop, and peek.", "code"),
    ("Implement binary search in Python.", "code"),
    # Math
    ("If a train travels 120km in 2 hours, what is its speed?", "math"),
    ("Explain how to solve 3x + 7 = 22 step by step.", "math"),
    ("What is the derivative of x^3 + 2x^2 - 5x + 1?", "math"),
    # Medical
    ("What are the common symptoms of influenza?", "medical"),
    ("Explain the difference between type 1 and type 2 diabetes.", "medical"),
    ("What is the mechanism of action of metformin?", "medical"),
    # Reasoning
    ("If all cats are mammals and all mammals breathe air, do all cats breathe air?", "reasoning"),
    ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", "reasoning"),
    # Cross-domain
    ("A patient needs 500mg of medication per day split into 3 doses. How many mg per dose?", "medical+math"),
    ("Write Python code to calculate medication dosage based on patient weight.", "code+medical"),
]


# ════════════════════════════════════════════════════════════════════════
#  LOADING — model, stacks from manifest, router
# ════════════════════════════════════════════════════════════════════════

def gpu_mb():
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0


def load_everything(device):
    """Load base model, inject layers, load stacks from manifest, load router."""

    # 1. Base model
    print("  [1/3] Loading Gemma 3 12B IT (4-bit NF4) ...")
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
    model._hf_peft_config_loaded = True
    print(f"  GPU after base: {gpu_mb():.0f} MB")

    # 2. Inject stacked layers + load stacks from manifest
    print("  [2/3] Injecting StackedMoELoRALayer + loading stacks ...")
    model, stacked_layers = inject_stacked_layers(model)

    manifest_path = Path(SAVE_DIR) / "manifest.json"
    domain_names = []
    domain_stack_paths = {}  # {domain: [path1, path2, ...]}
    stacks_per_domain = {}

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for block in manifest["domains"]:
            name = block["name"]
            stack_files = [sf for sf in block["stack_files"] if os.path.exists(sf)]
            if not stack_files:
                continue
            # Load stacks into frozen_stacks
            for sf in stack_files:
                load_single_stack(model, stacked_layers, sf, device)
            stacks_per_domain[name] = len(stack_files)
            domain_stack_paths[name] = stack_files
            domain_names.append(name)
            print(f"    {name}: {len(stack_files)} stacks loaded")

        counts = [stacks_per_domain.get(d, 0) for d in domain_names]
        for layer in stacked_layers:
            layer._domain_stack_counts = counts
    else:
        print(f"  [Warn] No manifest at {manifest_path} — base model only")

    n = sum(len(l.frozen_stacks) for l in stacked_layers) // max(len(stacked_layers), 1)
    print(f"  Stacks/layer: {n}  |  Domains: {domain_names}")
    print(f"  GPU after stacks: {gpu_mb():.0f} MB")

    # 3. Meta-router
    print("  [3/3] Loading meta-router ...")
    router = None
    if os.path.exists(ROUTER_PATH):
        ckpt = torch.load(ROUTER_PATH, map_location=device, weights_only=False)
        router = MetaRouter(
            token_dim=ckpt["token_dim"],
            n_domains=ckpt["n_domains"],
        ).to(device)
        router.load_state_dict(ckpt["state_dict"])
        router.eval()
        print(f"    Router: {sum(p.numel() for p in router.parameters())/1e6:.3f}M params")
        print(f"    Domains from router: {ckpt.get('domain_names', '?')}")
        print(f"    Version: {ckpt.get('version', '?')}")
    else:
        print(f"  [Warn] Router not found at {ROUTER_PATH}")

    model.eval()
    print(f"  GPU ready: {gpu_mb():.0f} MB\n")

    return model, tokenizer, stacked_layers, router, domain_names, domain_stack_paths


# ════════════════════════════════════════════════════════════════════════
#  RUN MODES
# ════════════════════════════════════════════════════════════════════════

def run_benchmark(engine, compare=False):
    print(f"\n{'='*80}")
    print(f"  BENCHMARK — {'UNGATED vs ROUTED' if compare else 'ROUTED'}")
    print(f"{'='*80}")

    for prompt, expected_domain in BENCHMARK_PROMPTS:
        print(f"\n  > {prompt[:72]}...")
        print(f"    Expected: {expected_domain}")

        resp_r, stats = engine.routed_generate(prompt)
        print(f"    Route:    [{stats['route']}]")
        print(f"    Timing:   route={stats['t_route']:.2f}s  load={stats['t_load']:.2f}s  gen={stats['t_gen']:.2f}s  total={stats['t_total']:.2f}s")
        print(f"    ROUTED:   {resp_r[:300]}")

        if compare:
            resp_u, t_u = engine.ungated_generate(prompt)
            print(f"    UNGATED:  {resp_u[:300]}")
            print(f"    Ungated time: {t_u:.2f}s")

    print(f"\n  GPU after cleanup: {gpu_mb():.0f} MB")


def run_interactive(engine):
    print(f"""
{'═'*70}
  BrainStacks Interactive — Gemma 3 12B · {len(engine.domain_names)} Domains
  
  Type your message and press Enter.
  
  Commands:
    /greedy          toggle greedy/sampling
    /temp 0.5        set temperature
    /tokens 512      set max tokens
    /stats           toggle timing display
    /stacks          show loaded vs on-disk stacks
    /flush           unload all stacks from GPU
    /route <text>    show routing without generating
    /bench           run benchmarks
    /compare         run ungated vs routed comparison
    /gpu             show GPU memory
    /quit            exit
{'═'*70}
""")
    greedy = GREEDY
    temperature = TEMPERATURE
    max_tokens = GEN_MAX_TOKENS
    show_stats = True

    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break

        if not prompt:
            continue

        if prompt.startswith("/"):
            cmd = prompt.lower().split()
            if cmd[0] == "/quit":
                print("Bye!"); break
            elif cmd[0] == "/greedy":
                greedy = not greedy
                print(f"  Greedy: {greedy}")
            elif cmd[0] == "/temp" and len(cmd) > 1:
                temperature = float(cmd[1])
                print(f"  Temperature: {temperature}")
            elif cmd[0] == "/tokens" and len(cmd) > 1:
                max_tokens = int(cmd[1])
                print(f"  Max tokens: {max_tokens}")
            elif cmd[0] == "/stats":
                show_stats = not show_stats
                print(f"  Show stats: {show_stats}")
            elif cmd[0] == "/route":
                text = " ".join(cmd[1:]) or "Hello, how are you?"
                weights = engine.route(text)
                route_str = engine.format_route(weights)
                print(f"  Route: [{route_str}]")
                for i, d in enumerate(engine.domain_names):
                    print(f"    {d}: {float(weights[i].float()):.4f}")
            elif cmd[0] == "/stacks":
                for d in engine.domain_names:
                    n_disk = len(engine.domain_stack_paths.get(d, []))
                    loaded = "✓ GPU" if d in engine._loaded_domains else "  disk"
                    print(f"    {loaded}  {d}: {n_disk} stacks")
                print(f"    Currently in GPU: {engine._loaded_domains or 'none'}")
            elif cmd[0] == "/flush":
                engine.unload_all()
                print(f"  Flushed all stacks from GPU. Memory: {gpu_mb():.0f} MB")
            elif cmd[0] == "/bench":
                run_benchmark(engine, compare=False)
            elif cmd[0] == "/compare":
                run_benchmark(engine, compare=True)
            elif cmd[0] == "/gpu":
                print(f"  GPU allocated: {gpu_mb():.0f} MB")
                if torch.cuda.is_available():
                    print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e6:.0f} MB")
            else:
                print("  Unknown command. Try /greedy /temp /tokens /stats /stacks /flush /route /bench /compare /gpu /quit")
            continue

        resp, stats = engine.routed_generate(
            prompt, max_tokens=max_tokens,
            temperature=temperature, greedy=greedy,
        )
        print(f"\nBrainStacks > {resp}")
        if show_stats:
            swap_info = ""
            if stats.get("swapped_in") or stats.get("swapped_out"):
                swap_info = f"  +{stats.get('swapped_in',[])} -{stats.get('swapped_out',[])}"
            elif not stats.get("swapped_in") and not stats.get("swapped_out"):
                swap_info = "  (cache hit — zero disk I/O)"
            print(f"  [{stats['route']}]  route={stats['t_route']:.2f}s  load={stats['t_load']:.2f}s  gen={stats['t_gen']:.2f}s{swap_info}")
        print()

# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BrainStacks Inference — Gemma 3 12B IT · Meta-Routed                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Model:   {MODEL_NAME:<60}║
║  Stacks:  {SAVE_DIR:<60}║
║  Router:  {ROUTER_PATH:<60}║
║  Mode:    {MODE:<60}║
║  Device:  {str(device):<60}║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    model, tokenizer, stacked_layers, router, domain_names, domain_stack_paths = load_everything(device)

    if router is None:
        print("  [Error] No router loaded — cannot do routed inference")
        return

    engine = DiskOffloadEngine(
        model, stacked_layers, router, tokenizer,
        domain_names, domain_stack_paths, device
    )

    if MODE == "single":
        print(f"\n  > {SINGLE_PROMPT}")
        resp, stats = engine.routed_generate(SINGLE_PROMPT)
        print(f"  Route: [{stats['route']}]")
        print(f"  {resp}")
        print(f"  [{stats['t_total']:.2f}s  route={stats['t_route']:.2f}s  load={stats['t_load']:.2f}s  gen={stats['t_gen']:.2f}s]")

    elif MODE == "benchmark":
        run_benchmark(engine, compare=False)

    elif MODE == "compare":
        run_benchmark(engine, compare=True)

    else:  # interactive
        run_interactive(engine)


if __name__ == "__main__":
    main()
    