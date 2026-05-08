import os
import gc
import re
import json
import warnings
import glob

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sympy
from sympy.parsing.latex import parse_latex
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR      = "/home/RL/Rani_u23ai131_8809617110"
MODEL_DIR     = f"{BASE_DIR}/model_cache"
DATASET_CACHE = f"{BASE_DIR}/dataset_cache"
OUTPUT_DIR    = f"{BASE_DIR}/results1"

os.makedirs(DATASET_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT FLAGS
# ═════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME             = "baseline"

# Existing improvements (already tested)
IMPROVEMENT_1_SYMPY_REWARD  = False
IMPROVEMENT_2_KL_PENALTY    = False
IMPROVEMENT_3_FORMAT_REWARD = False
IMPROVEMENT_4_STEP_REWARD   = False
IMPROVEMENT_5_MORE_ROLLOUTS = False   # K=8 → K=16

# NEW improvements to test
IMPROVEMENT_6_TEMP_ANNEAL   = False   # temperature annealing: high early, low late
IMPROVEMENT_7_LENGTH_NORM   = False   # length-normalized policy loss

# ═════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

TRAIN_STEPS        = 25       # Decreased step count to iterate experiments faster
BASE_ROLLOUTS      = 8        # ← raised from 4 to 8. K=4 gives ~1 correct/step
                              #   randomly, making LOO advantage near-zero.
                              #   K=8 gives 2–3 correct/step reliably.
                              #   Original paper used K=128 — use as many as VRAM allows.
LR                 = 2e-7
MAX_NEW_TOKENS     = 400
EVAL_MAX_TOKENS    = 800
SEED               = 42
GRAD_CLIP          = 1.0
TEMPERATURE        = 0.8      # slightly higher than before for more diversity
TOP_P              = 0.92
REPETITION_PENALTY = 1.0
KL_BETA            = 0.1
PATIENCE           = 5        # increased: 3 was too aggressive for noisy eval

# Temperature annealing schedule (for IMPROVEMENT_6)
TEMP_START = 1.0   # high temperature early = explore
TEMP_END   = 0.4   # low temperature late = exploit

torch.manual_seed(SEED)
K = 16 if IMPROVEMENT_5_MORE_ROLLOUTS else BASE_ROLLOUTS

print("=" * 60)
print(f"  EXPERIMENT : {EXPERIMENT_NAME}")
print(f"  K={K}  LR={LR}  Steps={TRAIN_STEPS}")
print(f"  Temp={TEMPERATURE}  GradClip={GRAD_CLIP}  KL_beta={KL_BETA}")
print("=" * 60)
print(f"  imp1_sympy    : {IMPROVEMENT_1_SYMPY_REWARD}")
print(f"  imp2_kl       : {IMPROVEMENT_2_KL_PENALTY}")
print(f"  imp3_format   : {IMPROVEMENT_3_FORMAT_REWARD}")
print(f"  imp4_step     : {IMPROVEMENT_4_STEP_REWARD}")
print(f"  imp5_rollouts : {IMPROVEMENT_5_MORE_ROLLOUTS}")
print(f"  imp6_anneal   : {IMPROVEMENT_6_TEMP_ANNEAL}")
print(f"  imp7_lennorm  : {IMPROVEMENT_7_LENGTH_NORM}")
print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# MEMORY
# ═════════════════════════════════════════════════════════════════════════════

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def gpu_mem():
    if not torch.cuda.is_available():
        return "no GPU"
    used  = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{used:.1f}/{total:.1f} GB"


# ═════════════════════════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_box(text):
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None
    start, depth = match.end(), 1
    for i, ch in enumerate(text[start:]):
        if ch == "{":   depth += 1
        elif ch == "}": depth -= 1
        if depth == 0:
            content = text[start:start+i].strip()
            if len(content) > 30 or re.fullmatch(r"0{4,}", content):
                return None
            return content
    return None

def extract_answer(text):
    if not text:
        return None
    m = extract_box(text)
    if m is not None:
        return m
    nums = re.findall(r"-?\b\d{1,6}(?:\.\d{1,4})?\b", text)
    return nums[-1] if nums else None

def extract_gt(gt_text):
    m = extract_box(gt_text)
    return m if m else gt_text.strip()

def _to_sympy(s):
    s = str(s).strip().rstrip(".,;! ")
    try:
        return parse_latex(s)
    except Exception:
        pass
    try:
        s2 = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
        s2 = re.sub(r"\\sqrt\{([^}]*)\}",             r"sqrt(\1)",  s2)
        s2 = re.sub(r"\\left|\\right|\\,",            "",           s2)
        s2 = re.sub(r"\\cdot|\\times",                "*",          s2)
        s2 = re.sub(r"\\pi\b",                        "pi",         s2)
        s2 = re.sub(r"[{}]", "", s2).replace("^", "**")
        return sympy.sympify(s2, evaluate=True)
    except Exception:
        return None

def answers_equal(pred, gt):
    if pred is None or gt is None:
        return False
    pred = str(pred).rstrip(".,;! ").strip()
    gt   = str(gt).rstrip(".,;! ").strip()
    if pred == gt:
        return True
    try:
        sp, sg = _to_sympy(pred), _to_sympy(gt)
        if sp is not None and sg is not None:
            if sympy.simplify(sp - sg) == 0:
                return True
    except Exception:
        pass
    try:
        if abs(float(pred.replace(",","")) - float(gt.replace(",",""))) < 1e-4:
            return True
    except Exception:
        pass
    return False

def is_correct(pred, gt):
    if pred is None:
        return False
    gt_val = extract_gt(gt)
    if IMPROVEMENT_1_SYMPY_REWARD:
        return answers_equal(pred, gt_val)
    p = str(pred).replace(" ","").replace("\\left","").replace("\\right","").rstrip(".,;!").strip()
    g = str(gt_val).replace(" ","").replace("\\left","").replace("\\right","").rstrip(".,;!").strip()
    if p == g:
        return True
    try:
        if abs(float(p.replace(",","")) - float(g.replace(",",""))) < 1e-4:
            return True
    except Exception:
        pass
    return False


# ═════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_reward(response: str, gt: str) -> float:
    pred      = extract_answer(response)
    gt_val    = extract_gt(gt)
    has_boxed = bool(re.search(r"\\boxed\{", response))

    if pred is None:
        reward = -1.0
    elif is_correct(pred, gt):
        reward = +2.0
    else:
        reward = 0.0

    # imp1: sympy proximity bonus
    if IMPROVEMENT_1_SYMPY_REWARD and pred is not None and not is_correct(pred, gt):
        try:
            p_val   = float(str(pred).replace(",", ""))
            g_val   = float(str(gt_val).replace(",", ""))
            rel_err = abs(p_val - g_val) / (abs(g_val) + 1e-8)
            if rel_err < 0.01:   reward += 0.5
            elif rel_err < 0.05: reward += 0.2
            elif rel_err < 0.20: reward += 0.05
        except Exception:
            pass

    # imp3: format reward
    if IMPROVEMENT_3_FORMAT_REWARD:
        if has_boxed and is_correct(pred, gt):
            reward += 0.5
        elif has_boxed and pred is not None:
            reward += 0.15

    # imp4: step-wise reward
    if IMPROVEMENT_4_STEP_REWARD:
        math_lines = [
            ln for ln in response.split("\n")
            if len(ln.strip()) > 8 and re.search(r"[=+\-*/]", ln)
        ]
        reward += min(len(math_lines), 5) * 0.04

    return float(reward)


# ═════════════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    cache_file = os.path.join(DATASET_CACHE, "math_benchmark.json")
    if os.path.exists(cache_file):
        print(f"Loading dataset from cache: {cache_file}")
        with open(cache_file) as f:
            data = json.load(f)
        print(f"  {len(data)} samples loaded.")
    else:
        print("Downloading dataset...")
        ds   = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
        data = [{"problem": row["problem"], "solution": row["solution"]} for row in ds]
        with open(cache_file, "w") as f:
            json.dump(data, f)
        print(f"  Saved {len(data)} samples.")

    rng     = np.random.default_rng(SEED)
    indices = rng.permutation(len(data)).tolist()
    data    = [data[i] for i in indices]

    train_sample = data[0]
    eval_samples = data[1:51]

    print(f"  Train: {train_sample['problem'][:80]}...")
    print(f"  GT   : {extract_gt(train_sample['solution'])}")
    print(f"  Eval : {len(eval_samples)} problems")
    return train_sample, eval_samples


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_models():
    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    print(f"  Policy model | GPU: {gpu_mem()}")

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  Reference frozen | GPU: {gpu_mem()}")

    return tokenizer, model, ref_model


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT
# ═════════════════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, question: str):
    prompt_text = (
        "Solve the following math problem step by step.\n"
        "Write your reasoning carefully, and always end your response "
        "by putting your final answer inside \\boxed{}.\n\n"
        f"Problem: {question}\n\nSolution:"
    )
    msgs = [{"role": "user", "content": prompt_text}]
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt_text
    tok_len = tokenizer(text, return_tensors="pt").input_ids.shape[1]
    return text, tok_len


# ═════════════════════════════════════════════════════════════════════════════
# ROLLOUT SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

_HARD_GARBAGE = re.compile(
    r"(def\s+\w+\(|import\s+\w+|assert\b|\.py\b|[\u4e00-\u9fff]{3,})",
    re.IGNORECASE,
)

def _is_hard_garbage(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return True
    if _HARD_GARBAGE.search(text):
        return True
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio < 0.05

@torch.no_grad()
def sample_rollouts(model, tokenizer, prompt: str, prompt_len: int,
                    num_rollouts: int, temperature: float = None):
    """
    IMPROVEMENT 6 — TEMPERATURE ANNEALING:
    Accepts a per-step temperature so the caller can schedule it.
    High temperature early (exploration), low temperature late (exploitation).
    This matches the intuition: early steps need diverse rollouts to find
    ANY correct solution; later steps should exploit what's been learned.
    """
    temp = temperature if temperature is not None else TEMPERATURE

    model_was_train = model.training
    model.eval()

    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    results   = []

    for i in range(num_rollouts):
        try:
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=5,
                do_sample=True,
                temperature=temp,
                top_p=TOP_P,
                top_k=50,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            raw_text = tokenizer.decode(
                out[0][input_len:], skip_special_tokens=True
            ).strip()

            if _is_hard_garbage(raw_text):
                results.append((raw_text, "<garbage>", 0))
                continue

            ans     = extract_answer(raw_text)
            display = f"\\boxed{{{ans}}}" if ans else "<no-ans>"
            n_toks  = out.shape[1] - input_len   # response token count
            results.append((raw_text, display, n_toks))

        except RuntimeError as e:
            print(f"  ⚠ Rollout {i+1} error: {e}")
            cleanup()
            results.append(("", "<error>", 0))
        except Exception as e:
            print(f"  ⚠ Rollout {i+1} error: {e}")
            results.append(("", "<error>", 0))

    if model_was_train:
        model.train()

    return results   # list of (raw_text, display_answer, n_response_tokens)


# ═════════════════════════════════════════════════════════════════════════════
# POLICY LOSS
# ═════════════════════════════════════════════════════════════════════════════

def compute_policy_loss(model, tokenizer, prompt_text: str,
                        prompt_tok_len: int, raw_response: str,
                        advantage: float, n_response_tokens: int = None):
    """
    IMPROVEMENT 7 — LENGTH-NORMALIZED POLICY LOSS:
    Standard mean log-prob biases the gradient toward SHORT responses
    (fewer tokens = each token counts more). This causes the model to
    prefer brief (often wrong) answers over careful step-by-step reasoning.

    When imp7 is on: divide by sqrt(n_tokens) instead of n_tokens.
    sqrt normalization is a middle ground — it reduces the length bias
    without completely removing the sequence-level signal.

    Without imp7: loss = -adv * mean(log_prob)   ← standard
    With    imp7: loss = -adv * sum(log_prob) / sqrt(n)  ← length-normalized
    """
    if not raw_response.strip():
        return None

    full   = prompt_text + raw_response
    budget = prompt_tok_len + MAX_NEW_TOKENS + 32

    enc = tokenizer(
        full, return_tensors="pt",
        truncation=True, max_length=budget,
    ).to(model.device)
    ids = enc.input_ids

    prompt_enc_len = tokenizer(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=budget,
    ).input_ids.shape[1]

    if ids.shape[1] <= prompt_enc_len:
        return None

    outputs      = model(input_ids=ids, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = ids[:, 1:]

    if not torch.isfinite(shift_logits).all():
        return None

    log_probs   = F.log_softmax(shift_logits.float(), dim=-1)
    token_lp    = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    response_lp = token_lp[:, prompt_enc_len - 1:]

    if response_lp.numel() == 0:
        return None

    finite_mask = torch.isfinite(response_lp)
    if not finite_mask.any():
        return None
    response_lp = response_lp[finite_mask]

    if IMPROVEMENT_7_LENGTH_NORM and n_response_tokens and n_response_tokens > 1:
        # Length-normalized: sum / sqrt(n) reduces bias toward short responses
        loss_val = -float(advantage) * response_lp.sum() / (n_response_tokens ** 0.5)
    else:
        # Standard: mean log-prob
        loss_val = -float(advantage) * response_lp.mean()

    if not torch.isfinite(loss_val):
        return None
    return loss_val


def compute_kl(model, ref_model, tokenizer, prompt: str, response: str):
    full   = prompt + response
    budget = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) + MAX_NEW_TOKENS + 32
    enc    = tokenizer(full, return_tensors="pt",
                       truncation=True, max_length=budget).to(model.device)
    ids    = enc.input_ids
    p_len  = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    policy_logits = model(input_ids=ids, use_cache=False).logits
    with torch.no_grad():
        ref_logits = ref_model(
            input_ids=ids.to(ref_model.device), use_cache=False
        ).logits.to(model.device)

    sl           = slice(p_len - 1, -1)
    policy_log_p = F.log_softmax(policy_logits[:, sl, :].float(), dim=-1)
    ref_p        = F.softmax(ref_logits[:, sl, :].float(), dim=-1)
    kl = (ref_p * (ref_p.clamp(min=1e-8).log() - policy_log_p)).sum(-1).mean()
    return kl


# ═════════════════════════════════════════════════════════════════════════════
# GRPO ADVANTAGES — Leave-One-Out baseline
# ═════════════════════════════════════════════════════════════════════════════

def grpo_advantages(rewards: list) -> np.ndarray:
    """
    WHY LOO IS CRITICAL WITH K=8+:
    With K=4 and ~1 correct answer per step, LOO baseline for the correct
    rollout = mean of 3 wrong answers ≈ 0 → advantage ≈ 2.0 - 0 = 2.0 (fine).
    But for wrong rollouts: LOO baseline = mean including the one correct
    answer → baseline ≈ 0.5 → advantage ≈ 0 - 0.5 = -0.5 (weak).

    With K=8 and ~2 correct answers per step, LOO gives more reliable
    estimates across both correct and wrong rollouts, providing stronger
    and more consistent gradient signal.
    """
    r = np.array(rewards, dtype=np.float32)
    n = len(r)
    if n <= 1:
        return np.zeros_like(r)

    total    = r.sum()
    loo_mean = (total - r) / (n - 1)
    adv      = r - loo_mean
    std      = adv.std()
    if std < 1e-6:
        return np.zeros_like(r)

    adv = adv / (std + 1e-8)
    return np.clip(adv, -2.0, 2.0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, tokenizer, eval_samples) -> float:
    model.eval()
    correct = 0
    for s in eval_samples:
        prompt_text, prompt_tok_len = build_prompt(tokenizer, s["problem"])
        inputs = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=prompt_tok_len + 10,
        ).to(model.device)
        try:
            out = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_TOKENS,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                out[0][prompt_tok_len:], skip_special_tokens=True
            )
            if is_correct(extract_answer(response), s["solution"]):
                correct += 1
        except Exception as e:
            print(f"  ⚠ Eval error: {e}")
    model.train()
    return correct / len(eval_samples)


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train(model, ref_model, tokenizer, train_sample, eval_samples):
    PROMPT_TEXT, PROMPT_TOK_LEN = build_prompt(tokenizer, train_sample["problem"])
    GT     = train_sample["solution"]
    GT_VAL = extract_gt(GT)

    print(f"  Prompt tok len : {PROMPT_TOK_LEN}")
    print(f"  GT answer      : {GT_VAL}")

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-6, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=TRAIN_STEPS,
        pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=10,
    )
    model.train()

    history = {
        "step": [], "mean_reward": [], "train_acc": [],
        "eval_acc": [], "mean_kl": [], "grad_norm": [],
        "temperature": [], "n_updates": [],
    }

    print("\nPre-training eval...")
    pre_eval_acc = evaluate(model, tokenizer, eval_samples)
    print(f"Pre-training eval accuracy: {pre_eval_acc:.3f}")

    best_eval_acc  = pre_eval_acc
    patience_count = 0
    best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    n_total_updates = 0

    for step in range(1, TRAIN_STEPS + 1):
        cur_lr = scheduler.get_last_lr()[0]

        # ── IMPROVEMENT 6: Temperature annealing ──────────────────────────
        # Linear decay from TEMP_START → TEMP_END over all steps.
        # Early steps: high temp → diverse rollouts → more likely to find
        # correct solutions even if model isn't great yet.
        # Late steps: low temp → model exploits what it learned.
        if IMPROVEMENT_6_TEMP_ANNEAL:
            frac = (step - 1) / max(TRAIN_STEPS - 1, 1)
            cur_temp = TEMP_START - frac * (TEMP_START - TEMP_END)
        else:
            cur_temp = TEMPERATURE

        print(f"\n── Step {step}/{TRAIN_STEPS} | GPU: {gpu_mem()} | "
              f"LR: {cur_lr:.2e} | Temp: {cur_temp:.2f} ──")

        # ── Sample K rollouts ──────────────────────────────────────────────
        try:
            rollout_pairs = sample_rollouts(
                model, tokenizer, PROMPT_TEXT, PROMPT_TOK_LEN, K,
                temperature=cur_temp,
            )
        except Exception as e:
            print(f"  ⚠ Sampling failed: {e}")
            scheduler.step()
            continue

        for idx, (raw, disp, ntok) in enumerate(rollout_pairs):
            rp = raw[:70].replace('\n', ' ') if raw else '<empty>'
            print(f"  [{idx+1}] ans={disp}  toks={ntok}  raw={rp!r}")

        # ── Filter ────────────────────────────────────────────────────────
        valid = []
        for raw_text, display, n_toks in rollout_pairs:
            if display in ("<garbage>", "<error>") or not raw_text.strip():
                continue
            try:
                rw = compute_reward(raw_text, GT)
                if np.isfinite(rw):
                    valid.append((raw_text, rw, n_toks))
            except Exception:
                continue

        n_discarded = len(rollout_pairs) - len(valid)
        if n_discarded:
            print(f"  ⚠ {n_discarded}/{K} rollouts discarded")

        if len(valid) < 2:
            print(f"  ⚠ Only {len(valid)} valid rollouts — skipping")
            scheduler.step()
            continue

        raw_texts  = [t for t, _, _ in valid]
        rewards    = [r for _, r, _ in valid]
        n_toks_lst = [n for _, _, n in valid]
        advantages = grpo_advantages(rewards)

        step_correct   = sum(1 for r in raw_texts if is_correct(extract_answer(r), GT))
        step_train_acc = step_correct / len(raw_texts)

        n_correct  = step_correct
        n_wrong    = sum(1 for r in raw_texts
                         if extract_answer(r) is not None
                         and not is_correct(extract_answer(r), GT))
        n_no_ans   = len(raw_texts) - n_correct - n_wrong

        print(f"  Rewards   : {[f'{r:.3f}' for r in rewards]}")
        print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")
        print(f"  Train acc : {step_train_acc:.2f}  "
              f"(correct={n_correct} wrong={n_wrong} no-ans={n_no_ans})")

        # ── Compute losses ─────────────────────────────────────────────────
        losses  = []
        step_kl = 0.0

        for raw_response, adv, n_tok in zip(raw_texts, advantages, n_toks_lst):
            if abs(float(adv)) < 1e-8:
                continue

            pg = compute_policy_loss(
                model, tokenizer, PROMPT_TEXT, PROMPT_TOK_LEN,
                raw_response, float(adv), n_tok,
            )
            if pg is None:
                print(f"  ⚠ pg_loss=None")
                continue

            # KL regularization (always active at KL_BETA; extra if imp2)
            kl_beta = KL_BETA + (0.05 if IMPROVEMENT_2_KL_PENALTY else 0.0)
            try:
                kl = compute_kl(model, ref_model, tokenizer,
                                 PROMPT_TEXT, raw_response)
                if torch.isfinite(kl):
                    pg      = pg + kl_beta * kl
                    step_kl += kl.item()
            except Exception as e:
                print(f"  ⚠ KL failed: {e}")

            if torch.isfinite(pg):
                losses.append(pg)

        if not losses:
            print("  ⚠ No valid losses — skip update")
            scheduler.step()
            history["step"].append(step)
            history["mean_reward"].append(float(np.mean(rewards)))
            history["train_acc"].append(step_train_acc)
            history["mean_kl"].append(0.0)
            history["grad_norm"].append(0.0)
            history["temperature"].append(cur_temp)
            history["n_updates"].append(n_total_updates)
            history["eval_acc"].append((step, None))
            continue

        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        n_total_updates += 1

        mean_kl = step_kl / max(len(losses), 1)
        print(f"  Loss={total_loss.item():.5f} | KL={mean_kl:.4f} | "
              f"GradNorm={grad_norm:.3f} | Updates={n_total_updates}")

        history["step"].append(step)
        history["mean_reward"].append(float(np.mean(rewards)))
        history["train_acc"].append(step_train_acc)
        history["mean_kl"].append(mean_kl)
        history["grad_norm"].append(float(grad_norm))
        history["temperature"].append(cur_temp)
        history["n_updates"].append(n_total_updates)
        history["eval_acc"].append((step, None))

        

        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        del losses, total_loss
        cleanup()

    # NEW — single eval after all steps complete
    model.eval()
    print(f"\n📊 Running final evaluation after training...")
    final_eval_acc = evaluate(model, tokenizer, eval_samples)
    delta = final_eval_acc - pre_eval_acc
    sign  = "+" if delta >= 0 else ""
    print(f"🎉 Done. Final eval: {final_eval_acc:.3f}  ({sign}{delta:.3f} vs pre-train)")
    print(f"   Total optimizer updates: {n_total_updates}")
    return history, pre_eval_acc, final_eval_acc


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS + SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def save_results(history, pre_eval_acc, best_eval_acc=None):
    if not history["step"]:
        print("No steps recorded."); return

    eval_steps = [s for s, a in history["eval_acc"] if a is not None]
    eval_accs  = [a for s, a in history["eval_acc"] if a is not None]
    reported_best = best_eval_acc or (max(eval_accs) if eval_accs else None)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"Experiment: {EXPERIMENT_NAME}", fontsize=13, fontweight="bold")

    axes[0].plot(history["step"], history["mean_reward"], color="steelblue", lw=1.5)
    axes[0].set_title("Mean Reward"); axes[0].set_xlabel("Step"); axes[0].grid(alpha=0.3)

    w  = min(5, len(history["train_acc"]))
    sm = np.convolve(history["train_acc"], np.ones(w)/w, mode="valid")
    axes[1].plot(history["step"], history["train_acc"], alpha=0.3, color="orange")
    axes[1].plot(history["step"][w-1:], sm, color="orange", lw=2)
    axes[1].set_title("Train Acc"); axes[1].set_xlabel("Step")
    axes[1].set_ylim(-0.05, 1.05); axes[1].grid(alpha=0.3)

    if eval_steps:
        axes[2].plot(eval_steps, eval_accs, "o-", color="green", lw=2, ms=6)
    axes[2].axhline(pre_eval_acc, color="gray", ls="--",
                    label=f"Pre-train ({pre_eval_acc:.2f})")
    axes[2].set_title("Eval Acc (post-training)"); axes[2].set_xlabel("Step")
    axes[2].set_ylim(-0.05, 1.05); axes[2].legend(); axes[2].grid(alpha=0.3)
    # Add this to show the final eval as a single point at the last step
    if reported_best is not None:
        last_step = history["step"][-1] if history["step"] else TRAIN_STEPS
        axes[2].scatter([last_step], [reported_best], color="green",
                        s=100, zorder=5, label=f"Final ({reported_best:.3f})")
        axes[2].legend()

    if history.get("grad_norm"):
        axes[3].plot(history["step"], history["grad_norm"], color="red", lw=1.5)
        axes[3].axhline(GRAD_CLIP, color="gray", ls="--", label=f"clip={GRAD_CLIP}")
        axes[3].set_title("Grad Norm"); axes[3].set_xlabel("Step")
        axes[3].legend(); axes[3].grid(alpha=0.3)

    if history.get("temperature"):
        axes[4].plot(history["step"], history["temperature"], color="purple", lw=1.5)
        axes[4].set_title("Temperature"); axes[4].set_xlabel("Step")
        axes[4].set_ylim(0, 1.2); axes[4].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_results.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    n = len(history["train_acc"])
    final_train_acc = float(np.mean(history["train_acc"][max(0, n-5):])) if n else 0.0
    n_updates = history["n_updates"][-1] if history.get("n_updates") else "?"

    print("\n" + "=" * 55)
    print(f"  EXPERIMENT SUMMARY: {EXPERIMENT_NAME}")
    print("=" * 55)
    print(f"  Pre-training eval acc    : {pre_eval_acc:.3f}")
    print(f"  Final train acc (last 5) : {final_train_acc:.3f}")
    if reported_best is not None:
        delta = reported_best - pre_eval_acc
        sign  = "+" if delta >= 0 else ""
        print(f"  Best eval acc            : {reported_best:.3f}")
        print(f"  Delta (vs pre-training)  : {sign}{delta:.3f}")
    print(f"  Total optimizer updates  : {n_updates}")
    print("=" * 55)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "flags": {
            "sympy_reward":  IMPROVEMENT_1_SYMPY_REWARD,
            "kl_penalty":    IMPROVEMENT_2_KL_PENALTY,
            "format_reward": IMPROVEMENT_3_FORMAT_REWARD,
            "step_reward":   IMPROVEMENT_4_STEP_REWARD,
            "more_rollouts": IMPROVEMENT_5_MORE_ROLLOUTS,
            "temp_anneal":   IMPROVEMENT_6_TEMP_ANNEAL,
            "length_norm":   IMPROVEMENT_7_LENGTH_NORM,
        },
        "hyperparams": {
            "lr": LR, "K": K, "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE, "kl_beta": KL_BETA, "grad_clip": GRAD_CLIP,
        },
        "pre_eval_acc":      pre_eval_acc,
        "final_train_acc":   final_train_acc,
        "final_eval_acc":    reported_best,
        "reward_history":    history["mean_reward"],
        "train_acc_history": history["train_acc"],
        "grad_norm_history": history.get("grad_norm", []),
    }
    json_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {json_path}")


def print_comparison_table():
    summaries = []
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_summary.json"))):
        try:
            with open(f) as fh:
                summaries.append(json.load(fh))
        except Exception:
            pass
    if not summaries:
        print("No summaries found."); return

    print("\n" + "=" * 85)
    print(f"  {'EXPERIMENT':<22} {'Pre':>6} {'Eval':>6} {'Delta':>7}  Flags")
    print("=" * 85)
    for s in summaries:
        pre   = s.get("pre_eval_acc", 0)
        ev    = s.get("final_eval_acc") or 0.0
        delta = ev - pre
        sign  = "+" if delta >= 0 else ""
        flags = ", ".join(k for k, v in s.get("flags", {}).items() if v) or "none"
        print(f"  {s['experiment']:<22} {pre:>6.3f} {ev:>6.3f}"
              f" {sign+f'{delta:.3f}':>7}  [{flags}]")
    print("=" * 85)


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(exp_name, flags, train_sample, eval_samples):
    global EXPERIMENT_NAME
    global IMPROVEMENT_1_SYMPY_REWARD, IMPROVEMENT_2_KL_PENALTY
    global IMPROVEMENT_3_FORMAT_REWARD, IMPROVEMENT_4_STEP_REWARD
    global IMPROVEMENT_5_MORE_ROLLOUTS, IMPROVEMENT_6_TEMP_ANNEAL
    global IMPROVEMENT_7_LENGTH_NORM, K

    EXPERIMENT_NAME             = exp_name
    IMPROVEMENT_1_SYMPY_REWARD  = flags.get("sympy",    False)
    IMPROVEMENT_2_KL_PENALTY    = flags.get("kl",       False)
    IMPROVEMENT_3_FORMAT_REWARD = flags.get("format",   False)
    IMPROVEMENT_4_STEP_REWARD   = flags.get("step",     False)
    IMPROVEMENT_5_MORE_ROLLOUTS = flags.get("rollouts", False)
    IMPROVEMENT_6_TEMP_ANNEAL   = flags.get("anneal",   False)
    IMPROVEMENT_7_LENGTH_NORM   = flags.get("lennorm",  False)
    K = 16 if IMPROVEMENT_5_MORE_ROLLOUTS else BASE_ROLLOUTS

    print("\n" + "=" * 60)
    print(f"RUNNING: {EXPERIMENT_NAME}  flags={flags}  K={K}")
    print("=" * 60)

    tokenizer, model, ref_model = load_models()
    history, pre_eval_acc, best_eval_acc = train(
        model, ref_model, tokenizer, train_sample, eval_samples
    )
    save_results(history, pre_eval_acc, best_eval_acc)

    del model, ref_model
    cleanup()
    print(f"GPU after cleanup: {gpu_mem()}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    train_sample, eval_samples = load_data()

    experiments = [
        ("imp1_sympy",    {"sympy": True}),
        ("imp2_kl",       {"kl": True}),
        ("imp4_step",     {"step": True}),
        ("imp6_anneal",   {"anneal": True}),
        ("optimal_combo", {
            "sympy": True, 
            "kl": True, 
            "step": True, 
            "anneal": True
        }),
    ]

    for name, flags in experiments:
        run_experiment(name, flags, train_sample, eval_samples)

    print_comparison_table()