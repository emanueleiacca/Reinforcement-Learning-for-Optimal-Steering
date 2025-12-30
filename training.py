# ============================================================
# training.py (FROM SCRATCH) — PPO Steering for SD1.5
# ------------------------------------------------------------
# Inputs:
#   - /kaggle/working/i2p_train_processed_1000.jsonl
#   - /kaggle/working/P_pca_32.pt   (contains P and delta_mean)
# Outputs:
#   - /kaggle/working/ppo_train_log.jsonl
#   - /kaggle/working/ppo_ckpt.pt
#   - /kaggle/working/ppo_best.pt
#   - /kaggle/working/ppo_samples/...
#
# Key upgrades:
#   - Uses delta_mean + P@a
#   - Adds gating scalar g in (0,1)
#   - Fixed train/eval split
#   - Eval drift metric (baseline vs steered)
# ============================================================

import os, json, math, random, gc
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


# -------------------------
# 0) Config
# -------------------------

@dataclass
class CFG:
    WORKDIR: str = "/kaggle/working"
    JSONL_PATH: str = "/kaggle/working/i2p_train_processed_1000.jsonl"
    PCA_PATH: str = "/kaggle/working/P_pca_32.pt"

    LOG_PATH: str = "/kaggle/working/ppo_train_log.jsonl"
    CKPT_PATH: str = "/kaggle/working/ppo_ckpt.pt"
    BEST_PATH: str = "/kaggle/working/ppo_best.pt"
    SAMPLES_DIR: str = "/kaggle/working/ppo_samples"

    MODEL_ID: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    SEED: int = 123
    TRAIN_POOL: int = 900
    EVAL_POOL: int = 100

    # Diffusion
    NUM_INFER_STEPS: int = 30
    GUIDANCE_SCALE: float = 7.5

    # Latents
    LATENT_C: int = 4
    LATENT_H: int = 64
    LATENT_W: int = 64
    LATENT_DIM: int = 4 * 64 * 64

    # Action
    ACTION_DIM: int = 32
    GATE_DIM: int = 1  # extra scalar
    ACTOR_OUT_DIM: int = 33  # ACTION_DIM + GATE_DIM

    # Steering schedule
    APPLY_FROM_STEP: int = 5
    LAMBDA: float = 1.0
    MAX_RATIO: float = 0.02

    # Reward weights
    W_UNSAFE: float = 1.0         # terminal unsafe penalty
    W_ACT_STEP: float = 0.01      # dense action penalty
    W_GATE_STEP: float = 0.001    # small dense gate penalty

    # PPO
    TOTAL_EPISODES: int = 400
    EPISODES_PER_UPDATE: int = 8
    PPO_EPOCHS: int = 4
    PPO_BATCH: int = 256
    GAMMA: float = 0.99
    LAM: float = 0.95
    CLIP_EPS: float = 0.2
    VF_COEF: float = 0.5
    ENT_COEF: float = 0.01
    LR: float = 3e-4
    MAX_GRAD_NORM: float = 1.0

    # Eval
    EVAL_EVERY: int = 20
    N_EVAL: int = 8


cfg = CFG()
os.makedirs(cfg.WORKDIR, exist_ok=True)
os.makedirs(cfg.SAMPLES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_UNET = torch.float16 if DEVICE == "cuda" else torch.float32
DTYPE_POLICY = torch.float32

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(cfg.SEED)

def gpu_mem_str():
    if DEVICE != "cuda":
        return "cpu"
    alloc = torch.cuda.memory_allocated() / (1024**3)
    rsvd  = torch.cuda.memory_reserved() / (1024**3)
    return f"{alloc:.2f}G alloc | {rsvd:.2f}G rsvd"

print(f"[INFO] DEVICE={DEVICE} | mem={gpu_mem_str()}")
print(f"[INFO] JSONL={cfg.JSONL_PATH}")
print(f"[INFO] PCA={cfg.PCA_PATH}")


# -------------------------
# 1) Load dataset + fixed split
# -------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out

def is_usable(e: Dict[str, Any]) -> bool:
    if not (e.get("prompt_raw") or "").strip():
        return False
    if e.get("sd_seed") is None:
        return False
    try:
        _ = int(e["sd_seed"])
    except Exception:
        return False
    return True

entries = [e for e in load_jsonl(cfg.JSONL_PATH) if is_usable(e)]
assert len(entries) >= cfg.TRAIN_POOL + cfg.EVAL_POOL, f"Need >= {cfg.TRAIN_POOL+cfg.EVAL_POOL}, got {len(entries)}"
print(f"[DATA] usable entries: {len(entries)}")

rng = random.Random(cfg.SEED)
rng.shuffle(entries)
train_set = entries[:cfg.TRAIN_POOL]
eval_set  = entries[cfg.TRAIN_POOL:cfg.TRAIN_POOL + cfg.EVAL_POOL]

split_path = os.path.join(cfg.WORKDIR, f"fixed_split_seed{cfg.SEED}_train{cfg.TRAIN_POOL}_eval{cfg.EVAL_POOL}.json")
with open(split_path, "w", encoding="utf-8") as f:
    json.dump({
        "seed": cfg.SEED,
        "train_hf_idx": [x.get("hf_idx") for x in train_set],
        "eval_hf_idx":  [x.get("hf_idx") for x in eval_set],
    }, f, ensure_ascii=False, indent=2)
print(f"[DATA] fixed split saved to {split_path}")


# -------------------------
# 2) Load PCA (P + delta_mean)
# -------------------------

def load_pca(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "P" not in obj:
        raise ValueError("PCA checkpoint must be a dict with key 'P'.")

    P = obj["P"].to(torch.float32)
    delta_mean = obj.get("delta_mean", None)
    if delta_mean is None:
        print("[WARN] delta_mean missing; using zeros.")
        delta_mean = torch.zeros((cfg.LATENT_DIM,), dtype=torch.float32)
    else:
        delta_mean = delta_mean.to(torch.float32)

    if P.shape == (cfg.ACTION_DIM, cfg.LATENT_DIM):
        P = P.t()

    assert P.shape == (cfg.LATENT_DIM, cfg.ACTION_DIM), f"P shape {P.shape} != ({cfg.LATENT_DIM},{cfg.ACTION_DIM})"
    assert delta_mean.shape == (cfg.LATENT_DIM,), f"delta_mean shape {delta_mean.shape} != ({cfg.LATENT_DIM},)"

    # Normalize columns for stability
    P = P / (P.norm(dim=0, keepdim=True) + 1e-8)

    return P.to(DEVICE, dtype=DTYPE_POLICY), delta_mean.to(DEVICE, dtype=DTYPE_POLICY)

P, delta_mean = load_pca(cfg.PCA_PATH)
print(f"[PCA] P={tuple(P.shape)} delta_mean={tuple(delta_mean.shape)}")


# -------------------------
# 3) Load SD pipeline
# -------------------------

print(f"[SD] loading {cfg.MODEL_ID}")
pipe = StableDiffusionPipeline.from_pretrained(cfg.MODEL_ID, torch_dtype=DTYPE_UNET)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing("max")
try:
    pipe.enable_vae_slicing()
except Exception:
    pass

pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

print("[SD] ready")

# Prompt safety truncation
def truncate_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    tok = pipe.tokenizer
    ids = tok(prompt, truncation=True, max_length=tok.model_max_length, return_tensors="pt").input_ids[0]
    return tok.decode(ids, skip_special_tokens=True)

@torch.no_grad()
def encode_cfg(prompt: str):
    prompt = truncate_prompt(prompt)
    pos, neg = pipe.encode_prompt(
        prompt=prompt,
        device=DEVICE,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )
    return pos, neg, prompt

def unique_consecutive(xs):
    out, last = [], None
    for x in xs:
        xi = int(x)
        if xi != last:
            out.append(x)
        last = xi
    return out

@torch.no_grad()
def decode_latents(latents: torch.Tensor) -> Image.Image:
    z = latents / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(z.to(pipe.vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img_np = img[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img_np * 255).astype(np.uint8))

def l2(x: torch.Tensor) -> float:
    return float(x.float().norm().item())


# -------------------------
# 4) Safety scorer (CPU)
# -------------------------

class CLIPSafetyScorer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    @torch.no_grad()
    def unsafe_flag(self, pil_img: Image.Image) -> float:
        img = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        _, has_nsfw = self.checker(images=[img], clip_input=inputs.pixel_values)
        return float(bool(has_nsfw[0]))

scorer = CLIPSafetyScorer()
print("[SAFETY] scorer ready on CPU")


# -------------------------
# 5) UNet feature tap + state encoder
# -------------------------

class UNetFeatureTap:
    def __init__(self, unet):
        self.cache = {}
        self.hooks = []
        def hook_mid(m, inp, out): self.cache["mid"] = out
        def hook_down(m, inp, out): self.cache["down"] = out
        self.hooks.append(unet.mid_block.register_forward_hook(hook_mid))
        self.hooks.append(unet.down_blocks[-1].register_forward_hook(hook_down))

    def clear(self):
        self.cache.clear()

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

tap = UNetFeatureTap(pipe.unet)

class StateEncoder(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.out_dim = out_dim
        self.proj = None

    def _as_tensor(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        return x

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        mid = self._as_tensor(feats["mid"])
        down = self._as_tensor(feats["down"])

        mid_vec = mid.mean(dim=(2,3)) if mid.dim() == 4 else mid
        down_vec = down.mean(dim=(2,3)) if down.dim() == 4 else down

        x = torch.cat([down_vec, mid_vec], dim=1).to(DTYPE_POLICY)

        if self.proj is None:
            self.proj = nn.Linear(x.shape[1], self.out_dim).to(x.device).to(DTYPE_POLICY)
            print(f"[STATE] proj init: in={x.shape[1]} -> out={self.out_dim}")

        return self.proj(x)

encoder = StateEncoder(out_dim=1024).to(DEVICE).to(DTYPE_POLICY)


# -------------------------
# 6) Actor-Critic (squashed Gaussian)
# -------------------------

class SquashedGaussianActorCritic(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden: int = 1024):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def act(self, s: torch.Tensor):
        h = self.actor(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = log_std.exp()

        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)

        # log prob (tanh corrected)
        logp = (-0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2*log_std + math.log(2*math.pi))).sum(dim=-1, keepdim=True)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        v = self.critic(s)
        return a, logp, v

    def evaluate(self, s: torch.Tensor, a: torch.Tensor):
        h = self.actor(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = log_std.exp()

        a_cl = a.clamp(-0.999999, 0.999999)
        atanh = 0.5 * torch.log((1 + a_cl) / (1 - a_cl))

        logp = (-0.5 * (((atanh - mu) / (std + 1e-8)) ** 2 + 2*log_std + math.log(2*math.pi))).sum(dim=-1, keepdim=True)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        ent = (0.5 * (1 + math.log(2*math.pi)) + log_std).sum(dim=-1, keepdim=True)
        v = self.critic(s)
        return logp, ent, v

model = SquashedGaussianActorCritic(1024, cfg.ACTOR_OUT_DIM, hidden=1024).to(DEVICE).to(DTYPE_POLICY)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR)


# -------------------------
# 7) PPO helpers (GAE)
# -------------------------

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float):
    # rewards, values, dones are [T]
    T = rewards.shape[0]
    adv = torch.zeros(T, device=rewards.device, dtype=torch.float32)
    last = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
        next_value = values[t]
    ret = adv + values
    return adv, ret


# -------------------------
# 8) Rollout (sequential CFG + steering)
# -------------------------

@torch.no_grad()
def rollout_one(prompt_raw: str, seed: int, do_steer: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Image.Image, float, float, str]:
    """
    Returns:
      S [T,1024] (empty if do_steer False)
      A [T,33]   (empty if do_steer False)
      LOGP [T,1] (empty if do_steer False)
      V [T,1]    (empty if do_steer False)
      pil (final image)
      unsafe_flag (0/1)
      R_total (scalar)
      prompt_trunc (str)
    """
    pos_emb, neg_emb, prompt_trunc = encode_cfg(prompt_raw)

    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    latents = torch.randn((1, cfg.LATENT_C, cfg.LATENT_H, cfg.LATENT_W), generator=g, device=DEVICE, dtype=DTYPE_UNET)
    latents = latents * pipe.scheduler.init_noise_sigma

    pipe.scheduler.set_timesteps(cfg.NUM_INFER_STEPS, device=DEVICE)
    timesteps = unique_consecutive(pipe.scheduler.timesteps)[:cfg.NUM_INFER_STEPS]

    # sigma schedule derived from alphas_cumprod (DDIM-like proxy)
    sigmas = []
    for t in timesteps:
        ti = int(t)
        alpha_bar = float(pipe.scheduler.alphas_cumprod[ti].item())
        sigmas.append(math.sqrt(max(1.0 - alpha_bar, 0.0)))

    T = len(timesteps)
    rewards = torch.zeros(T, device=DEVICE, dtype=torch.float32)
    dones = torch.zeros(T, device=DEVICE, dtype=torch.float32)
    dones[-1] = 1.0

    states, actions, logps, values = [], [], [], []

    for i, t in enumerate(timesteps):
        latent_in = pipe.scheduler.scale_model_input(latents, t)

        # uncond
        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            eps_u = pipe.unet(latent_in, t, encoder_hidden_states=neg_emb, return_dict=False)[0]

        # cond + tap
        tap.clear()
        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            eps_c = pipe.unet(latent_in, t, encoder_hidden_states=pos_emb, return_dict=False)[0]

        eps = eps_u + cfg.GUIDANCE_SCALE * (eps_c - eps_u)

        if do_steer:
            feats = {"mid": tap.cache["mid"], "down": tap.cache["down"]}
            s = encoder(feats)

            a_full, logp, v = model.act(s)          # [1,33]
            a32 = a_full[..., :cfg.ACTION_DIM]      # [1,32]
            g_logit = a_full[..., cfg.ACTION_DIM:]  # [1,1]
            gate = torch.sigmoid(g_logit)           # [1,1]

            # dense penalties
            rewards[i] += -cfg.W_ACT_STEP * a32.detach().pow(2).mean()
            rewards[i] += -cfg.W_GATE_STEP * gate.detach().pow(2).mean()

            # delta_mean + P@a
            delta_flat = delta_mean + (P @ a32.view(-1))  # [16384]
            delta = delta_flat.view(1, 4, 64, 64).to(eps.dtype)

            sigma_t = sigmas[i]
            delta_term = (cfg.LAMBDA * sigma_t) * delta

            # apply gate
            delta_term = delta_term * gate.to(delta_term.dtype)

            # ratio clip (stability)
            ratio = l2(delta_term) / (l2(eps) + 1e-8)
            if ratio > cfg.MAX_RATIO:
                delta_term = delta_term * (cfg.MAX_RATIO / (ratio + 1e-8))

            if i >= cfg.APPLY_FROM_STEP:
                eps = eps + delta_term

            states.append(s.squeeze(0).float())
            actions.append(a_full.squeeze(0).float())
            logps.append(logp.squeeze(0).float())
            values.append(v.squeeze(0).float())

        latents = pipe.scheduler.step(eps, t, latents, return_dict=False)[0]

        # reduce memory
        del latent_in, eps_u, eps_c, eps
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    pil = decode_latents(latents)
    unsafe = scorer.unsafe_flag(pil)
    rewards[-1] += -cfg.W_UNSAFE * unsafe

    R_total = float(rewards.sum().item())

    if do_steer:
        S = torch.stack(states).to(DEVICE)
        A = torch.stack(actions).to(DEVICE)
        LOGP = torch.stack(logps).to(DEVICE).unsqueeze(-1)
        V = torch.stack(values).to(DEVICE).unsqueeze(-1)
    else:
        S = torch.empty((0, 1024), device=DEVICE)
        A = torch.empty((0, cfg.ACTOR_OUT_DIM), device=DEVICE)
        LOGP = torch.empty((0, 1), device=DEVICE)
        V = torch.empty((0, 1), device=DEVICE)

    return S, A, LOGP, V, pil, float(unsafe), R_total, prompt_trunc


# -------------------------
# 9) Evaluation (with drift)
# -------------------------

@torch.no_grad()
def eval_agent(episode: int) -> Dict[str, float]:
    Rs, unsafe_flags, drifts = [], [], []

    for k in range(min(cfg.N_EVAL, len(eval_set))):
        e = eval_set[k]
        prompt = e["prompt_raw"]
        seed = int(e["sd_seed"])

        # baseline (no steer)
        _, _, _, _, pil0, _, _, _ = rollout_one(prompt, seed, do_steer=False)
        # steered
        _, _, _, _, pil1, unsafe, R, prompt_trunc = rollout_one(prompt, seed, do_steer=True)

        # drift proxy
        a0 = np.array(pil0.resize((64,64))).astype(np.float32)
        a1 = np.array(pil1.resize((64,64))).astype(np.float32)
        drift = float(np.mean(np.abs(a0 - a1)) / 255.0)

        Rs.append(R)
        unsafe_flags.append(unsafe)
        drifts.append(drift)

        fp = os.path.join(cfg.SAMPLES_DIR, f"eval_ep{episode:04d}_k{k:02d}_unsafe{unsafe:.0f}_drift{drift:.3f}.png")
        pil1.save(fp)

    return {
        "eval_mean_R": float(np.mean(Rs)) if Rs else 0.0,
        "eval_unsafe_rate": float(np.mean(unsafe_flags)) if unsafe_flags else 0.0,
        "eval_mean_drift": float(np.mean(drifts)) if drifts else 0.0,
    }


# -------------------------
# 10) Training loop (PPO)
# -------------------------

# init log file
with open(cfg.LOG_PATH, "w", encoding="utf-8") as f:
    pass

best_eval_R = -1e9
buffer_S, buffer_A, buffer_LOGP, buffer_V, buffer_R, buffer_D = [], [], [], [], [], []

print(f"[TRAIN] start | mem={gpu_mem_str()}")

pbar = tqdm(range(1, cfg.TOTAL_EPISODES + 1), desc="PPO", unit="ep")

for ep in pbar:
    entry = rng.choice(train_set)
    prompt = entry["prompt_raw"]
    seed = int(entry["sd_seed"])

    S, A, LOGP, V, pil, unsafe, R_total, prompt_trunc = rollout_one(prompt, seed, do_steer=True)

    # Build per-step rewards/dones for PPO
    # We stored rewards internally; here we approximate advantage using the terminal R_total equally across steps.
    # (You can extend to store per-step rewards if you want; this keeps it simple and stable.)
    T = S.shape[0]
    if T == 0:
        continue

    # For PPO, we use a per-step reward vector with terminal reward at last step
    r_vec = torch.zeros((T,), device=DEVICE, dtype=torch.float32)
    r_vec[-1] = torch.tensor(R_total, device=DEVICE, dtype=torch.float32)
    d_vec = torch.zeros((T,), device=DEVICE, dtype=torch.float32)
    d_vec[-1] = 1.0

    buffer_S.append(S)
    buffer_A.append(A)
    buffer_LOGP.append(LOGP)
    buffer_V.append(V.squeeze(-1))
    buffer_R.append(r_vec)
    buffer_D.append(d_vec)

    # logging
    a_mean = float(A[:, :cfg.ACTION_DIM].norm(dim=-1).mean().item())
    g_mean = float(torch.sigmoid(A[:, cfg.ACTION_DIM:]).mean().item())
    row = {
        "episode": ep,
        "hf_idx": entry.get("hf_idx"),
        "seed": seed,
        "unsafe": float(unsafe),
        "R": float(R_total),
        "a_mean": a_mean,
        "g_mean": g_mean,
        "mem": gpu_mem_str(),
    }
    with open(cfg.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    # PPO update
    if len(buffer_S) >= cfg.EPISODES_PER_UPDATE:
        S_all = torch.cat(buffer_S, dim=0)
        A_all = torch.cat(buffer_A, dim=0)
        old_logp_all = torch.cat(buffer_LOGP, dim=0).detach()
        V_all = torch.cat(buffer_V, dim=0).detach()
        R_all = torch.cat(buffer_R, dim=0)
        D_all = torch.cat(buffer_D, dim=0)

        buffer_S.clear(); buffer_A.clear(); buffer_LOGP.clear(); buffer_V.clear(); buffer_R.clear(); buffer_D.clear()

        adv, ret = compute_gae(R_all, V_all, D_all, gamma=cfg.GAMMA, lam=cfg.LAM)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = S_all.shape[0]
        for _ in range(cfg.PPO_EPOCHS):
            idx = torch.randperm(N, device=DEVICE)
            for start in range(0, N, cfg.PPO_BATCH):
                mb = idx[start:start+cfg.PPO_BATCH]
                s_mb = S_all[mb]
                a_mb = A_all[mb]
                oldlp_mb = old_logp_all[mb]
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                logp_new, ent, v_new = model.evaluate(s_mb, a_mb)
                ratio = torch.exp(logp_new - oldlp_mb)

                surr1 = ratio.squeeze(-1) * adv_mb
                surr2 = torch.clamp(ratio.squeeze(-1), 1.0-cfg.CLIP_EPS, 1.0+cfg.CLIP_EPS) * adv_mb
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v = F.mse_loss(v_new.squeeze(-1), ret_mb)
                loss = loss_pi + cfg.VF_COEF * loss_v - cfg.ENT_COEF * ent.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                opt.step()

        # cleanup
        del S_all, A_all, old_logp_all, V_all, R_all, D_all, adv, ret
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # eval
    if ep % cfg.EVAL_EVERY == 0:
        metrics = eval_agent(ep)
        print(f"\n[EVAL] ep={ep} {metrics}")

        # save best by mean_R
        if metrics["eval_mean_R"] > best_eval_R:
            best_eval_R = metrics["eval_mean_R"]
            torch.save({
                "episode": ep,
                "best_eval_R": best_eval_R,
                "cfg": cfg.__dict__,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
            }, cfg.BEST_PATH)
            print(f"[BEST] saved {cfg.BEST_PATH} (best_eval_R={best_eval_R:.4f})")

        # save rolling ckpt
        torch.save({
            "episode": ep,
            "best_eval_R": best_eval_R,
            "cfg": cfg.__dict__,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        }, cfg.CKPT_PATH)

    pbar.set_postfix({
        "unsafe": f"{unsafe:.0f}",
        "R": f"{R_total:+.2f}",
        "a": f"{a_mean:.2f}",
        "g": f"{g_mean:.2f}",
        "mem": gpu_mem_str(),
    })

print("[DONE] log:", cfg.LOG_PATH)
print("[DONE] ckpt:", cfg.CKPT_PATH)
print("[DONE] best:", cfg.BEST_PATH)
print("[DONE] samples:", cfg.SAMPLES_DIR)
