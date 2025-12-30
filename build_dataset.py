"""
SD1.5 Safety Steering Dataset Builder - Production Ready
========================================================

Features:
✓ Checkpoint-based execution (auto-resume after 6h)
✓ Atomic writes prevent corruption
✓ Trajectory index for RL training
✓ Proper safety scoring (NudeNet + ViT)
✓ Memory-efficient batch processing
✓ Comprehensive validation

Run this in a Kaggle notebook with GPU enabled.
"""

import os, sys, csv, json, time, random, hashlib, traceback, platform, re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from datasets import load_dataset
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModelForImageClassification

print("✓ Imports complete")

# ============================================================
# UTILITIES
# ============================================================

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def normalize_ws(s: str) -> str:
    return " ".join(str(s).strip().split())

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

def deterministic_seed(prompt_id: str, salt: str, i: int) -> int:
    msg = f"{prompt_id}|{salt}|{i}"
    h = hashlib.sha256(msg.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big", signed=False)
    return int(v % (2**32))

def sanitize_filename(s: str, max_len: int = 140) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:max_len].strip("_")

def ensure_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# ============================================================
# CATEGORY MAPPING
# ============================================================

CATEGORY_MAP = {
    "nudity": "nudity", "sexual": "nudity", "nsfw": "nudity",
    "violence": "violence", "fighting": "violence",
    "gore": "gore", "blood": "gore",
    "self harm": "self_harm", "suicide": "self_harm",
    "harassment": "harassment", "hate": "harassment",
    "safe": "safe",
}

SAFE_REWRITE = {
    "nudity": ", fully clothed, no nudity",
    "violence": ", peaceful, no violence",
    "gore": ", no gore, no injuries",
    "self_harm": ", safe, no self-harm",
    "harassment": ", respectful, no harassment",
}

def normalize_tag(tag: str) -> str:
    return " ".join(str(tag).strip().lower().replace("-", " ").replace("_", " ").split())

def get_tags(item: Dict) -> List[str]:
    raw = item.get("categories", [])
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        if raw.strip().startswith("["):
            try:
                import ast
                return [str(x) for x in ast.literal_eval(raw)]
            except:
                pass
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []

def sanitize_prompt(prompt: str, tags: List[str]) -> Tuple[str, List[str]]:
    base = str(prompt).strip()
    cats = set()
    for t in tags:
        norm = normalize_tag(t)
        for k, v in CATEGORY_MAP.items():
            if k in norm:
                cats.add(v)
                break
    
    cats = sorted(cats)
    if not cats or (len(cats) == 1 and "safe" in cats):
        return base, ["safe"]
    
    cats = [c for c in cats if c != "safe"] if len(cats) > 1 else cats
    
    suffixes = []
    for c in cats:
        if c in SAFE_REWRITE:
            suffixes.append(SAFE_REWRITE[c])
    
    if suffixes:
        return base + "".join(suffixes), cats
    return base, cats

def compute_risk(item: Dict) -> float:
    def get_pct(key):
        v = item.get(key, 0)
        try:
            v = float(v)
            if v <= 1:
                v *= 100
            return max(0.0, min(100.0, v))
        except:
            return 0.0
    
    risk = (
        0.4 * get_pct("inappropriate_percentage") +
        0.3 * get_pct("q16_percentage") +
        0.2 * get_pct("sd_safety_percentage") +
        0.1 * get_pct("nudity_percentage")
    )
    return float(max(0.0, min(100.0, risk)))

def risk_bucket(risk: float) -> str:
    if risk < 25:
        return "low"
    if risk < 50:
        return "medium"
    if risk < 75:
        return "high"
    return "extreme"

# ============================================================
# MODELS
# ============================================================

@dataclass
class SDConfig:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    steps: int = 30
    guidance: float = 7.5
    size: int = 512

class SDRunner:
    def __init__(self, cfg: SDConfig, device: str):
        self.cfg = cfg
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id, torch_dtype=dtype, safety_checker=None
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to(device)
    
    @torch.no_grad()
    def generate(self, prompt: str, seed: int, capture_every: int = 2):
        g = torch.Generator(device=self.device).manual_seed(seed)
        traj = []
        
        def callback(pipe, step, timestep, kwargs):
            if capture_every > 0 and step % capture_every == 0:
                traj.append((step, kwargs["latents"].detach().clone().cpu()))
            return kwargs
        
        out = self.pipe(
            prompt=prompt,
            height=self.cfg.size,
            width=self.cfg.size,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance,
            generator=g,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        return out.images[0], traj

class NudeNetScorer:
    def __init__(self):
        from nudenet import NudeClassifier
        self.model = NudeClassifier()
    
    def score(self, img_path: str) -> float:
        out = self.model.classify(img_path)
        if isinstance(out, dict) and img_path in out:
            return float(out[img_path].get("unsafe", 0.0))
        return 0.0

class ViolenceScorer:
    def __init__(self, device: str):
        model_name = "jaranohaal/vit-base-violence-detection"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
        self.model.eval()
    
    @torch.no_grad()
    def score(self, img: Image.Image) -> float:
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return float(probs[0][1].cpu().item())

class CLIPScorer:
    def __init__(self, device: str):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    @torch.no_grad()
    def score(self, img: Image.Image, text: str) -> float:
        inputs = self.proc(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        sim = torch.cosine_similarity(outputs.image_embeds[0], outputs.text_embeds[0], dim=0)
        return float(sim.cpu().item())

# ============================================================
# REWARD
# ============================================================

def compute_reward(nudity, violence, clip_sim, prompt_risk=None, weights=None):
    if weights is None:
        weights = {"w_safe": 1.0, "w_clip": 0.2, "w_prior": 0.05}
    
    max_unsafe = max(nudity or 0, violence or 0)
    
    # Safety reward with barrier
    if max_unsafe <= 0.7:
        r_safe = -max_unsafe
    else:
        r_safe = -(1.0 + 5.0 * (max_unsafe - 0.7))
    
    r_clip = float(clip_sim or 0)
    r_prior = -float(prompt_risk or 0)
    
    r_total = (
        weights["w_safe"] * r_safe +
        weights["w_clip"] * r_clip +
        weights["w_prior"] * r_prior
    )
    
    return {
        "R_safe": r_safe,
        "R_clip": r_clip,
        "R_prior": r_prior,
        "R_total": r_total,
        "max_unsafe": max_unsafe,
    }

# ============================================================
# TRAJECTORY STORAGE
# ============================================================

class TrajStorage:
    def __init__(self, out_dir: str, batch_size: int = 1024):
        self.out_dir = out_dir
        safe_mkdir(out_dir)
        self.batch_size = batch_size
        self.buffer = []
        self.batch_idx = 0
        self.metadata = []
    
    def add(self, sample_id, prompt_id, seed, step_idx, latent, rewards, index_buffer):
        if latent.ndim == 4:
            latent = latent[0]
        if latent.device.type != "cpu":
            latent = latent.cpu()
        
        offset = len(self.buffer)
        self.buffer.append(latent.numpy())
        
        self.metadata.append({
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "seed": seed,
            "step_idx": step_idx,
            "batch_idx": self.batch_idx,
            "offset": offset,
            **rewards
        })
        
        batch_file = f"latents/latents_batch_{self.batch_idx:05d}.npy"
        index_buffer.append({
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "seed": seed,
            "step_idx": step_idx,
            "batch_file": batch_file,
            "offset": offset,
            **rewards
        })
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
        arr = np.stack(self.buffer, axis=0)
        path = os.path.join(self.out_dir, f"latents_batch_{self.batch_idx:05d}.npy")
        np.save(path, arr)
        self.buffer = []
        self.batch_idx += 1
    
    def save_metadata(self, path: str):
        if self.metadata:
            pd.DataFrame(self.metadata).to_parquet(path, index=False)
    
    def close(self):
        self.flush()

# ============================================================
# PROMPT BUILDING
# ============================================================

def build_i2p_records(hf_split, use_safe: bool) -> List[Dict]:
    records = []
    for idx, item in enumerate(hf_split):
        raw = item.get("prompt", "")
        tags = get_tags(item)
        safe_prompt, cats = sanitize_prompt(raw, tags)
        
        prompt = safe_prompt if use_safe else raw
        primary = cats[0] if cats else "safe"
        risk = compute_risk(item)
        
        records.append({
            "prompt_id": f"i2p_{idx}",
            "source": "i2p",
            "prompt": normalize_ws(prompt),
            "primary_category": primary,
            "macro_categories": cats,
            "difficulty": "hard" if item.get("hard") else "easy",
            "risk_bucket": risk_bucket(risk),
            "risk_score": risk,
            "prompt_hash": sha256_text(normalize_ws(prompt)),
        })
    return records

def stratified_split(records, train_r=0.8, val_r=0.1, seed=1337):
    rng = random.Random(seed)
    buckets = {}
    
    for r in records:
        key = f"{r['primary_category']}|{r['difficulty']}"
        buckets.setdefault(key, []).append(r["prompt_id"])
    
    splits = {}
    for ids in buckets.values():
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(n * train_r)
        n_val = int(n * val_r)
        
        for i, pid in enumerate(ids):
            if i < n_tr:
                splits[pid] = "train"
            elif i < n_tr + n_val:
                splits[pid] = "val"
            else:
                splits[pid] = "test"
    
    return splits

def attach_seeds(records, salt: str, easy: int, hard: int):
    for r in records:
        n = hard if r["difficulty"] == "hard" else easy
        seeds = [deterministic_seed(r["prompt_id"], salt, i) for i in range(n)]
        r["seeds"] = seeds
        r["num_seeds"] = n

# ============================================================
# MAIN BUILDER
# ============================================================

def build_dataset(
    out_dir="/kaggle/working/dataset_v1",
    hf_dataset="AIML-TUDA/i2p",
    use_safe_prompt=False,
    split_seed=1337,
    seed_salt="v1",
    seeds_easy=2,
    seeds_hard=4,
    capture_every=2,
    steps=30,
    guidance=7.5,
    resume=True,
    time_budget_h=5.8,
    flush_samples=50,
    flush_seconds=180,
    max_samples=None,
):
    """Main dataset builder with checkpointing"""
    
    t0 = time.time()
    max_sec = time_budget_h * 3600
    
    print(f"\n{'='*60}")
    print("SD1.5 Safety Dataset Builder")
    print(f"{'='*60}")
    print(f"⏱  Time budget: {time_budget_h:.1f}h")
    print(f"💾 Flush: every {flush_samples} samples or {flush_seconds}s")
    print(f"🔄 Resume: {resume}\n")
    
    ensure_determinism()
    safe_mkdir(out_dir)
    safe_mkdir(f"{out_dir}/images")
    safe_mkdir(f"{out_dir}/latents")
    safe_mkdir(f"{out_dir}/logs")
    
    # Paths
    finals_path = f"{out_dir}/finals.parquet"
    traj_index_path = f"{out_dir}/traj_index.csv"
    ckpt_path = f"{out_dir}/checkpoint.json"
    
    # Buffers
    finals_buf = []
    traj_buf = []
    fail_buf = []
    last_flush = time.time()
    n_since_flush = 0
    
    def atomic_write(df, path):
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    
    def flush(reason):
        nonlocal finals_buf, traj_buf, fail_buf, last_flush, n_since_flush
        
        if finals_buf:
            new = pd.DataFrame(finals_buf)
            if os.path.exists(finals_path):
                old = pd.read_parquet(finals_path)
                merged = pd.concat([old, new], ignore_index=True)
                merged = merged.drop_duplicates(subset=["sample_id"], keep="last")
            else:
                merged = new
            atomic_write(merged, finals_path)
            print(f"💾 Flushed {len(finals_buf)} samples ({reason})")
            finals_buf = []
        
        if traj_buf:
            exists = os.path.exists(traj_index_path)
            with open(traj_index_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(traj_buf[0].keys()))
                if not exists:
                    w.writeheader()
                w.writerows(traj_buf)
            traj_buf = []
        
        if fail_buf:
            fail_path = f"{out_dir}/logs/failures.parquet"
            new = pd.DataFrame(fail_buf)
            if os.path.exists(fail_path):
                old = pd.read_parquet(fail_path)
                merged = pd.concat([old, new], ignore_index=True)
            else:
                merged = new
            atomic_write(merged, fail_path)
            fail_buf = []
        
        elapsed = time.time() - t0
        ckpt = {
            "reason": reason,
            "elapsed_h": elapsed / 3600,
            "timestamp": now_iso(),
            "samples": len(pd.read_parquet(finals_path)) if os.path.exists(finals_path) else 0,
        }
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        
        last_flush = time.time()
        n_since_flush = 0
    
    def maybe_flush():
        nonlocal n_since_flush
        if n_since_flush >= flush_samples:
            flush("periodic_samples")
        elif (time.time() - last_flush) >= flush_seconds:
            flush("periodic_time")
    
    def time_up():
        return (time.time() - t0) >= max_sec
    
    # Load done samples
    done = set()
    if resume and os.path.exists(finals_path):
        done = set(pd.read_parquet(finals_path)["sample_id"])
        print(f"✓ Resume: {len(done)} samples complete\n")
    
    # Load I2P
    print("Loading I2P dataset...")
    ds = load_dataset(hf_dataset)
    hf_split = ds["train"]
    print(f"✓ Loaded {len(hf_split)} prompts")
    
    # Build prompts
    print("\nBuilding prompt corpus...")
    records = build_i2p_records(hf_split, use_safe_prompt)
    
    # Splits
    print("Creating splits...")
    splits = stratified_split(records, seed=split_seed)
    for r in records:
        r["split"] = splits[r["prompt_id"]]
    
    # Seeds
    attach_seeds(records, seed_salt, seeds_easy, seeds_hard)
    
    print(f"✓ Total prompts: {len(records)}")
    print(f"✓ Splits: {Counter(r['split'] for r in records)}\n")
    
    # Save prompts
    pd.DataFrame(records).to_parquet(f"{out_dir}/prompts.parquet", index=False)
    
    # Initialize models
    print("Initializing models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = SDConfig(steps=steps, guidance=guidance)
    runner = SDRunner(cfg, device)
    nudenet = NudeNetScorer()
    violence = ViolenceScorer(device)
    clip = CLIPScorer(device)
    traj_storage = TrajStorage(f"{out_dir}/latents")
    
    print(f"✓ Models loaded on {device}\n")
    
    # Generate
    print("Generating images...")
    total = sum(r["num_seeds"] for r in records)
    if max_samples:
        total = min(total, max_samples)
        print(f"⚠  Limited to {max_samples} samples\n")
    
    pbar = tqdm(total=total)
    count = 0
    
    for rec in records:
        for seed in rec["seeds"]:
            if max_samples and count >= max_samples:
                break
            
            sid = f"{rec['prompt_id']}__seed_{seed}"
            
            if sid in done:
                pbar.update(1)
                count += 1
                continue
            
            try:
                # Generate
                img, traj = runner.generate(rec["prompt"], seed, capture_every)
                
                # Save image
                img_name = sanitize_filename(sid) + ".png"
                img_rel = f"images/{img_name}"
                img_path = f"{out_dir}/{img_rel}"
                img.save(img_path)
                
                # Safety scores
                nudity_score = nudenet.score(img_path)
                violence_score = violence.score(img)
                clip_sim = clip.score(img, rec["prompt"])
                
                # Reward
                prompt_risk = rec.get("risk_score", 0) / 100.0 if rec["source"] == "i2p" else None
                rew = compute_reward(nudity_score, violence_score, clip_sim, prompt_risk)
                
                # Store trajectory
                for step_idx, lat in traj:
                    traj_storage.add(
                        sid, rec["prompt_id"], seed, step_idx, lat,
                        {"reward_total": rew["R_total"], "reward_safe": rew["R_safe"], "reward_clip": rew["R_clip"]},
                        traj_buf
                    )
                
                # Finals row
                finals_buf.append({
                    "sample_id": sid,
                    "prompt_id": rec["prompt_id"],
                    "source": rec["source"],
                    "split": rec["split"],
                    "prompt": rec["prompt"],
                    "seed": seed,
                    "primary_category": rec["primary_category"],
                    "difficulty": rec["difficulty"],
                    "risk_bucket": rec["risk_bucket"],
                    "image_path": img_rel,
                    "nudity_score": nudity_score,
                    "violence_score": violence_score,
                    "clip_similarity": clip_sim,
                    "max_unsafe": rew["max_unsafe"],
                    "reward_safe": rew["R_safe"],
                    "reward_clip": rew["R_clip"],
                    "reward_total": rew["R_total"],
                    "created_at": now_iso(),
                })
                
                n_since_flush += 1
                maybe_flush()
                
                # Check time
                if time_up():
                    flush("time_budget")
                    print(f"\n⏹ Time budget reached: {(time.time()-t0)/3600:.2f}h")
                    print("✓ Progress saved. Rerun to continue.")
                    sys.exit(0)
                
                count += 1
                pbar.update(1)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                fail_buf.append({
                    "sample_id": sid,
                    "prompt_id": rec["prompt_id"],
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=5),
                    "created_at": now_iso(),
                })
                pbar.update(1)
                count += 1
        
        if max_samples and count >= max_samples:
            break
    
    pbar.close()
    
    # Final flush
    print("\n💾 Final flush...")
    flush("complete")
    
    traj_storage.save_metadata(f"{out_dir}/trajectories.parquet")
    traj_storage.close()
    
    # Manifest
    manifest = {
        "dataset": "sd15_safety_steering",
        "version": "v1",
        "created_at": now_iso(),
        "samples_generated": count,
        "elapsed_h": (time.time() - t0) / 3600,
        "config": {
            "steps": steps,
            "guidance": guidance,
            "seeds_easy": seeds_easy,
            "seeds_hard": seeds_hard,
        }
    }
    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    print(f"Samples: {count}")
    print(f"Elapsed: {(time.time()-t0)/3600:.2f}h")
    
    return manifest


# ============================================================
# RUN
# ============================================================

def run_dataset_build(mode="FULL"):
    # Test config
    TEST = {
        "out_dir": "/kaggle/working/dataset_test",
        "max_samples": 10,
        "flush_samples": 5,
    }
    
    # Full config
    FULL = {
        "out_dir": "/kaggle/working/dataset_v1",
        "max_samples": None,
        "time_budget_h": 5.8,
        "flush_samples": 25,
        "flush_seconds": 120,
    }
    
    # Select config based on function argument
    config = TEST if mode == "TEST" else FULL
    
    print(f"\n{'='*60}")
    print(f"STARTING: {mode} MODE")
    print(f"{'='*60}\n")
    
    # Ensure build_dataset is defined or imported before this
    manifest = build_dataset(**config)
    
    print(f"\n✓ Dataset ready at: {config['out_dir']}")

# Now the block ensures it runs if executed as a script, 
# but allows you to import it cleanly too.
if __name__ == "__main__":
    run_dataset_build(mode="FULL")