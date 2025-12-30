# build_p_from_pca.py
# ============================================================
# Costruzione matrice P per azioni in R^32 tramite PCA su
# differenze di latenti (z_safe - z_raw) usando I2P preprocessato.
# Versione con DEBUG esteso e salvataggi persistenti per Kaggle.
# ============================================================

import json
import os
from typing import List, Dict, Any

import torch
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm

# ------------------ CONFIG ------------------

# File JSONL preprocessato (già creato prima)
I2P_JSONL_PATH = "/kaggle/working/i2p_train_processed.jsonl"

# Directory in cui salvare TUTTO
OUTPUT_DIR = "/kaggle/working"

# Dove salvare la matrice P + meta
OUTPUT_P_PATH = os.path.join(OUTPUT_DIR, "P_pca_32.pt")

# File con le coppie usate (debug / analisi)
USED_PAIRS_JSONL = os.path.join(OUTPUT_DIR, "pca_used_pairs.jsonl")

# Modello SD
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Numero massimo di coppie (raw/safe) da usare per la PCA
MAX_PAIRS = 1500   # puoi alzare/abbassare a seconda del budget

# Passi di denoising per ottenere il latente finale
NUM_INFERENCE_STEPS = 30

# Dimensioni latente SD 1.5
LATENT_C = 4
LATENT_H = 64
LATENT_W = 64
LATENT_DIM = LATENT_C * LATENT_H * LATENT_W

# Dimensionalità dello spazio azioni
ACTION_DIM = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 80)
print("[CONFIG] build_p_from_pca.py")
print(f"  I2P_JSONL_PATH    : {I2P_JSONL_PATH}")
print(f"  OUTPUT_DIR        : {OUTPUT_DIR}")
print(f"  OUTPUT_P_PATH     : {OUTPUT_P_PATH}")
print(f"  USED_PAIRS_JSONL  : {USED_PAIRS_JSONL}")
print(f"  MODEL_ID          : {MODEL_ID}")
print(f"  MAX_PAIRS         : {MAX_PAIRS}")
print(f"  NUM_INFER_STEPS   : {NUM_INFERENCE_STEPS}")
print(f"  LATENT_DIM        : {LATENT_DIM} (4x64x64)")
print(f"  ACTION_DIM        : {ACTION_DIM}")
print(f"  DEVICE            : {device}")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ SD PIPELINE ------------------

print(f"[INFO] Loading Stable Diffusion pipeline: {MODEL_ID}")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)
print("[INFO] Pipeline ready.")


def safe_truncate_prompt(prompt: str) -> str:
    """Tronca il prompt ai token massimi del modello, prevenendo errori tokenizer."""
    if not isinstance(prompt, str):
        prompt = ""
    base = prompt.strip()
    tokenizer = pipe.tokenizer
    max_length = tokenizer.model_max_length

    tokens = tokenizer(
        base,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids

    cleaned = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return cleaned


# ------------------ LETTURA JSONL ------------------

def load_i2p_entries(jsonl_path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries.append(entry)
    print(f"[INFO] Loaded {len(entries)} entries from {jsonl_path}")
    return entries


# ------------------ GENERAZIONE LATENTI ------------------

@torch.no_grad()
def get_final_latent_from_prompt(prompt: str, sd_seed: int) -> torch.Tensor:
    """
    Genera il latente finale per un dato prompt e seed.
    - Usa output_type='latent' per ottenere z invece dell'immagine.
    - Usa noise iniziale fissato (seed) per coerenza raw/safe.
    """
    prompt_clean = safe_truncate_prompt(prompt)

    g = torch.Generator(device=device).manual_seed(sd_seed)
    init_latents = torch.randn(
        (1, LATENT_C, LATENT_H, LATENT_W),
        generator=g,
        device=device,
        dtype=torch.float16,
    )

    out = pipe(
        prompt_clean,
        num_inference_steps=NUM_INFERENCE_STEPS,
        latents=init_latents,
        output_type="latent",
    )

    latents = getattr(out, "images", None)
    if latents is None:
        latents = getattr(out, "latents", None)
    if latents is None:
        raise RuntimeError("Pipeline output does not contain 'images' or 'latents'")

    z = latents[0].detach().float().cpu()
    return z  # [4,64,64]


# ------------------ COSTRUZIONE MATRIX DELTE ------------------

def build_delta_matrix(
    entries: List[Dict[str, Any]],
    max_pairs: int,
) -> torch.Tensor:
    """
    Costruisce una matrice D [N, LATENT_DIM] con
      D[i] = flatten(z_safe - z_raw)
    Usa solo entry con was_sanitized=True e sd_seed valido.
    Inoltre, salva un JSONL con le coppie effettivamente usate.
    """
    deltas = []
    used = 0

    # puliamo il file JSONL di coppie usate
    if os.path.exists(USED_PAIRS_JSONL):
        os.remove(USED_PAIRS_JSONL)
        print(f"[INFO] Removed existing {USED_PAIRS_JSONL}")

    print(f"[INFO] Building delta matrix with max_pairs={max_pairs}...")
    with open(USED_PAIRS_JSONL, "a", encoding="utf-8") as pairs_f:

        for entry in tqdm(entries, desc="Collecting deltas"):
            if used >= max_pairs:
                break

            was_sanitized = entry.get("was_sanitized", False)
            if not was_sanitized:
                continue

            prompt_raw = entry.get("prompt_raw", "")
            prompt_safe = entry.get("prompt_safe", "")
            sd_seed = entry.get("sd_seed", None)

            if sd_seed is None:
                continue

            try:
                sd_seed = int(sd_seed)
            except (TypeError, ValueError):
                continue

            if not isinstance(prompt_raw, str) or not isinstance(prompt_safe, str):
                continue
            if prompt_raw.strip() == "" or prompt_safe.strip() == "":
                continue

            hf_idx = entry.get("hf_idx", None)

            # Debug sui primi esempi
            if used < 3:
                print("-" * 60)
                print(f"[DEBUG] Candidate pair #{used+1}")
                print(f"  hf_idx       : {hf_idx}")
                print(f"  seed         : {sd_seed}")
                print(f"  prompt_raw   : {prompt_raw[:120]}{'...' if len(prompt_raw) > 120 else ''}")
                print(f"  prompt_safe  : {prompt_safe[:120]}{'...' if len(prompt_safe) > 120 else ''}")

            try:
                z_raw = get_final_latent_from_prompt(prompt_raw, sd_seed)
                z_safe = get_final_latent_from_prompt(prompt_safe, sd_seed)
            except Exception as e:
                print(f"[WARN] Skipping hf_idx={hf_idx} due to error: {e}")
                continue

            delta = (z_safe - z_raw).view(-1)  # [LATENT_DIM]
            deltas.append(delta)
            used += 1

            # Scriviamo anche su JSONL la coppia usata
            meta_line = {
                "hf_idx": hf_idx,
                "sd_seed": sd_seed,
                "prompt_raw": prompt_raw,
                "prompt_safe": prompt_safe,
            }
            pairs_f.write(json.dumps(meta_line) + "\n")

            if used % 20 == 0:
                print(f"[INFO] Collected {used} delta vectors so far...")

    if not deltas:
        raise RuntimeError("No valid delta vectors collected. Check dataset/filters.")

    D = torch.stack(deltas, dim=0)  # [N, LATENT_DIM]
    print(f"[INFO] Delta matrix shape: {D.shape}")
    print(f"[INFO] Actually used {D.shape[0]} pairs (out of max {max_pairs}).")
    return D


# ------------------ PCA via SVD ------------------

def compute_pca_projection(D: torch.Tensor, action_dim: int) -> torch.Tensor:
    """
    PCA su D [N, LATENT_DIM].
    Ritorna P [LATENT_DIM, action_dim] con colonne unit-norm (componenti principali).
    """
    print("[INFO] Computing PCA via SVD...")

    # Centra i dati
    mean = D.mean(dim=0, keepdim=True)
    D_centered = D - mean

    # Portiamo su CPU in float32 per SVD
    D_centered = D_centered.to(torch.float32).cpu()
    print(f"[INFO] Running torch.linalg.svd on matrix of shape {D_centered.shape} ...")
    U, S, Vh = torch.linalg.svd(D_centered, full_matrices=False)
    print("[INFO] SVD done.")
    print(f"       U shape : {U.shape}")
    print(f"       S shape : {S.shape}")
    print(f"       Vh shape: {Vh.shape}")

    # Debug: valori singolari principali
    top_k = min(10, S.shape[0])
    top_s = S[:top_k].tolist()
    print(f"[DEBUG] Top-{top_k} singular values: {top_s}")

    num_components = min(action_dim, Vh.shape[0])
    PCs = Vh[:num_components]  # [K, LATENT_DIM]

    P = PCs.T.contiguous()  # [LATENT_DIM, K]
    print(f"[INFO] Raw PCA P shape: {P.shape}")

    # Normalizza colonne (unit norm) per stabilità
    col_norms = P.norm(dim=0, keepdim=True) + 1e-8
    P = P / col_norms
    print("[INFO] Column-normalized P.")

    return P, mean.squeeze(0)


# ------------------ MAIN ------------------

def main():
    entries = load_i2p_entries(I2P_JSONL_PATH)

    # Costruisci matrice D delle delta
    D = build_delta_matrix(entries, max_pairs=MAX_PAIRS)

    # PCA → P
    P, delta_mean = compute_pca_projection(D, action_dim=ACTION_DIM)

    # Statistiche di debug sulle delta
    with torch.no_grad():
        norms = D.norm(dim=1)  # [N]
        print(f"[STATS] delta norm: mean={norms.mean().item():.4f}, "
              f"std={norms.std().item():.4f}, "
              f"min={norms.min().item():.4f}, "
              f"max={norms.max().item():.4f}")

    # Carichiamo anche l'elenco hf_idx usati dal JSONL (già scritto)
    used_hf_idx = []
    used_meta = []
    if os.path.exists(USED_PAIRS_JSONL):
        with open(USED_PAIRS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                used_hf_idx.append(m.get("hf_idx"))
                used_meta.append(m)

    # Salviamo TUTTO in un unico checkpoint .pt
    ckpt = {
        "P": P,                         # [LATENT_DIM, ACTION_DIM]
        "delta_mean": delta_mean,       # [LATENT_DIM]
        "used_hf_idx": used_hf_idx,     # lista di hf_idx usati
        "used_meta": used_meta,         # lista con prompt_raw/safe/seed
        "config": {
            "model_id": MODEL_ID,
            "latent_shape": [LATENT_C, LATENT_H, LATENT_W],
            "num_pairs": int(D.shape[0]),
            "max_pairs": int(MAX_PAIRS),
            "num_inference_steps": int(NUM_INFERENCE_STEPS),
            "action_dim": int(ACTION_DIM),
            "latent_dim": int(LATENT_DIM),
            "i2p_jsonl_path": I2P_JSONL_PATH,
        },
    }

    torch.save(ckpt, OUTPUT_P_PATH)
    print("=" * 80)
    print(f"[SUCCESS] Saved PCA projection matrix P to: {OUTPUT_P_PATH}")
    print(f"[SUCCESS] Used pairs JSONL saved to: {USED_PAIRS_JSONL}")
    print("=" * 80)


if __name__ == "__main__":
    main()