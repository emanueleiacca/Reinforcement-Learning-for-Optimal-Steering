import json
import ast
import math
from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================
ds = load_dataset("AIML-TUDA/i2p")
INPUT_HF_DS = ds["train"]
OUTPUT_PATH = "/kaggle/working/i2p_train_processed.jsonl"

SCHEMA_VERSION = 1  # se cambi formato in futuro, incrementa


# ============================================================
# 1. CATEGORY MAP
# ============================================================

_BASE_CATEGORY_MAP = {
    # Nudity Group
    "nudity": "nudity",
    "sexual activity": "nudity",
    "sexual content": "nudity",
    "nsfw": "nudity",
    "sexual": "nudity",
    "adult": "nudity",

    # Violence Group
    "violence": "violence",
    "fighting": "violence",
    "illegal activity": "violence",

    # Gore/Shock Group
    "graphic violence": "gore",
    "gore": "gore",
    "blood": "gore",
    "shock": "gore",
    "shocking": "gore",

    # Self Harm Group
    "self harm": "self_harm",
    "suicide": "self_harm",
    "drug use": "self_harm",
    "drugs": "self_harm",

    # Harassment Group
    "hate speech": "harassment",
    "harassment": "harassment",
    "bullying": "harassment",
    "hate": "harassment",

    # Explicit safe
    "safe": "safe",
}


def _normalize_tag(tag: str) -> str:
    t = str(tag).strip().lower()
    t = t.replace("-", " ")
    t = t.replace("_", " ")
    t = " ".join(t.split())
    return t


CATEGORY_MAP = {_normalize_tag(k): v for k, v in _BASE_CATEGORY_MAP.items()}


# ============================================================
# 2. SAFE REWRITE RULES
# ============================================================

SAFE_REWRITE = {
    "nudity": (
        ", fully clothed subjects, no nudity, no explicit body exposure"
    ),
    "violence": (
        ", without violence, without aggressive or harmful actions"
    ),
    "gore": (
        ", without gore, without graphic injury, without distressing elements"
    ),
    "self_harm": (
        ", without self-harm implications, safe and neutral composition"
    ),
    "harassment": (
        ", polite tone, neutral and respectful atmosphere"
    ),
}


# ============================================================
# 3. HELPERS
# ============================================================

def get_tags_robust(item) -> list:
    raw = item.get("categories", [])
    if not raw:
        return []

    if isinstance(raw, list):
        return [str(x) for x in raw]

    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return []

        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (SyntaxError, ValueError):
                pass

        if "," in cleaned:
            return [x.strip() for x in cleaned.split(",") if x.strip()]

        return [cleaned]

    return [str(raw)]


def sanitize_prompt(prompt: str, raw_tags: list):
    if not isinstance(prompt, str):
        prompt = ""
    base = prompt.strip()

    macros_raw = set()
    for t in raw_tags:
        norm = _normalize_tag(t)
        cat = CATEGORY_MAP.get(norm, norm)
        macros_raw.add(cat)

    macros_raw = sorted(macros_raw)

    if len(macros_raw) == 1 and macros_raw[0] == "safe":
        return base, macros_raw

    macros = [m for m in macros_raw if m != "safe"] if len(macros_raw) > 1 else macros_raw

    suffixes = set()
    for cat in macros:
        rewrite = SAFE_REWRITE.get(cat)
        if rewrite:
            suffixes.add(rewrite)

    if not suffixes and raw_tags:
        suffixes.add(
            ", safe neutral composition, without harmful or explicit content"
        )

    if suffixes:
        full_safe_prompt = base + "".join(sorted(suffixes))
        return full_safe_prompt, sorted(macros)

    return base, sorted(macros)


def _get_pct(entry, key: str) -> float:
    v = entry.get(key, 0)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0

    if v < 0:
        v = 0.0
    if v <= 1.0:
        v = v * 100.0
    if v > 100.0:
        v = 100.0
    return v


def compute_prompt_risk(entry) -> float:
    w_inap = 0.40
    w_q16 = 0.30
    w_sd = 0.20
    w_nudity = 0.05
    w_hard = 0.02

    inappropriate = _get_pct(entry, "inappropriate_percentage")
    q16 = _get_pct(entry, "q16_percentage")

    # TODO: se sd_safety_percentage è "quanto è SAFE", inverti qui:
    # sd_safety = 100.0 - _get_pct(entry, "sd_safety_percentage")
    sd_safety = _get_pct(entry, "sd_safety_percentage")

    nudity = _get_pct(entry, "nudity_percentage")

    hard_flag = entry.get("hard", 0)
    try:
        hard_flag = int(hard_flag)
    except (TypeError, ValueError):
        hard_flag = 0
    hard_flag = 1 if hard_flag else 0

    risk = (
        w_inap * inappropriate +
        w_q16 * q16 +
        w_sd * sd_safety +
        w_nudity * nudity +
        w_hard * (hard_flag * 100.0)
    )

    return float(max(0.0, min(risk, 100.0)))


def risk_to_bucket(risk: float) -> str:
    if risk < 25.0:
        return "low"
    if risk < 50.0:
        return "medium"
    if risk < 75.0:
        return "high"
    return "extreme"


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

print(f"Streaming processed entries to {OUTPUT_PATH}...")

num_items = len(INPUT_HF_DS)
print(f"Total HF train items: {num_items}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, item in enumerate(INPUT_HF_DS):

        raw_prompt = item.get("prompt", "")
        raw_tags = get_tags_robust(item)

        safe_prompt, macros = sanitize_prompt(raw_prompt, raw_tags)
        primary_category = macros[0] if macros else "safe"

        prompt_risk = compute_prompt_risk(item)
        prompt_risk_norm = prompt_risk / 100.0
        prompt_risk_bucket = risk_to_bucket(prompt_risk)

        was_sanitized = (safe_prompt != raw_prompt)

        entry = {
            "schema_version": SCHEMA_VERSION,

            "hf_idx": i,

            "prompt_raw": raw_prompt,
            "prompt_safe": safe_prompt,
            "was_sanitized": was_sanitized,

            "risk_tags": raw_tags,
            "macro_categories": macros,
            "primary_category": primary_category,

            "prompt_risk_score": prompt_risk,
            "prompt_risk_score_norm": prompt_risk_norm,
            "prompt_risk_bucket": prompt_risk_bucket,

            # alias per compatibilità con codice che si aspetta 'risk_score'
            "risk_score": prompt_risk,

            "inappropriate_percentage": item.get("inappropriate_percentage", 0),
            "q16_percentage": item.get("q16_percentage", 0),
            "sd_safety_percentage": item.get("sd_safety_percentage", 0),
            "nudity_percentage": item.get("nudity_percentage", 0),
            "hard_flag": int(bool(item.get("hard", 0))),

            "sd_seed": item.get("sd_seed"),
            "lexica_url": item.get("lexica_url"),
        }

        f.write(json.dumps(entry) + "\n")

print("Processing Complete. File saved:", OUTPUT_PATH)
