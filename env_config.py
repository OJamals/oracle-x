from __future__ import annotations
import os
from typing import List, Optional
try:  # optional dotenv support
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

"""Centralized environment configuration for model usage.

Standard (preferred) environment variables:
  - OPENAI_API_KEY       : API key for both completion + embedding endpoints
  - OPENAI_API_BASE      : Base URL for completion/chat models
  - OPENAI_MODEL         : Primary (non-embedding) model name
  - EMBEDDING_MODEL      : Embedding model name
  - EMBEDDING_API_BASE   : Base URL for embedding endpoint (falls back to OPENAI_API_BASE if unset)

Legacy variables (NON_EMBEDDING_MODEL, OPENAI_NON_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL, QWEN3_*)
are deprecated and no longer read. Set the new variables instead.
"""

# Primary (non‑embedding) model used for chat/completions (user preferred)
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')

# Embedding model
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')

# Base URLs
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
EMBEDDING_API_BASE = os.environ.get('EMBEDDING_API_BASE') or OPENAI_API_BASE

# Fallback models (comma separated env or sensible defaults). These are attempted
# if the preferred model is unsupported by the provider.
_DEFAULT_FALLBACK_MODELS: List[str] = [
    'gpt-4o-mini',
    'gpt-4o',
    'o4-mini',
    'o3-mini'
]
FALLBACK_MODELS: List[str] = [m.strip() for m in os.environ.get('FALLBACK_MODELS', '').split(',') if m.strip()] or _DEFAULT_FALLBACK_MODELS

# --- Getter helpers (retain old function names for backwards compatibility) ---
def get_openai_model() -> str:
    return OPENAI_MODEL

def get_non_embedding_model() -> str:  # backward compatibility
    return OPENAI_MODEL

def get_embedding_model() -> str:
    return EMBEDDING_MODEL

def get_openai_api_base() -> str | None:
    return OPENAI_API_BASE

def get_embedding_api_base() -> str | None:
    return EMBEDDING_API_BASE

def get_fallback_models() -> List[str]:
    return FALLBACK_MODELS

def resolve_model(client, preferred: Optional[str] = None, test: bool = True) -> str:
    """Attempt to resolve a supported chat/completions model.

    Tries the preferred model first, then falls back over FALLBACK_MODELS.
    We perform a very small test completion to detect "model_not_supported" style errors.
    If all candidates fail we return the original preferred to preserve current behavior.
    """
    preferred = preferred or OPENAI_MODEL
    tried = []
    candidates = [preferred] + [m for m in FALLBACK_MODELS if m != preferred]
    for model in candidates:
        if not test:
            return model  # skip probing (useful for offline tests)
        try:
            # Tiny inexpensive probe (max tokens kept minimal)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "ping"}, {"role": "user", "content": "Return OK"}],
                max_completion_tokens=5
            )
            if model != preferred:
                print(f"[INFO] Preferred model '{preferred}' unsupported. Falling back to '{model}'.")
            else:
                print(f"[DEBUG] Preferred model '{model}' validated.")
            return model
        except Exception as e:
            tried.append((model, str(e)))
            # Only log concise message to avoid clutter
            print(f"[DEBUG] Model probe failed for '{model}': {e}")
            continue
    print("[WARN] All model probes failed. Using preferred model despite probe failures.")
    return preferred

