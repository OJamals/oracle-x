#!/usr/bin/env python3
"""Enumerate embedding-capable models from an OpenAI-compatible endpoint.

Usage:
  python list_embeddings.py [--base URL]

If --base is omitted, OPENAI_API_BASE env var is used. The script will try sensible
variants (adding /v1) and query the /models endpoint. It filters models whose id or
capabilities indicate embedding support.
"""
from __future__ import annotations
import os, sys, json, ssl, urllib.request, urllib.error, argparse
from typing import List, Dict, Any
import difflib

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def fetch_models(base: str, insecure: bool = False) -> Dict[str, Any]:
    # Normalize base variants to try
    candidates = []
    b = base.rstrip("/")
    # If caller provided a full models URL, try it verbatim first.
    if b.endswith("/models") or b.endswith("/v1/models"):
        candidates.append(b)
    else:
        # Try both common variants
        candidates.append(b + "/v1/models")
        candidates.append(b + "/models")  # some servers already include v1 internally

    headers = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ctx = None
    if insecure:
        ctx = ssl._create_unverified_context()

    last_err = None
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
                raw = r.read().decode("utf-8", "replace")
            return {"url": url, "data": json.loads(raw)}
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            ssl.SSLError,
            json.JSONDecodeError,
        ) as e:
            last_err = e
    raise RuntimeError(f"Failed to fetch models from {base}: {last_err}")


def extract_embedding_models(models: List[Dict[str, Any]]) -> List[str]:
    out = []
    for m in models:
        mid = m.get("id") or m.get("name") or ""
        caps = m.get("capabilities") or {}
        if not isinstance(caps, dict):
            caps = {}
        if (
            "embed" in mid.lower()
            or "embedding" in mid.lower()
            or caps.get("embeddings") is True
            or caps.get("embedding") is True
        ):
            out.append(mid)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", help="Override base endpoint (default: OPENAI_API_BASE env)"
    )
    parser.add_argument(
        "--insecure", action="store_true", help="Skip TLS verification (dev only)"
    )
    args = parser.parse_args()

    base = args.base or os.environ.get("OPENAI_API_BASE") or "http://localhost:8000"
    print(f"Using base endpoint: {base}")

    try:
        resp = fetch_models(base, insecure=args.insecure)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Hints:")
        print("  - Ensure the endpoint is running and reachable.")
        print(
            "  - If using a local inference server, set OPENAI_API_BASE to its root (e.g. http://localhost:8000)."
        )
        print("  - Provide a valid OPENAI_API_KEY if the server requires auth.")
        sys.exit(1)

    models = resp["data"].get("data") if isinstance(resp["data"], dict) else None
    if not models:
        print(
            f"No model list returned from {resp['url']}. Raw payload keys: {list(resp['data'].keys()) if isinstance(resp['data'], dict) else type(resp['data'])}"
        )
        sys.exit(2)

    embed_models = extract_embedding_models(models)
    # Build non-embedding list
    embed_set = set(embed_models)
    non_embed = []
    for m in models:
        mid = m.get("id") if isinstance(m, dict) else str(m)
        if mid not in embed_set:
            desc = m.get("description") if isinstance(m, dict) else ""
            non_embed.append((mid, desc))

    print(f"Queried: {resp['url']}")
    print(f"Total models: {len(models)}")
    if embed_models:
        print("Embedding-capable models:")
        for mid in embed_models:
            print("  -", mid)
    else:
        print("No embedding-capable models detected by heuristics.")

    print("\nNon-embedding models:")
    for mid, desc in non_embed:
        if desc:
            print("  -", mid, "-", desc)
        else:
            print("  -", mid)

    # Optionally dump JSON for further inspection
    if os.environ.get("DUMP_MODELS_JSON") == "1":
        print("\nFull raw model list JSON:")
        print(json.dumps(resp["data"], indent=2))

    # Fuzzy-matching helper: detect variants like 'gpt-5-mini'
    def fuzzy_search(
        query: str, models_list: List[Dict[str, Any]], n: int = 5
    ) -> List[str]:
        ids = [m.get("id") or "" for m in models_list if isinstance(m, dict)]
        # Exact substring matches first
        subs = [mid for mid in ids if query.lower() in mid.lower()]
        if subs:
            return subs
        # Fallback to difflib close matches
        close = difflib.get_close_matches(query, ids, n=n, cutoff=0.6)
        return close

    # Probe model metadata endpoint if provider supports it
    def fetch_model_metadata(
        base_models_url: str, model_id: str, insecure: bool = False
    ) -> Dict[str, Any] | None:
        # base_models_url is expected to be the full /models URL used above
        # try several candidate metadata URLs
        candidates = []
        b = base_models_url.rstrip("/")
        if b.endswith("/models"):
            candidates.append(f"{b}/{model_id}")
            # also try removing /models and adding /v1/models/{id}
            root = b[: -len("/models")]
            candidates.append(f"{root}/v1/models/{model_id}")
            candidates.append(f"{root}/models/{model_id}")
        else:
            candidates.append(f"{b}/models/{model_id}")

        ctx = None
        if insecure:
            ctx = ssl._create_unverified_context()

        headers = {}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for url in candidates:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
                    raw = r.read().decode("utf-8", "replace")
                return {"url": url, "data": json.loads(raw)}
            except Exception:
                continue
        return None

    # Demonstrate fuzzy search for gpt-5-mini
    query = "gpt-5-mini"
    fuzzy = fuzzy_search(query, models)
    if fuzzy:
        print("\nFuzzy matches for", query, ":")
        for m in fuzzy:
            print("  -", m)
            meta = fetch_model_metadata(
                resp["url"],
                m,
                insecure=(
                    "--insecure" in sys.argv or os.environ.get("INSECURE") == "1"
                ),
            )
            if meta:
                print("    metadata URL:", meta["url"])
                # print a compact summary
                d = meta["data"]
                if isinstance(d, dict):
                    keys = list(d.keys())
                    print("    metadata keys:", keys)
            else:
                print("    no metadata endpoint found for", m)
    else:
        print("\nNo fuzzy matches found for", query)


if __name__ == "__main__":
    main()
