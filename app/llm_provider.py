"""C4a LLM provider layer: free-tier chain Groq -> Gemini over one OpenAI-compatible client,
with a round-robin load balancer across several API keys.

How a request is routed
- Slots = (provider, API key, model). They are grouped by (provider, model) in priority order:
  every Groq model in GROQ_MODEL order, then every Gemini model in GEMINI_MODEL order.
- Inside a group the keys take turns (round robin): request 1 starts on key 1, request 2 on key 2, ...
  so the load and the free-tier quota are spread evenly across keys.
- On failure the request moves on (failover): next key for the same model, then the next model, then the next
  provider. 429 / 503 with a short wait: move to another key if one is free, otherwise wait once.
  A long wait (e.g. a daily quota used up) puts that key + model in cooldown so it is skipped for a while.

Configuration (.env)
- GROQ_API_KEY / GEMINI_API_KEY: one key, or several comma-separated (GROQ_API_KEY_2 ... _9 also work).
  Keys must come from DIFFERENT accounts: free-tier limits are per account/organisation, so two keys from one
  account share one quota. Use your team's own accounts; creating accounts to evade limits may break provider terms.
- GROQ_MODEL / GEMINI_MODEL: one model, or several comma-separated, tried in order (each model has its own quota).
Keys are never logged; logs and results name a key by its position only (groq#2).
"""
from __future__ import annotations

import logging
import os
import threading
import time

from app import config

log = logging.getLogger(__name__)

PROVIDERS: dict[str, dict] = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY",
             "model_env": "GROQ_MODEL", "default": "openai/gpt-oss-120b,openai/gpt-oss-20b"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key": "GEMINI_API_KEY",
               "model_env": "GEMINI_MODEL", "default": "gemini-3.1-flash-lite,gemini-3.5-flash"},
}


class LLMUnavailable(Exception):
    """Every provider in LLM_CHAIN failed or has no key."""


def keys_for(provider: str) -> list[str]:
    """All API keys for a provider: comma-separated in <KEY>, plus <KEY>_2 ... <KEY>_9. Duplicates removed."""
    var = PROVIDERS[provider]["key"]
    raw = os.getenv(var, "").split(",") + [os.getenv(f"{var}_{i}", "") for i in range(2, 10)]
    out: list[str] = []
    for k in (k.strip() for k in raw):
        if k and k not in out:
            out.append(k)
    return out


def available() -> list[str]:
    return [p for p in config.LLM_CHAIN if p in PROVIDERS and keys_for(p)]


def models_for(provider: str, overrides: dict[str, str] | None = None) -> list[str]:
    p = PROVIDERS[provider]
    raw = (overrides or {}).get(provider) or os.getenv(p["model_env"], "").strip() or p["default"]
    return [m.strip() for m in raw.split(",") if m.strip()]


def model_for(provider: str, overrides: dict[str, str] | None = None) -> str:
    return models_for(provider, overrides)[0]


def pool_status() -> dict[str, dict]:
    """What /health shows: key count and model order per provider (never the keys themselves)."""
    now = time.time()
    with _lock:
        cooling = {k for k, until in _cooldown.items() if until > now}
    return {p: {"keys": len(keys_for(p)), "models": models_for(p),
                "cooling": sorted(f"{p}#{i + 1}/{m}" for (pp, i, m) in cooling if pp == p)}
            for p in available()}


# ---- balancer state (shared by API worker threads) ------------------------
_lock = threading.Lock()
_rr: dict[tuple[str, str], int] = {}                     # (provider, model) -> next key to start from
_cooldown: dict[tuple[str, int, str], float] = {}         # (provider, key index, model) -> skip until
_clients: dict[tuple[str, int], object] = {}


def _client(provider: str, idx: int):
    key = (provider, idx)
    if key not in _clients:
        from openai import OpenAI
        _clients[key] = OpenAI(api_key=keys_for(provider)[idx], base_url=PROVIDERS[provider]["base_url"],
                               timeout=config.LLM_TIMEOUT, max_retries=0)
    return _clients[key]


def _label(provider: str, idx: int, n_keys: int) -> str:
    return f"{provider}#{idx + 1}" if n_keys > 1 else provider


def plan(overrides: dict[str, str] | None = None) -> list[tuple[str, int, str]]:
    """The order of (provider, key index, model) slots for ONE request; advances the round robin."""
    slots: list[tuple[str, int, str]] = []
    now = time.time()
    with _lock:
        for provider in available():
            n = len(keys_for(provider))
            for model in models_for(provider, overrides):
                start = _rr.get((provider, model), 0) % n
                _rr[(provider, model)] = start + 1
                slots += [(provider, (start + j) % n, model) for j in range(n)]
        ready = [s for s in slots if _cooldown.get(s, 0) <= now]
    return ready or slots                                   # everything cooling: try anyway


def _cool(slot: tuple[str, int, str], seconds: float) -> None:
    with _lock:
        _cooldown[slot] = time.time() + min(seconds, 3600)


def chat_json(messages: list[dict], overrides: dict[str, str] | None = None,
              max_tokens: int = config.LLM_MAX_TOKENS) -> tuple[str, str, str]:
    """Return (content, provider label, model). Raises LLMUnavailable if every slot fails."""
    import openai

    if not available():
        raise LLMUnavailable("no API key set for any provider in LLM_CHAIN")
    slots = plan(overrides)
    errors = []
    for pos, slot in enumerate(slots):
        provider, idx, model = slot
        label = _label(provider, idx, len(keys_for(provider)))
        others_left = len(slots) - pos - 1
        kwargs = dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                      response_format={"type": "json_object"})
        if "gpt-oss" in model:          # reasoning model: keep hidden reasoning short (speed + token budget)
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        waited = False
        for _ in range(3):
            try:
                t = time.time()
                r = _client(provider, idx).chat.completions.create(**kwargs)
                content = r.choices[0].message.content or ""
                log.info("LLM %s/%s answered in %.1fs", label, model, time.time() - t)
                return content, label, model
            except openai.BadRequestError as e:
                if "response_format" in str(e) and "response_format" in kwargs:   # provider without JSON mode
                    kwargs.pop("response_format")
                    continue
                if "reasoning_effort" in str(e) and "extra_body" in kwargs:
                    kwargs.pop("extra_body")
                    continue
                errors.append(f"{label}/{model}: BadRequestError: {str(e)[:200]}")
                break
            except (openai.RateLimitError, openai.InternalServerError) as e:
                # "slow down" (429) or "busy" (503)
                delay = _retry_after(e)
                if delay > config.LLM_MAX_WAIT:           # e.g. daily quota used up: skip this key + model a while
                    _cool(slot, delay)
                elif others_left:                         # another key/model may be free right now: move on
                    _cool(slot, delay)
                elif not waited:                          # last option: wait once and retry
                    log.info("LLM %s %s, retrying in %.1fs", label, type(e).__name__, delay)
                    time.sleep(delay)
                    waited = True
                    continue
                errors.append(f"{label}/{model}: {type(e).__name__}: {str(e)[:200]}")
                break
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                # a bad or revoked key: bench this key for every model for an hour
                with _lock:
                    for m in models_for(provider, overrides):
                        _cooldown[(provider, idx, m)] = time.time() + 3600
                errors.append(f"{label}/{model}: {type(e).__name__}: {str(e)[:200]}")
                break
            except (openai.APITimeoutError, openai.APIConnectionError, openai.NotFoundError,
                    openai.APIStatusError) as e:
                errors.append(f"{label}/{model}: {type(e).__name__}: {str(e)[:200]}")
                break
        if errors and others_left:
            log.warning("LLM %s/%s failed (%s), trying the next slot", label, model,
                        errors[-1].split(": ", 1)[1][:120])
    raise LLMUnavailable("; ".join(errors))


def _retry_after(e) -> float:
    """Seconds the provider asks us to wait (Retry-After header, or 'try again in 1h2m4.2s' in the message)."""
    import re
    try:
        h = e.response.headers.get("retry-after")
        if h:
            return float(h)
    except Exception:
        pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(e))
    if m:
        return 3600 * float(m.group(1) or 0) + 60 * float(m.group(2) or 0) + float(m.group(3))
    return 2.0
