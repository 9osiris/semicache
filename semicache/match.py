"""matching: exact keys first, then embedding similarity over stored entries."""

from .embed import cosine
from .store import canonical_key, query_text


def filtered_params(body, ignore_keys):
    """request params that affect the answer. ignored keys never change the key."""
    return {k: v for k, v in body.items()
            if k not in ("messages", "model", "stream") and k not in ignore_keys}


def exact_lookup(store, model, messages, body, ignore_keys):
    """return (entry, key) on an exact match, else (None, key)."""
    params = filtered_params(body, ignore_keys)
    key = canonical_key(model, messages, params)
    return store.get_exact(key), key


def semantic_lookup(store, embedding, threshold):
    """best entry whose stored embedding clears the threshold, else None."""
    if embedding is None:
        return None, 0.0
    best = None
    best_score = 0.0
    for entry in store.entries.values():
        if not entry.embedding:
            continue
        score = cosine(embedding, entry.embedding)
        if score > best_score:
            best_score = score
            best = entry
    if best is not None and best_score >= threshold:
        return best, best_score
    return None, best_score


def build_reply(entry, model):
    """a cached reply re-wrapped as a fresh chat completion response."""
    reply = dict(entry.reply)
    reply["model"] = model or entry.model
    reply["usage"] = entry.usage
    return reply


def extract_reply_text(reply):
    """first choice message content, or empty string when there is none."""
    try:
        return reply["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""
