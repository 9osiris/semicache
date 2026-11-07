"""the cache store: entries on disk, exact keys, ttl, lru eviction."""

import hashlib
import json
import os
import time


def canonical_key(model, messages, params):
    """stable key for a request. json key order and spacing do not matter."""
    payload = {"model": model, "messages": messages, "params": params}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def query_text(messages, mode="last_user"):
    """text to embed for a request. last user turn, or all user turns joined."""
    users = [m.get("content", "") for m in messages
             if isinstance(m, dict) and m.get("role") == "user"]
    if not users:
        return ""
    if mode == "all_user":
        return "\n".join(str(u) for u in users)
    return str(users[-1])


class Entry:
    def __init__(self, key, model, messages, reply, usage, embedding=None,
                 created_at=None, last_hit=None, hits=0):
        self.key = key
        self.model = model
        self.messages = messages
        self.reply = reply
        self.usage = usage or {}
        self.embedding = embedding
        self.created_at = created_at if created_at is not None else time.time()
        self.last_hit = last_hit if last_hit is not None else self.created_at
        self.hits = hits

    def to_dict(self):
        return {
            "key": self.key,
            "model": self.model,
            "messages": self.messages,
            "reply": self.reply,
            "usage": self.usage,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "last_hit": self.last_hit,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            key=d["key"], model=d.get("model", ""), messages=d.get("messages", []),
            reply=d.get("reply", {}), usage=d.get("usage", {}),
            embedding=d.get("embedding"), created_at=d.get("created_at"),
            last_hit=d.get("last_hit"), hits=d.get("hits", 0))


class CacheStore:
    """json file of entries. load is tolerant, a corrupt file starts empty."""

    def __init__(self, path, ttl_seconds=86400, max_entries=1000):
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.entries = {}
        self.corrupt = False
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for d in raw.get("entries", []):
                e = Entry.from_dict(d)
                self.entries[e.key] = e
        except (OSError, ValueError, KeyError, TypeError):
            self.corrupt = True
            self.entries = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        data = {"version": 1, "entries": [e.to_dict() for e in self.entries.values()]}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def expired(self, entry, now=None):
        now = time.time() if now is None else now
        return self.ttl_seconds > 0 and (now - entry.created_at) > self.ttl_seconds

    def prune(self, now=None):
        now = time.time() if now is None else now
        dead = [k for k, e in self.entries.items() if self.expired(e, now)]
        for k in dead:
            del self.entries[k]
        return len(dead)

    def get_exact(self, key, now=None):
        now = time.time() if now is None else now
        entry = self.entries.get(key)
        if entry is None or self.expired(entry, now):
            if entry is not None:
                del self.entries[key]
            return None
        return entry

    def put(self, entry, now=None):
        now = time.time() if now is None else now
        self.entries[entry.key] = entry
        self.prune(now)
        # lru eviction: drop least recently hit until under the cap
        while len(self.entries) > self.max_entries:
            oldest = min(self.entries.values(), key=lambda e: e.last_hit)
            del self.entries[oldest.key]
        self.save()

    def hit(self, entry, now=None):
        now = time.time() if now is None else now
        entry.hits += 1
        entry.last_hit = now
        self.save()

    def clear(self):
        self.entries = {}
        self.save()

    def count(self):
        return len(self.entries)
