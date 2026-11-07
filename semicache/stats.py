"""counters persisted next to the cache file. survives restarts."""

import json
import os


FIELDS = ("requests", "exact_hits", "semantic_hits", "misses",
          "tokens_saved", "cost_saved_usd", "unknown_cost_hits")


class Stats:
    def __init__(self, path):
        self.path = path
        self.data = {k: 0 for k in FIELDS}
        self.data["cost_saved_usd"] = 0.0
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k in FIELDS:
                if k in raw and isinstance(raw[k], (int, float)):
                    self.data[k] = raw[k]
        except (OSError, ValueError):
            pass

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def record(self, kind, prompt_tokens=0, completion_tokens=0, cost=0.0,
               unknown_cost=False):
        self.data["requests"] += 1
        if kind in ("exact", "semantic"):
            self.data["exact_hits" if kind == "exact" else "semantic_hits"] += 1
            self.data["tokens_saved"] += prompt_tokens + completion_tokens
            self.data["cost_saved_usd"] += cost
            if unknown_cost:
                self.data["unknown_cost_hits"] += 1
        else:
            self.data["misses"] += 1
        self.save()

    def hit_rate(self):
        total = self.data["requests"]
        if not total:
            return 0.0
        return (self.data["exact_hits"] + self.data["semantic_hits"]) / total

    def to_dict(self):
        out = dict(self.data)
        out["hit_rate"] = round(self.hit_rate(), 4)
        return out
