"""embeddings client and cosine similarity. failures return None, never raise."""

import json
import math
import urllib.error
import urllib.request


def cosine(a, b):
    """cosine similarity of two vectors. 0.0 when either is empty or zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Embedder:
    def __init__(self, base_url, model="", api_key="", timeout=10):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def enabled(self):
        return bool(self.base_url)

    def embed(self, text):
        """one embedding vector, or None when the endpoint is down or bad."""
        if not self.enabled or not text:
            return None
        body = {"input": text}
        if self.model:
            body["model"] = self.model
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
            vec = data["data"][0]["embedding"]
            return [float(x) for x in vec]
        except (OSError, ValueError, KeyError, IndexError, TypeError,
                urllib.error.HTTPError, urllib.error.URLError):
            return None
