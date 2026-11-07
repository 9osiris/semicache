"""the proxy server: cache first, upstream second, stats always."""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .embed import Embedder
from .match import (build_reply, exact_lookup, extract_reply_text,
                    filtered_params, semantic_lookup)
from .pricing import cost_usd
from .stats import Stats
from .store import CacheStore, Entry, query_text


class CacheServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.store = CacheStore(cfg["cache_file"], cfg["ttl_seconds"],
                                cfg["max_entries"])
        stats_path = os.path.join(os.path.dirname(cfg["cache_file"]) or ".",
                                  "stats.json")
        self.stats = Stats(stats_path)
        self.embedder = Embedder(cfg["embeddings_base_url"],
                                 cfg["embeddings_model"],
                                 os.environ.get("OPENAI_API_KEY", ""))
        self.upstream = cfg["upstream_base_url"].rstrip("/")

    def forward(self, body, headers):
        """send the request upstream, return (status, parsed json or None)."""
        req = urllib.request.Request(
            self.upstream + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        auth = headers.get("Authorization")
        if auth:
            req.add_header("Authorization", auth)
        elif os.environ.get("OPENAI_API_KEY"):
            req.add_header("Authorization",
                           "Bearer " + os.environ["OPENAI_API_KEY"])
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.load(e)
            except ValueError:
                return e.code, {"error": {"message": e.read().decode("utf-8", "replace")}}
        except (OSError, ValueError) as e:
            return 502, {"error": {"message": "upstream unreachable: %s" % e}}

    def forward_raw(self, body, headers):
        """forward a request and return raw bytes. used for streaming replies."""
        req = urllib.request.Request(
            self.upstream + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        auth = headers.get("Authorization")
        if auth:
            req.add_header("Authorization", auth)
        elif os.environ.get("OPENAI_API_KEY"):
            req.add_header("Authorization",
                           "Bearer " + os.environ["OPENAI_API_KEY"])
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.headers.get_content_type(), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, "application/json", e.read()
        except OSError as e:
            return 502, "application/json", json.dumps(
                {"error": {"message": "upstream unreachable: %s" % e}}).encode("utf-8")

    def handle_chat(self, body, headers):
        """one chat completion. returns (status, reply, cache status, similarity).

        streaming requests come back as (None, None, "stream-passthrough",
        None): the http layer relays the raw sse bytes, nothing is cached.
        """
        model = body.get("model", "")
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return 400, {"error": {"message": "messages must be a non-empty list"}}, "error", None
        if body.get("stream"):
            return None, None, "stream-passthrough", None
        entry, key = exact_lookup(self.store, model, messages, body,
                                  self.cfg["ignore_keys"])
        if entry is not None:
            self.store.hit(entry)
            usage = entry.usage
            cost, unknown = cost_usd(entry.model,
                                     usage.get("prompt_tokens", 0),
                                     usage.get("completion_tokens", 0))
            self.stats.record("exact", usage.get("prompt_tokens", 0),
                              usage.get("completion_tokens", 0), cost, unknown)
            return 200, build_reply(entry, model), "hit-exact", None
        embedding = None
        if self.embedder.enabled:
            text = query_text(messages, self.cfg["embed_text"])
            embedding = self.embedder.embed(text)
        if embedding is not None:
            match, score = semantic_lookup(self.store, embedding,
                                           self.cfg["similarity_threshold"])
            if match is not None:
                self.store.hit(match)
                usage = match.usage
                cost, unknown = cost_usd(match.model,
                                         usage.get("prompt_tokens", 0),
                                         usage.get("completion_tokens", 0))
                self.stats.record("semantic", usage.get("prompt_tokens", 0),
                                  usage.get("completion_tokens", 0), cost, unknown)
                return 200, build_reply(match, model), "hit-semantic", round(score, 4)
        status, reply = self.forward(body, headers)
        if status == 200 and extract_reply_text(reply):
            params = filtered_params(body, self.cfg["ignore_keys"])
            new_entry = Entry(key, model, messages, reply,
                              reply.get("usage", {}), embedding=embedding)
            self.store.put(new_entry)
        self.stats.record("miss")
        return status, reply, "miss", None

    def serve(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send_json(self, status, payload, cache_status=None,
                           similarity=None):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if cache_status:
                    self.send_header("X-Semicache", cache_status)
                if similarity is not None:
                    self.send_header("X-Semicache-Similarity", str(similarity))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/semicache/stats":
                    out = server.stats.to_dict()
                    out["entries"] = server.store.count()
                    self._send_json(200, out)
                    return
                if self.path == "/v1/models":
                    # pass the model list straight through
                    req = urllib.request.Request(server.upstream + "/models")
                    auth = self.headers.get("Authorization")
                    if auth:
                        req.add_header("Authorization", auth)
                    try:
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            self._send_json(resp.status, json.load(resp))
                    except urllib.error.HTTPError as e:
                        try:
                            self._send_json(e.code, json.load(e))
                        except ValueError:
                            self._send_json(e.code, {"error": {"message": "upstream error"}})
                    except OSError:
                        self._send_json(502, {"error": {"message": "upstream unreachable"}})
                    return
                self._send_json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self._send_json(404, {"error": {"message": "not found"}})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw)
                except ValueError:
                    self._send_json(400, {"error": {"message": "body is not json"}})
                    return
                if not isinstance(body, dict):
                    self._send_json(400, {"error": {"message": "body must be an object"}})
                    return
                if body.get("stream"):
                    # streaming bypasses the cache, relay the raw sse bytes
                    status, ctype, raw = server.forward_raw(body, self.headers)
                    self.send_response(status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("X-Semicache", "stream-passthrough")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                status, reply, cache_status, sim = server.handle_chat(body, self.headers)
                self._send_json(status, reply, cache_status, sim)

        httpd = ThreadingHTTPServer((self.cfg["bind"], self.cfg["port"]), Handler)
        print("semicache listening on %s:%d, upstream %s"
              % (self.cfg["bind"], self.cfg["port"], self.upstream))
        print("embeddings %s"
              % ("on" if self.embedder.enabled else "off (exact match only)"))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def run_server(cfg):
    CacheServer(cfg).serve()
