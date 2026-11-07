import io
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from semicache.config import ConfigError, load, scaffold
from semicache.embed import Embedder, cosine
from semicache.match import (build_reply, exact_lookup, extract_reply_text,
                             filtered_params, semantic_lookup)
from semicache.pricing import cost_usd, price_for
from semicache.proxy import CacheServer, run_server
from semicache.stats import Stats
from semicache.store import (CacheStore, Entry, canonical_key, query_text)
from semicache.cli import main

CHECKS = [0, 0]


def check(name, cond):
    CHECKS[0] += 1
    if cond:
        CHECKS[1] += 1
        print("ok - %s" % name)
    else:
        print("FAIL - %s" % name)


def tmpdir():
    return tempfile.mkdtemp()


# fake embeddings: bag of words over a fixed vocab, l2 normalized.
# near-duplicate sentences share words, so they score high; unrelated ones low.
VOCAB = ["what", "is", "the", "capital", "of", "france", "germany", "spain",
         "how", "do", "i", "bake", "bread", "cake", "weather", "today",
         "in", "paris", "berlin", "madrid", "tell", "me", "a", "joke"]


def bow_vector(text):
    words = text.lower().replace("?", "").replace("'", "").split()
    vec = [float(words.count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else [0.0] * len(VOCAB)


class FakeUpstream(BaseHTTPRequestHandler):
    calls = 0
    embed_calls = 0

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            data = json.dumps({"object": "list",
                               "data": [{"id": "gpt-4o-mini"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/embeddings":
            FakeUpstream.embed_calls += 1
            text = body.get("input", "")
            data = json.dumps({"object": "list",
                               "data": [{"object": "embedding", "index": 0,
                                         "embedding": bow_vector(text)}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/v1/chat/completions":
            FakeUpstream.calls += 1
            if body.get("stream"):
                sse = (b"data: {\"choices\": [{\"delta\": {\"content\": \"streamed\"}}]}\n\n"
                       b"data: [DONE]\n\n")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse)))
                self.end_headers()
                self.wfile.write(sse)
                return
            last = ""
            for m in body.get("messages", []):
                if m.get("role") == "user":
                    last = m.get("content", "")
            reply = {"id": "chatcmpl-test", "object": "chat.completion",
                     "created": 1, "model": body.get("model", ""),
                     "choices": [{"index": 0,
                                  "message": {"role": "assistant",
                                              "content": "reply to: " + last},
                                  "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                               "total_tokens": 15}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()


def start_server(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def post_json(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def make_cfg(d, **over):
    cfg = {"upstream_base_url": "http://127.0.0.1:9/v1",
           "embeddings_base_url": "", "embeddings_model": "",
           "similarity_threshold": 0.85, "ttl_seconds": 86400,
           "max_entries": 1000, "port": 18791, "bind": "127.0.0.1",
           "cache_file": os.path.join(d, "cache.json"),
           "embed_text": "last_user",
           "ignore_keys": ["temperature", "top_p", "max_tokens", "seed"]}
    cfg.update(over)
    return cfg


def test_config():
    d = tmpdir()
    cfg = load(os.path.join(d, "missing.json"))
    check("config defaults load when file missing",
          cfg["port"] == 8788 and cfg["similarity_threshold"] == 0.85)
    bad = os.path.join(d, "bad.json")
    with open(bad, "w") as f:
        json.dump({"similarity_threshold": 1.5}, f)
    try:
        load(bad)
        check("config rejects threshold over 1", False)
    except ConfigError:
        check("config rejects threshold over 1", True)
    with open(bad, "w") as f:
        json.dump({"nope": 1}, f)
    try:
        load(bad)
        check("config rejects unknown keys", False)
    except ConfigError:
        check("config rejects unknown keys", True)
    with open(bad, "w") as f:
        json.dump({"ttl_seconds": -5}, f)
    try:
        load(bad)
        check("config rejects negative ttl", False)
    except ConfigError:
        check("config rejects negative ttl", True)
    with open(bad, "w") as f:
        json.dump({"embed_text": "all"}, f)
    try:
        load(bad)
        check("config rejects bad embed_text", False)
    except ConfigError:
        check("config rejects bad embed_text", True)
    with open(bad, "w") as f:
        json.dump({"port": 0}, f)
    try:
        load(bad)
        check("config rejects port 0", False)
    except ConfigError:
        check("config rejects port 0", True)
    with open(bad, "w") as f:
        json.dump({"max_entries": 0}, f)
    try:
        load(bad)
        check("config rejects max_entries 0", False)
    except ConfigError:
        check("config rejects max_entries 0", True)
    with open(bad, "w") as f:
        f.write("{not json")
    try:
        load(bad)
        check("config rejects broken json", False)
    except ConfigError:
        check("config rejects broken json", True)
    path = os.path.join(d, "new.json")
    scaffold(path)
    check("scaffold writes a file", os.path.exists(path))
    try:
        scaffold(path)
        check("scaffold refuses to overwrite", False)
    except ConfigError:
        check("scaffold refuses to overwrite", True)
    check("scaffolded config loads", load(path)["upstream_base_url"].startswith("http"))


def test_store():
    d = tmpdir()
    k1 = canonical_key("m", [{"role": "user", "content": "hi"}], {})
    k2 = canonical_key("m", [{"role": "user", "content": "hi"}], {})
    check("canonical key is stable", k1 == k2)
    k3 = canonical_key("m", [{"content": "hi", "role": "user"}], {})
    check("canonical key ignores dict order", k1 == k3)
    k4 = canonical_key("m", [{"role": "user", "content": "bye"}], {})
    check("different messages give different keys", k1 != k4)
    store = CacheStore(os.path.join(d, "c.json"), ttl_seconds=60, max_entries=2)
    e = Entry(k1, "m", [{"role": "user", "content": "hi"}],
              {"choices": [{"message": {"content": "hello"}}]},
              {"prompt_tokens": 3, "completion_tokens": 2},
              embedding=[0.1, 0.2])
    store.put(e)
    got = store.get_exact(k1)
    check("store roundtrip", got is not None and got.reply["choices"][0]["message"]["content"] == "hello")
    check("store keeps embedding", got.embedding == [0.1, 0.2])
    store.hit(got)
    check("hit bumps count", got.hits == 1)
    store2 = CacheStore(os.path.join(d, "c.json"), ttl_seconds=60, max_entries=2)
    check("store reloads from disk", store2.get_exact(k1) is not None)
    check("store reload keeps hits", store2.get_exact(k1).hits == 1)
    old = Entry("old", "m", [], {}, {}, created_at=time.time() - 120)
    store.put(old)
    check("expired entry is gone", store.get_exact("old") is None)
    store.put(Entry("a", "m", [], {}, {}))
    store.put(Entry("b", "m", [], {}, {}))
    store.put(Entry("c", "m", [], {}, {}))
    check("lru eviction caps entries", store.count() == 2)
    with open(os.path.join(d, "bad.json"), "w") as f:
        f.write("garbage{{{")
    bad = CacheStore(os.path.join(d, "bad.json"))
    check("corrupt cache starts empty", bad.corrupt and bad.count() == 0)
    check("missing message returns empty query text", query_text([]) == "")
    check("query text takes last user turn",
          query_text([{"role": "user", "content": "one"},
                      {"role": "assistant", "content": "x"},
                      {"role": "user", "content": "two"}]) == "two")
    check("query text all_user joins turns",
          query_text([{"role": "user", "content": "one"},
                      {"role": "user", "content": "two"}], "all_user") == "one\ntwo")


def test_embed_pricing():
    check("cosine identical is 1", abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9)
    check("cosine orthogonal is 0", abs(cosine([1, 0], [0, 1])) < 1e-9)
    check("cosine zero vector is 0", cosine([0, 0], [1, 0]) == 0.0)
    check("cosine opposite is -1", abs(cosine([1, 0], [-1, 0]) + 1.0) < 1e-9)
    emb = Embedder("")
    check("embedder disabled with no url", not emb.enabled and emb.embed("hi") is None)
    dead = Embedder("http://127.0.0.1:9/v1", timeout=2)
    check("embedder returns none on dead port", dead.embed("hi") is None)
    cost, unknown = cost_usd("gpt-4o-mini", 1000000, 1000000)
    check("pricing math for known model",
          abs(cost - 0.75) < 1e-9 and not unknown)
    cost, unknown = cost_usd("some-future-model", 1000, 1000)
    check("unknown model costs 0 and flags it", cost == 0.0 and unknown)
    check("provider prefix stripped",
          price_for("openrouter/gpt-4o-mini") == price_for("gpt-4o-mini"))


def test_match():
    params = filtered_params({"model": "m", "messages": [],
                              "temperature": 0.7, "top_p": 1,
                              "stream": False, "foo": 1},
                             ["temperature", "top_p"])
    check("filtered params drop ignored keys", params == {"foo": 1})
    d = tmpdir()
    store = CacheStore(os.path.join(d, "c.json"))
    msgs = [{"role": "user", "content": "hi"}]
    entry, key = exact_lookup(store, "m", msgs, {"temperature": 0.5}, ["temperature"])
    check("exact lookup misses on empty store", entry is None)
    store.put(Entry(key, "m", msgs, {"choices": [{"message": {"content": "hey"}}]},
                    {"prompt_tokens": 1, "completion_tokens": 1}))
    entry2, _ = exact_lookup(store, "m", msgs, {"temperature": 0.9}, ["temperature"])
    check("exact lookup hits despite ignored param change", entry2 is not None)
    entry3, _ = exact_lookup(store, "m", msgs, {"foo": "bar"}, ["temperature"])
    check("exact lookup misses on unignored param change", entry3 is None)
    e = Entry("k", "m", msgs, {"choices": [{"message": {"content": "hey"}}]},
              {"prompt_tokens": 1, "completion_tokens": 1},
              embedding=[1.0, 0.0])
    store.put(e)
    m, s = semantic_lookup(store, [1.0, 0.0], 0.85)
    check("semantic lookup finds close entry", m is not None and m.key == "k")
    m, s = semantic_lookup(store, [0.0, 1.0], 0.85)
    check("semantic lookup rejects far entry", m is None)
    m, s = semantic_lookup(store, None, 0.85)
    check("semantic lookup with no embedding is none", m is None)
    reply = build_reply(e, "other-model")
    check("build reply swaps model", reply["model"] == "other-model")
    check("extract reply text",
          extract_reply_text({"choices": [{"message": {"content": "yo"}}]}) == "yo")
    check("extract reply text handles garbage", extract_reply_text({}) == "")


def test_stats():
    d = tmpdir()
    path = os.path.join(d, "stats.json")
    s = Stats(path)
    s.record("exact", 10, 5, 0.01)
    s.record("semantic", 10, 5, 0.02)
    s.record("miss")
    check("stats counts hits and misses",
          s.data["exact_hits"] == 1 and s.data["semantic_hits"] == 1
          and s.data["misses"] == 1)
    check("stats hit rate", abs(s.hit_rate() - 2 / 3) < 1e-9)
    check("stats tokens saved", s.data["tokens_saved"] == 30)
    check("stats cost saved", abs(s.data["cost_saved_usd"] - 0.03) < 1e-9)
    s2 = Stats(path)
    check("stats persist", s2.data["requests"] == 3)


def test_proxy_e2e():
    upstream = start_server(FakeUpstream)
    uport = upstream.server_address[1]
    d = tmpdir()
    cfg = make_cfg(d,
                   upstream_base_url="http://127.0.0.1:%d/v1" % uport,
                   embeddings_base_url="http://127.0.0.1:%d/v1" % uport,
                   embeddings_model="bow")
    server = CacheServer(cfg)
    msgs = [{"role": "user", "content": "what is the capital of france"}]
    body = {"model": "gpt-4o-mini", "messages": msgs}
    FakeUpstream.calls = 0
    status, reply, kind, sim = server.handle_chat(body, {})
    check("first request is a miss", kind == "miss" and status == 200)
    check("miss calls upstream once", FakeUpstream.calls == 1)
    check("miss stores a reply",
          reply["choices"][0]["message"]["content"] == "reply to: what is the capital of france")
    status, reply, kind, sim = server.handle_chat(body, {})
    check("repeat request is an exact hit", kind == "hit-exact")
    check("exact hit does not call upstream", FakeUpstream.calls == 1)
    hot = {"model": "gpt-4o-mini", "messages": msgs, "temperature": 0.9}
    status, reply, kind, sim = server.handle_chat(hot, {})
    check("temperature change still hits exact", kind == "hit-exact")
    para = {"model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "what is the capital of france in europe"}]}
    status, reply, kind, sim = server.handle_chat(para, {})
    check("paraphrase hits semantic", kind == "hit-semantic" and sim is not None and sim >= 0.85)
    check("semantic hit does not call upstream", FakeUpstream.calls == 1)
    other = {"model": "gpt-4o-mini",
             "messages": [{"role": "user", "content": "how do i bake bread"}]}
    status, reply, kind, sim = server.handle_chat(other, {})
    check("unrelated prompt misses", kind == "miss")
    check("unrelated miss calls upstream", FakeUpstream.calls == 2)
    stream = {"model": "gpt-4o-mini", "messages": msgs, "stream": True}
    status, reply, kind, sim = server.handle_chat(stream, {})
    check("streaming returns passthrough sentinel",
          kind == "stream-passthrough" and status is None)
    status, reply, kind, sim = server.handle_chat({"model": "m", "messages": []}, {})
    check("empty messages is a 400", status == 400)
    stats = server.stats.to_dict()
    check("stats recorded the run",
          stats["requests"] == 5 and stats["exact_hits"] == 2
          and stats["semantic_hits"] == 1 and stats["misses"] == 2)
    check("stats saved tokens", stats["tokens_saved"] == 45)
    upstream.shutdown()


def test_http():
    upstream = start_server(FakeUpstream)
    uport = upstream.server_address[1]
    d = tmpdir()
    cfg = make_cfg(d, port=18792,
                   upstream_base_url="http://127.0.0.1:%d/v1" % uport,
                   embeddings_base_url="http://127.0.0.1:%d/v1" % uport,
                   embeddings_model="bow")
    t = threading.Thread(target=run_server, args=(cfg,), daemon=True)
    t.start()
    base = "http://127.0.0.1:18792"
    for _ in range(50):
        try:
            with urllib.request.urlopen(base + "/semicache/stats", timeout=2) as r:
                break
        except OSError:
            time.sleep(0.1)
    body = {"model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "what is the capital of france"}]}
    s1, h1, b1 = post_json(base + "/v1/chat/completions", body)
    check("http first request misses", h1.get("X-Semicache") == "miss")
    s2, h2, b2 = post_json(base + "/v1/chat/completions", body)
    check("http repeat hits exact", h2.get("X-Semicache") == "hit-exact")
    check("http hit body matches miss body",
          json.loads(b1)["choices"] == json.loads(b2)["choices"])
    para = {"model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "what is the capital of france in europe"}]}
    s3, h3, b3 = post_json(base + "/v1/chat/completions", para)
    check("http paraphrase hits semantic", h3.get("X-Semicache") == "hit-semantic")
    check("http semantic hit carries similarity header",
          h3.get("X-Semicache-Similarity") is not None
          and float(h3["X-Semicache-Similarity"]) >= 0.85)
    stream = {"model": "gpt-4o-mini",
              "messages": [{"role": "user", "content": "tell me a joke"}],
              "stream": True}
    s4, h4, b4 = post_json(base + "/v1/chat/completions", stream)
    check("http streaming relays sse",
          h4.get("X-Semicache") == "stream-passthrough"
          and b"data: [DONE]" in b4)
    with urllib.request.urlopen(base + "/semicache/stats", timeout=5) as r:
        stats = json.load(r)
    check("http stats endpoint works",
          stats["requests"] >= 3 and stats["exact_hits"] >= 1)
    with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
        models = json.load(r)
    check("http models passthrough", models["data"][0]["id"] == "gpt-4o-mini")
    try:
        urllib.request.urlopen(base + "/nope", timeout=5)
        check("http unknown path is 404", False)
    except urllib.error.HTTPError as e:
        check("http unknown path is 404", e.code == 404)
    upstream.shutdown()


def test_cli():
    d = tmpdir()
    cfgpath = os.path.join(d, "semicache.json")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--config", cfgpath, "init"])
    check("cli init writes config", rc == 0 and os.path.exists(cfgpath))
    with redirect_stdout(buf):
        rc = main(["--config", cfgpath, "stats"])
    check("cli stats runs on empty cache", rc == 0 and "requests" in buf.getvalue())
    with redirect_stdout(buf):
        rc = main(["--config", cfgpath, "clear"])
    check("cli clear runs", rc == 0 and "cleared 0" in buf.getvalue())
    # point the config at a dead upstream so check has something to fail on
    with open(cfgpath, "w") as f:
        json.dump({"upstream_base_url": "http://127.0.0.1:9/v1",
                   "embeddings_base_url": ""}, f)
    with redirect_stdout(buf):
        rc = main(["--config", cfgpath, "check"])
    check("cli check fails on dead upstream", rc == 1)
    try:
        with redirect_stdout(buf):
            main(["--version"])
        check("cli version prints", False)
    except SystemExit as e:
        check("cli version prints", e.code == 0 and "semicache" in buf.getvalue())


if __name__ == "__main__":
    test_config()
    test_store()
    test_embed_pricing()
    test_match()
    test_stats()
    test_proxy_e2e()
    test_http()
    test_cli()
    print("%d/%d checks passed" % (CHECKS[1], CHECKS[0]))
    sys.exit(0 if CHECKS[1] == CHECKS[0] else 1)
