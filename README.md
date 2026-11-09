semicache - stop paying twice for the same answer [ai infra tool]

disclaimer:
this is a scratch caching proxy i wrote for my own agent work, not a
serious production gateway. it sits between your code and any
openai-compatible chat api and answers repeat prompts from disk instead
of paying for them again. if you want real infra, use gptcache or
something maintained.

why i made this:
i run the same prompts through my agents over and over, same question
with different temperature, same system prompt, same eval set against
probe and verdict, and every repeat costs the same as the first one.
that annoyed me enough to write the obvious thing: a local proxy that
checks a cache before it calls the model. exact matches hit on the full
request (temperature and seed dont matter, they never change the
question), and if you point it at an embeddings endpoint it also hits
on near-duplicates, "what is the capital of france" and "what is the
capital of france in europe" are the same question and should cost
once. entries expire after a ttl, old ones get evicted lru-style, and
stats tells you how many tokens and dollars you didnt spend. i avoided
requests and redis and all of that, stdlib only, the cache is a json
file in ~/.semicache so you can read it with your eyes. streaming
requests cache too: the chunks get collected as they fly past, and
when the stream ends cleanly the assembled reply is stored like any
other. the next identical request, streamed or not, replays it as
sse. tool-call streams and streams cut off mid-flight still pass
through uncached, there is nothing safe to store from those.

the guide

```bash
pip install .
semicache init
# edit ~/.semicache/semicache.json, set upstream_base_url
semicache serve
```

point your client at http://127.0.0.1:8788/v1 instead of the real api.
same openai-compatible shape, same auth header forwarded. every reply
carries an X-Semicache header: miss, hit-exact, hit-semantic (with an
X-Semicache-Similarity score), or stream-passthrough, which now only
means the stream could not be cached (tool calls, cut off, upstream
error). a cached stream replays with hit-exact or hit-semantic.

```bash
semicache stats          # hits, misses, tokens and dollars saved
semicache stats --json
semicache clear          # empty the cache, reset stats
semicache check          # validate config, ping the upstreams
```

config keys, all in semicache.json:

- upstream_base_url: the real chat api, required
- embeddings_base_url: optional, empty means exact match only
- embeddings_model: sent with embedding requests
- similarity_threshold: 0.85 by default, a paraphrase needs to clear it
- ttl_seconds: 86400 by default, 0 means entries never expire
- max_entries: 1000 by default, least recently hit evicted first
- port / bind: 8788 on 127.0.0.1 by default, it never listens publicly
- embed_text: last_user or all_user, what gets embedded for matching
- ignore_keys: request params that never change the cache key
  (temperature, top_p, max_tokens, seed by default)

the cost math uses a small built-in price table for common models.
unlisted models still cache fine, their saved cost just shows as
$0.0000 and stats notes how many hits that was.

tests: python3 tests/test_semicache.py, 96 checks, fake chat and
embeddings servers on 127.0.0.1, no api key needed.
