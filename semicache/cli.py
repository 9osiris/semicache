"""cli: init, serve, stats, clear, check."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from . import __version__
from .config import (DEFAULT_CONFIG_PATH, ConfigError, default_cache_file,
                     load, scaffold)
from .proxy import run_server
from .stats import Stats
from .store import CacheStore


def _stats_path(cfg):
    return os.path.join(os.path.dirname(cfg["cache_file"]) or ".", "stats.json")


def cmd_init(args):
    try:
        path = scaffold(args.config)
    except ConfigError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    print("wrote %s" % path)
    print("edit upstream_base_url, then run: semicache serve")
    return 0


def cmd_serve(args):
    try:
        cfg = load(args.config)
    except ConfigError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if args.port:
        cfg["port"] = args.port
    run_server(cfg)
    return 0


def cmd_stats(args):
    try:
        cfg = load(args.config)
    except ConfigError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    stats = Stats(_stats_path(cfg))
    store = CacheStore(cfg["cache_file"], cfg["ttl_seconds"], cfg["max_entries"])
    data = stats.to_dict()
    data["entries"] = store.count()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print("semicache stats")
    print("  requests      %d" % data["requests"])
    print("  exact hits    %d" % data["exact_hits"])
    print("  semantic hits %d" % data["semantic_hits"])
    print("  misses        %d" % data["misses"])
    print("  hit rate      %.1f%%" % (data["hit_rate"] * 100))
    print("  tokens saved  %d" % data["tokens_saved"])
    print("  cost saved    $%.4f" % data["cost_saved_usd"])
    if data["unknown_cost_hits"]:
        print("  note: %d hits were for unpriced models, cost shown is partial"
              % data["unknown_cost_hits"])
    print("  entries       %d" % data["entries"])
    return 0


def cmd_clear(args):
    try:
        cfg = load(args.config)
    except ConfigError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    store = CacheStore(cfg["cache_file"], cfg["ttl_seconds"], cfg["max_entries"])
    n = store.count()
    store.clear()
    stats_path = _stats_path(cfg)
    if os.path.exists(stats_path):
        os.remove(stats_path)
    print("cleared %d entries and reset stats" % n)
    return 0


def _ping(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError:
        return None


def cmd_check(args):
    try:
        cfg = load(args.config)
    except ConfigError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    ok = True
    print("config: ok (%s)" % (args.config or DEFAULT_CONFIG_PATH))
    store = CacheStore(cfg["cache_file"], cfg["ttl_seconds"], cfg["max_entries"])
    if store.corrupt:
        print("cache file: corrupt, would start empty")
    else:
        print("cache file: ok, %d entries" % store.count())
    status = _ping(cfg["upstream_base_url"].rstrip("/") + "/models")
    if status is None:
        print("upstream: unreachable (%s)" % cfg["upstream_base_url"])
        ok = False
    else:
        print("upstream: answered %d" % status)
    if cfg["embeddings_base_url"]:
        status = _ping(cfg["embeddings_base_url"].rstrip("/") + "/models")
        if status is None:
            print("embeddings: unreachable (%s)" % cfg["embeddings_base_url"])
            ok = False
        else:
            print("embeddings: answered %d" % status)
    else:
        print("embeddings: off, exact match only")
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="semicache",
        description="semantic cache for openai-compatible chat apis")
    parser.add_argument("--version", action="version",
                        version="semicache " + __version__)
    parser.add_argument("--config", default=None,
                        help="path to semicache.json (default ~/.semicache/semicache.json)")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="write an example config file")
    p_init.set_defaults(fn=cmd_init)

    p_serve = sub.add_parser("serve", help="run the cache proxy")
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(fn=cmd_serve)

    p_stats = sub.add_parser("stats", help="show hit/miss and savings stats")
    p_stats.add_argument("--json", action="store_true")
    p_stats.set_defaults(fn=cmd_stats)

    p_clear = sub.add_parser("clear", help="empty the cache and reset stats")
    p_clear.set_defaults(fn=cmd_clear)

    p_check = sub.add_parser("check", help="validate config and ping upstreams")
    p_check.set_defaults(fn=cmd_check)

    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 0
    return args.fn(args)
