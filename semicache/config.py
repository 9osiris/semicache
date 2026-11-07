"""config loading and validation for semicache."""

import json
import os

DEFAULTS = {
    "upstream_base_url": "http://127.0.0.1:11434/v1",
    "embeddings_base_url": "",
    "embeddings_model": "",
    "similarity_threshold": 0.85,
    "ttl_seconds": 86400,
    "max_entries": 1000,
    "port": 8788,
    "bind": "127.0.0.1",
    "cache_file": "",
    "embed_text": "last_user",
    "ignore_keys": ["temperature", "top_p", "max_tokens", "seed"],
}

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.semicache/semicache.json")


class ConfigError(Exception):
    pass


def default_cache_file():
    return os.path.expanduser("~/.semicache/cache.json")


def load(path=None):
    """load a config file, fill defaults, validate types. path may be None."""
    path = path or DEFAULT_CONFIG_PATH
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            raise ConfigError("cannot read config %s: %s" % (path, e))
        if not isinstance(raw, dict):
            raise ConfigError("config %s is not a json object" % path)
        unknown = sorted(set(raw) - set(DEFAULTS))
        if unknown:
            raise ConfigError("unknown config keys: %s" % ", ".join(unknown))
        cfg.update(raw)
    if not cfg["cache_file"]:
        cfg["cache_file"] = default_cache_file()
    validate(cfg)
    return cfg


def validate(cfg):
    """raise ConfigError on bad values. keeps bad configs from silently running."""
    if not isinstance(cfg["upstream_base_url"], str) or not cfg["upstream_base_url"]:
        raise ConfigError("upstream_base_url must be a non-empty string")
    if not isinstance(cfg["embeddings_base_url"], str):
        raise ConfigError("embeddings_base_url must be a string")
    if not isinstance(cfg["embeddings_model"], str):
        raise ConfigError("embeddings_model must be a string")
    th = cfg["similarity_threshold"]
    if not isinstance(th, (int, float)) or isinstance(th, bool) or not 0 < th <= 1:
        raise ConfigError("similarity_threshold must be a number in (0, 1]")
    if not isinstance(cfg["ttl_seconds"], int) or isinstance(cfg["ttl_seconds"], bool) \
            or cfg["ttl_seconds"] < 0:
        raise ConfigError("ttl_seconds must be a non-negative integer")
    if not isinstance(cfg["max_entries"], int) or isinstance(cfg["max_entries"], bool) \
            or cfg["max_entries"] < 1:
        raise ConfigError("max_entries must be a positive integer")
    if not isinstance(cfg["port"], int) or isinstance(cfg["port"], bool) \
            or not 1 <= cfg["port"] <= 65535:
        raise ConfigError("port must be an integer in 1..65535")
    if cfg["embed_text"] not in ("last_user", "all_user"):
        raise ConfigError("embed_text must be last_user or all_user")
    if not isinstance(cfg["ignore_keys"], list) \
            or not all(isinstance(k, str) for k in cfg["ignore_keys"]):
        raise ConfigError("ignore_keys must be a list of strings")


def scaffold(path=None):
    """write an example config. refuses to overwrite unless the file is missing."""
    path = path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        raise ConfigError("config already exists at %s" % path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    example = dict(DEFAULTS)
    example["cache_file"] = ""
    example["upstream_base_url"] = "https://api.openai.com/v1"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
        f.write("\n")
    return path
