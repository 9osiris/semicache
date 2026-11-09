"""streaming cache: collect sse chunks, store the assembled reply, replay it."""

import json
import time


def assemble_sse(raw):
    """pull text, usage and a done flag out of raw sse bytes.

    returns text, usage, complete (saw data: [DONE]), and tool_calls
    (any chunk carried a tool call delta, those streams are not cacheable).
    comment lines and unparseable data lines are skipped, not fatal.
    """
    text = []
    usage = {}
    complete = False
    tool_calls = False
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            complete = True
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if "tool_calls" in delta or "function_call" in delta:
            tool_calls = True
        content = delta.get("content")
        if isinstance(content, str):
            text.append(content)
    return {"text": "".join(text), "usage": usage,
            "complete": complete, "tool_calls": tool_calls}


def stream_reply(model, text, usage):
    """the plain reply shape a stored stream entry keeps."""
    return {"id": "chatcmpl-cached", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage or {}}


def render_sse(model, text, usage=None, chunk_chars=16):
    """re-chunk a stored reply into sse bytes. slices are plain character
    windows, so the reassembled text is byte-identical to the original."""
    out = []
    for i in range(0, len(text), chunk_chars):
        chunk = {"id": "chatcmpl-cached", "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": model,
                 "choices": [{"index": 0,
                              "delta": {"content": text[i:i + chunk_chars]},
                              "finish_reason": None}]}
        out.append(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")
    final = {"id": "chatcmpl-cached", "object": "chat.completion.chunk",
             "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    if usage:
        final["usage"] = usage
    out.append(b"data: " + json.dumps(final).encode("utf-8") + b"\n\n")
    out.append(b"data: [DONE]\n\n")
    return b"".join(out)
