"""Day 1 — The provider: one function between the harness and the model.

Concept: everything above this file speaks a neutral message format; only
this file knows Gemini's wire format. Swapping providers means rewriting
this file and nothing else.

Design rules:
  * Standard library only — urllib, json, os, time.
  * One public call, complete(), with a fixed return shape:
    {"text": str, "tool_calls": [{"name", "args", "signature"}],
     "usage": {"input": int, "output": int}}.
  * Neutral messages in, neutral dicts out; translation lives in _to_wire.
  * Transient failures (rate limits, 5xx, network) retry with backoff;
    everything else fails loudly with the server's own words.
"""

import json
import os
import time
import urllib.error
import urllib.request

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

RETRY_STATUSES = {429, 500, 502, 503}


def api_key():
    """Return the API key from BAJRANG_API_KEY, falling back to GEMINI_API_KEY."""
    key = os.environ.get("BAJRANG_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found: set BAJRANG_API_KEY or GEMINI_API_KEY "
            "to a Gemini API key from https://aistudio.google.com/apikey")
    return key


def _to_wire(messages):
    """Translate neutral messages into Gemini `contents` entries."""
    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["text"]}]})
        elif m["role"] == "assistant":
            parts = [{"text": m["text"]}] if m.get("text") else []
            for call in m.get("tool_calls") or []:
                part = {"functionCall": {"name": call["name"], "args": call["args"]}}
                # Gemini 3 rejects a replayed functionCall unless the thought
                # signature it issued with that call is echoed back verbatim.
                if call.get("signature"):
                    part["thoughtSignature"] = call["signature"]
                parts.append(part)
            contents.append({"role": "model", "parts": parts})
        elif m["role"] == "tool":
            # Tool results travel as user-role functionResponse parts.
            response = {"name": m["name"], "response": {"result": m["text"]}}
            contents.append({"role": "user", "parts": [{"functionResponse": response}]})
    return contents


def _post(url, body, retries=5):
    """POST JSON to url and return the decoded reply, retrying transient errors."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code not in RETRY_STATUSES or attempt == retries - 1:
                detail = err.read().decode("utf-8", "replace")[:400]
                raise RuntimeError(f"Gemini API error {err.code}: {detail}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == retries - 1:
                raise RuntimeError(f"Gemini API unreachable: {err}") from err
        time.sleep(2 ** attempt * 2)
    raise RuntimeError("Gemini API: retries exhausted")


def complete(model, system, messages, tools):
    """Send one conversation turn to the model and return its neutral reply."""
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": _to_wire(messages),
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 65536},
    }
    if tools:
        body["tools"] = [{"functionDeclarations": [t["schema"] for t in tools]}]
    url = f"{API_ROOT}/{model}:generateContent?key={api_key()}"
    reply = _post(url, body)

    candidates = reply.get("candidates") or [{}]
    parts = candidates[0].get("content", {}).get("parts", [])
    text, calls = [], []
    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            calls.append({"name": fc["name"], "args": fc.get("args", {}),
                          "signature": part.get("thoughtSignature")})
        elif "text" in part and not part.get("thought"):
            text.append(part["text"])
    meta = reply.get("usageMetadata", {})
    usage = {"input": meta.get("promptTokenCount", 0),
             "output": meta.get("candidatesTokenCount", 0)}
    return {"text": "".join(text), "tool_calls": calls, "usage": usage}
