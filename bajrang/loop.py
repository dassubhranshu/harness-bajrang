"""Day 1 — The agent loop: call the model, run its tools, repeat.

Concept: an agent is a while-loop. The model either answers (done) or asks
for tools; the harness runs them, appends the results, and asks again.

Design rules:
  * The loop owns no policy: permission (before_tool), display (on_event)
    and context shaping (before_turn) are injected by the caller.
  * A tool failure is data for the model, never a crash of the loop.
  * The turn budget is finite; when it runs out, the model gets one last
    tool-free call to wrap up, so the caller always receives text.
"""

from . import provider


def run_loop(model, system, messages, tools, on_event, before_tool,
             max_turns=80, before_turn=None):
    """Drive the model/tool cycle until the model answers; return its text.

    tools maps name -> Tool (.spec, .run). messages is mutated in place so
    the caller keeps the full transcript.
    """
    specs = [t.spec for t in tools.values()]
    for _ in range(max_turns):
        if before_turn is not None:
            # Replace contents, not the binding: the caller holds this list.
            messages[:] = before_turn(messages)
        reply = provider.complete(model, system, messages, specs)
        messages.append({"role": "assistant", "text": reply["text"],
                         "tool_calls": reply["tool_calls"]})
        on_event("assistant", reply)
        if not reply["tool_calls"]:
            return reply["text"]
        for call in reply["tool_calls"]:
            on_event("tool_start", call)
            result = _execute(call, tools, before_tool)
            on_event("tool_end", {"name": call["name"], "result": result})
            messages.append({"role": "tool", "name": call["name"],
                             "text": str(result)})

    messages.append({"role": "user", "text": "Turn limit reached; wrap up now."})
    reply = provider.complete(model, system, messages, None)
    messages.append({"role": "assistant", "text": reply["text"], "tool_calls": []})
    on_event("assistant", reply)
    return reply["text"]


def _execute(call, tools, before_tool):
    """Run one tool call, returning its result or an error/blocked string."""
    tool = tools.get(call["name"])
    if tool is None:
        return f"ERROR: unknown tool {call['name']}"
    reason = before_tool(call)
    if reason is not None:
        return f"BLOCKED: {reason}"
    try:
        return tool.run(**call["args"])
    except Exception as exc:  # the model sees the failure; the loop survives
        return f"ERROR: {type(exc).__name__}: {exc}"
