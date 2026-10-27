"""Day 1 demo — one hand-written tool driven through the agent loop.

Concept: a tool is just an object with a JSON schema (.spec) and a Python
callable (.run). The model decides when to call it; the loop executes it.

Design rules:
  * No framework: the tool, the event printer and the permission check are
    plain Python, written out in full so every moving part is visible.
  * Schema parameters are strings and .run converts them, because models
    are more reliable producing strings than typed numbers.

Run from the repo root:  python demos/day1_dice.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bajrang import provider  # noqa: E402
from bajrang.loop import run_loop  # noqa: E402


class RollDice:
    """A minimal tool: roll `count` six-sided dice."""

    spec = {"schema": {
        "name": "roll_dice",
        "description": "Roll count six-sided dice",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "string", "description": "How many dice"}},
            "required": ["count"],
        },
    }}

    def run(self, count):
        """Return a list of `count` random rolls between 1 and 6."""
        return [random.randint(1, 6) for _ in range(int(count))]


def on_event(kind, payload):
    """Print each loop event as one transcript line."""
    if kind == "assistant":
        for call in payload["tool_calls"]:
            print(f"assistant -> tool call {call['name']}({call['args']})")
        if payload["text"]:
            print(f"assistant: {payload['text']}")
    elif kind == "tool_end":
        print(f"tool result [{payload['name']}]: {payload['result']}")


def allow_all(call):
    """Permit every tool call (day 2 replaces this with real policy)."""
    return None


def main(task="Roll 3 dice and tell me whether the total beats 10"):
    """Run one task through the loop and return the final answer."""
    messages = [{"role": "user", "text": task}]
    print(f"user: {task}")
    return run_loop(provider.DEFAULT_MODEL, "You are a helpful assistant.",
                    messages, {"roll_dice": RollDice()}, on_event, allow_all)


if __name__ == "__main__":
    # An optional command-line argument replaces the default task.
    if len(sys.argv) > 1:
        main(" ".join(sys.argv[1:]))
    else:
        main()
