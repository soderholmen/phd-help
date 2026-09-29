"""Client-side tool-call validation (SPEC §2 tool-calling contract).

Every tool call the served model returns is checked before execution:
name in the offered set, deduped, ``arguments`` JSON-parsed, required
params present, and any per-tool check (e.g. "the patch's ``find`` anchor
actually exists in the section"). Rejections come back as one bounce
message each — the loop retries the model with them; they never reach the
user, and unparseable arguments are never eval'd.
"""

import json
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ValidCall:
    id: str
    name: str
    args: dict


def validate_tool_calls(calls, offered, validators=None):
    """Validate raw OpenAI-style tool calls against the offered contract.

    ``offered`` maps tool name -> required parameter names.
    ``validators`` optionally maps tool name -> callable(args) -> error
    message or None. Returns ``(valid, errors)`` where ``errors`` holds one
    bounce-back message per rejected call.
    """
    validators = validators or {}
    valid = []
    errors = []
    seen = set()
    for c in calls:
        fn = c.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments")
        if name not in offered:
            errors.append(f"unknown tool '{name}'; offered: "
                          f"{sorted(offered)}")
            continue
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            errors.append(f"tool '{name}' arguments are not valid JSON")
            continue
        if not isinstance(args, dict):
            errors.append(f"tool '{name}' arguments must be a JSON object")
            continue
        missing = [p for p in offered[name] if p not in args]
        if missing:
            errors.append(f"tool '{name}' is missing required "
                          f"parameter(s): {', '.join(missing)}")
            continue
        check = validators.get(name)
        if check is not None:
            problem = check(args)
            if problem:
                errors.append(f"tool '{name}' failed validation: {problem}")
                continue
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        valid.append(ValidCall(id=c.get("id", ""), name=name, args=args))
    return valid, errors
