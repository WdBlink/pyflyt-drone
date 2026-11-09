"""Development-only candidate; not installed in the active flight protocol.

Only unwrap one complete JSON code fence. Do not extract embedded prose, repair
arguments/JSON, call a model again or impute a counterfactual flight score.
"""


def unwrap_json_fence(text):
    if not isinstance(text, str):
        return text, False
    stripped = text.strip()
    lines = stripped.split("\n")
    if (len(lines) >= 3 and lines[0].strip().casefold() in ("```", "```json")
            and lines[-1].strip() == "```"
            and not any(line.strip().startswith("```") for line in lines[1:-1])):
        return "\n".join(lines[1:-1]), True
    return text, False
