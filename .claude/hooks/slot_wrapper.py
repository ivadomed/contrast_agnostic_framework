#!/usr/bin/env python3
"""
PreToolUse hook: transparently wraps every Bash tool call in slot 0 via claude_slot.

Writes the command to a temp file (avoids all quoting/escaping issues),
then replaces the command with: claude_slot <tmpfile>; cleanup; exit $?

Skips wrapping if the command already references set_slot / ml_job / claude_slot.
"""
import json, sys, os, tempfile, stat

data = json.load(sys.stdin)

if data.get("tool_name") != "Bash":
    sys.exit(0)

cmd = data["tool_input"].get("command", "")

# Don't double-wrap
if any(kw in cmd for kw in ("ml_job", "claude_slot", "systemd-run", "set_slot")):
    sys.exit(0)

# Write command to a temp script — no quoting/escaping needed
try:
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="cslot_")
    with os.fdopen(fd, "w") as f:
        f.write(cmd)
    os.chmod(path, stat.S_IRWXU)
except Exception:
    sys.exit(0)  # graceful fallback: run unchanged

new_cmd = f"claude_slot {path}; _ec=$?; rm -f {path}; exit $_ec"

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": {"command": new_cmd},
    }
}))
