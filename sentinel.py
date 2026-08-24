#!/usr/bin/env python3
"""LLMINUX sentinel — NPU command parser.

Sends natural language commands to Qwen3-1.7B on the XDNA 2 NPU (via FLM)
and returns structured JSON verbs.

Requires: flm.service running on :52625 with qwen3:1.7b loaded.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

FLM_URL = "http://localhost:52625/api/chat"
MODEL = "qwen3:1.7b"
PROMPT_FILE = Path(__file__).parent / "sentinel_prompt.txt"


def load_system_prompt():
    return PROMPT_FILE.read_text().strip()


def fix_json(raw: str) -> str:
    s = raw.strip()
    if "<think>" in s:
        s = s.split("</think>")[-1].strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    while s.endswith("}") and s.count("{") < s.count("}"):
        s = s[:-1]
    return s


def parse_command(user_input: str, system_prompt: str) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_input} /no_think"},
        ],
        "stream": False,
        "options": {"num_predict": 100, "temperature": 0},
    })

    result = subprocess.run(
        ["curl", "-s", FLM_URL, "-H", "Content-Type: application/json", "-d", payload],
        capture_output=True, text=True, timeout=30,
    )

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"FLM request failed (exit {result.returncode}): {result.stderr[:200]}")

    resp = json.loads(result.stdout)
    if "error" in resp:
        raise RuntimeError(f"FLM error: {resp['error']}")

    content = fix_json(resp["message"]["content"])
    parsed = json.loads(content)

    eval_ms = round(resp.get("eval_duration", 0) / 1e6)
    total_ms = round(resp.get("total_duration", 0) / 1e6)
    parsed["_latency_ms"] = total_ms
    parsed["_eval_ms"] = eval_ms

    return parsed


def main():
    system_prompt = load_system_prompt()
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        try:
            result = parse_command(user_input, system_prompt)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print("LLMINUX sentinel (qwen3:1.7b on NPU)")
    print("Type a command, or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("llminux> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        try:
            result = parse_command(user_input, system_prompt)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"error: {e}")
        print()


if __name__ == "__main__":
    main()
