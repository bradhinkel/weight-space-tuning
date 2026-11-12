"""A.9: send one prompt to all four demo servers and print the replies side by side (raw text, greedy).

    python exp_a/demo_query.py --item 0                  # item 0 of exp_a/demo_prompts.json (still over-called)
    python exp_a/demo_query.py --q "What's the weather in Paris?" --tools exp_a/demo_tools.json
    python exp_a/demo_query.py --q "Hi, who are you?"   # no tools

The chat template renders the tools into the system prompt (tool_choice "none" = no server-side parsing); replies are
shown raw so a <tool_call> is visible.
"""
import argparse
import json
import urllib.request
from pathlib import Path

HOST = "localhost"
SERVERS = {"base": 8101, "broken": 8102, "fable_b25": 8103, "lora": 8104}


def openai_tools(funcs):
    def fix(s):
        if isinstance(s, dict):
            s = {k: fix(v) for k, v in s.items()}
            if s.get("type") == "dict":
                s["type"] = "object"
        elif isinstance(s, list):
            s = [fix(v) for v in s]
        return s
    return [{"type": "function", "function": fix(f)} for f in funcs]


def ask(port, name, messages, tools, max_tokens):
    body = {"model": name, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False}
    if tools:
        # tool_choice "none": the chat template still renders the tools into the prompt, but vLLM does not parse the
        # reply, so the model's raw decision (a <tool_call> block or prose) comes back as text. ("auto" would need
        # --enable-auto-tool-choice plus a parser, and the parser would hide the raw output.)
        body["tools"], body["tool_choice"] = tools, "none"
    req = urllib.request.Request(f"http://{HOST}:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]
    return msg.get("content") or json.dumps(msg.get("tool_calls"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", type=int)
    ap.add_argument("--q")
    ap.add_argument("--tools", help="JSON file with a list of BFCL/OpenAI-style function specs")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--host", default="localhost")
    a = ap.parse_args()
    global HOST
    HOST = a.host
    if a.item is not None:
        it = json.loads((Path(__file__).parent / "demo_prompts.json").read_text())["prompts"][a.item]
        messages, funcs = it["question"], it["functions"]
    else:
        messages = [{"role": "user", "content": a.q}]
        funcs = json.loads(Path(a.tools).read_text()) if a.tools else []
    tools = openai_tools([f.get("function", f) for f in funcs]) if funcs else None
    print("USER:", messages[-1]["content"])
    print("TOOLS:", [f.get("function", f)["name"] for f in funcs])
    for name, port in SERVERS.items():
        try:
            out = ask(port, name, messages, tools, a.max_tokens)
        except Exception as e:  # server not up
            out = f"[error: {e}]"
        verdict = ("ERROR" if out.startswith("[error") else
                   "CALLS A TOOL" if "<tool_call>" in out or "<function=" in out else "answers in text")
        print(f"\n=== {name} ({verdict}) ===\n{out.strip()[:600]}")


if __name__ == "__main__":
    main()
