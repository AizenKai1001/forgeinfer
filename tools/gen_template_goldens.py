"""Golden chat-template renderings for our own template code, recorded from the reference implementation (Hugging Face transformers + the model's own Jinja template, an ORACLE only).

Run on the homelab:   ~/forgeinfer/venv/bin/python tools/gen_template_goldens.py
Writes tests/golden/chat_<name>.json = {"name", "cases": [{"messages", "tools", "kwargs", "text"}]}: the exact string a chat server must feed the model for each conversation.
"""
import json
import os
import sys


ROOT = os.path.expanduser("~/forgeinfer/oracle/models")
MODELS = {"smollm2": "HuggingFaceTB__SmolLM2-135M-Instruct", "qwen3": "Qwen__Qwen3-0.6B", "qwen35": "Qwen__Qwen3.5-0.8B"}

TOOLS = [
    {"type": "function", "function": {"name": "lookup", "description": "Look something up in the encyclopedia.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "What to look for."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "run_python", "description": "Run a Python program and return what it printed.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

CONVERSATIONS = {
    "plain_user": [{"role": "user", "content": "Hello!"}],
    "system_user": [{"role": "system", "content": "You are Ember, a friendly assistant."}, {"role": "user", "content": "What is the capital of France?"}],
    "multi_turn": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "Tell me a joke."}],
    "ends_with_assistant": [{"role": "user", "content": "Say hi"}, {"role": "assistant", "content": "Hi!"}],
    "unicode_and_newlines": [{"role": "user", "content": "Zażółć gęślą jaźń\n\nSecond paragraph with <tags> & symbols: {braces} [brackets] \"quotes\" 日本語"}],
    "tool_round_trip": [
        {"role": "system", "content": "You are Ember."},
        {"role": "user", "content": "Who wrote Hamlet?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {"query": "Hamlet author"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "Hamlet is a tragedy by William Shakespeare."},
        {"role": "assistant", "content": "William Shakespeare wrote Hamlet."},
        {"role": "user", "content": "And Macbeth?"},
    ],
    "two_tool_calls": [
        {"role": "user", "content": "Compute 17*23 and look up Paris."},
        {"role": "assistant", "content": "I will do both.", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "run_python", "arguments": {"code": "print(17*23)"}}},
            {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": {"query": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "a", "name": "run_python", "content": "391"},
        {"role": "tool", "tool_call_id": "b", "name": "lookup", "content": "Paris is the capital of France."},
    ],
}


def main() -> int:
    from transformers import AutoTokenizer  # only the oracle run needs it: tools/gen_template_goldens_gguf.py imports the conversations above without it

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
    os.makedirs(out_dir, exist_ok=True)
    for name, folder in MODELS.items():
        tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, folder))
        cases = []
        for cname, msgs in CONVERSATIONS.items():
            for use_tools in (False, True):
                for gen in (True, False):
                    variants = [{}]
                    if name.startswith("qwen3"):
                        variants = [{"enable_thinking": True}, {"enable_thinking": False}]
                    for kw in variants:
                        tools = TOOLS if use_tools else None
                        try:
                            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen, **kw)
                        except Exception as e:                                  # a template may reject a combination (e.g. tool messages without tool support)
                            text = None
                            err = f"{type(e).__name__}: {e}"
                        case = {"name": cname, "messages": msgs, "tools": tools, "add_generation_prompt": gen, "kwargs": kw, "text": text}
                        if text is None:
                            case["error"] = err
                        cases.append(case)
        with open(os.path.join(out_dir, f"chat_{name}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"name": name, "cases": cases}, f, ensure_ascii=False, indent=1)
        ok = sum(1 for c in cases if c["text"] is not None)
        print(f"{name}: {len(cases)} renderings ({ok} ok, {len(cases) - ok} rejected by the template)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
