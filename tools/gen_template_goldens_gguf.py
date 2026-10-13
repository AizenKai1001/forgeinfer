"""Golden chat-template renderings recorded from the Jinja template stored IN a GGUF file (`tokenizer.chat_template`), rendered the way Hugging Face's `apply_chat_template` renders a template
(a sandboxed Jinja environment with trim_blocks / lstrip_blocks, the loop-controls extension and its `tojson` that keeps non-ASCII text). An ORACLE only: the Rust renderers are checked byte for byte
against these files. Needs only jinja2 (no transformers, no model weights).

    python tools/gen_template_goldens_gguf.py qwen25 <dev>/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

Writes tests/golden/chat_<name>.json in the layout of tools/gen_template_goldens.py: {"name", "cases": [{"name", "messages", "tools", "add_generation_prompt", "kwargs", "text"}]}.
"""
import json
import os
import struct
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_template_goldens import CONVERSATIONS, TOOLS  # noqa: E402  (the same conversations as the other templates)

# conversations that matter for templates with an injected default system turn
EXTRA = {
    "system_only": [{"role": "system", "content": "Only a system message."}],
    "system_then_assistant": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Bye"}],
    "mid_system": [{"role": "user", "content": "Hi"}, {"role": "system", "content": "From now on answer in French."}, {"role": "user", "content": "How are you?"}],
}


def read_chat_template(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF file"
    _version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def rs():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8")

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rv(t):
        if t == 8:
            return rs()
        if t in scalar:
            fmt = "<" + scalar[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            return [rv(et) for _ in range(n)]
        raise ValueError(f"metadata type {t}")

    for _ in range(n_kv):
        key = rs()
        t, = struct.unpack("<I", f.read(4))
        v = rv(t)
        if key == "tokenizer.chat_template":
            return v
    raise SystemExit("the file has no tokenizer.chat_template")


def environment():
    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    name, gguf = sys.argv[1], sys.argv[2]
    template = environment().from_string(read_chat_template(gguf))
    convs = dict(CONVERSATIONS)
    convs.update(EXTRA)
    cases = []
    for cname, msgs in convs.items():
        for use_tools in (False, True):
            for gen in (True, False):
                tools = TOOLS if use_tools else None
                try:
                    text = template.render(messages=msgs, tools=tools, add_generation_prompt=gen, bos_token="", eos_token="<|im_end|>")
                    err = None
                except Exception as e:  # a template may reject a combination
                    text, err = None, f"{type(e).__name__}: {e}"
                case = {"name": cname, "messages": msgs, "tools": tools, "add_generation_prompt": gen, "kwargs": {}, "text": text}
                if err:
                    case["error"] = err
                cases.append(case)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", f"chat_{name}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"name": name, "cases": cases}, f, ensure_ascii=False, indent=1)
    ok = sum(1 for c in cases if c["text"] is not None)
    print(f"{name}: {len(cases)} renderings ({ok} ok, {len(cases) - ok} rejected by the template) -> {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
