#!/usr/bin/env python3
"""Executable spec: chat templates (SmolLM2, Qwen3, Qwen3.5) and parsing of model output (reasoning, answer, tool calls).

Jinja-free algorithms in the shape we port to Rust. Rendering is verified against tests/golden/chat_*.json (the exact output of each model's own Jinja
template under Hugging Face transformers: 28 + 56 + 56 cases), also with the messages in OpenAI wire form. Parsing is verified by the table below.
    python tools/ref_chat_templates.py       (str.strip() is Python's: in Rust use a py_isspace = Unicode White_Space plus U+001C..U+001F)
Research note: Projects/ForgeInfer/Research/ForgeInfer Research - Chat Templates and Tool Call Parsing.md
"""
import json, os, sys

class TemplateError(Exception): pass

ESC = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}

def py_json(v):
    """json.dumps(v, ensure_ascii=False): what `tojson` prints (separators ', ' and ': ', key order kept, non-ASCII kept)."""
    if isinstance(v, dict):
        return "{" + ", ".join(py_json(str(k)) + ": " + py_json(x) for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(py_json(x) for x in v) + "]"
    if isinstance(v, str):
        return '"' + "".join(ESC.get(c) or (c if c >= " " else "\\u%04x" % ord(c)) for c in v) + '"'
    return "null" if v is None else "true" if v is True else "false" if v is False else repr(v)

def text_of(c):
    """Qwen3.5 render_content, text only: a string, a list of {"text": ..} parts, or None."""
    if c is None or isinstance(c, str):
        return c or ""
    if isinstance(c, list) and all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in c):
        return "".join(p["text"] for p in c)
    raise TemplateError("Unexpected content type.")

def jl(a):
    """json.loads that answers None for text that is not valid JSON."""
    try:
        return json.loads(a)
    except ValueError:
        return None

def is_query(text):
    """A user message that is not just tool results wrapped in a user turn."""
    return not (text.startswith("<tool_response>") and text.endswith("</tool_response>"))

def split_think(content, given=None):
    """(reasoning, answer) of an assistant message: the `reasoning_content` field if it is a string, else the <think> block split out of the text."""
    if isinstance(given, str):
        return given, content
    if "</think>" not in content:
        return "", content
    return content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n"), content.split("</think>")[-1].lstrip("\n")

def wire_to_template(messages):
    """OpenAI wire format -> template input: tool_call `arguments` arrive as a JSON string and must become an object (Qwen3.5 needs a mapping)."""
    out = json.loads(json.dumps(messages))
    for f in [c.get("function", c) for m in out for c in m.get("tool_calls") or []]:
        v = jl(f["arguments"] or "{}") if isinstance(f.get("arguments"), str) else None
        f["arguments"] = f["arguments"] if v is None else v          # text that is not JSON stays text (Qwen3 prints it as it is)
    return out

# ---------------------------------------------------------------------------------------------------------------- rendering
def render_smollm2(messages, tools=None, add_generation_prompt=True, enable_thinking=None):
    """ChatML with an injected default system turn; `tools`, `tool_calls` and `reasoning_content` are ignored by this template."""
    out = ""
    for i, m in enumerate(messages):
        if i == 0 and m["role"] != "system":
            out += "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        out += "<|im_start|>" + m["role"] + "\n" + (m.get("content") or "") + "<|im_end|>\n"
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

Q3_HEAD = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
Q3_TAIL = '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'

def render_qwen3(messages, tools=None, add_generation_prompt=True, enable_thinking=None):
    if not messages:
        raise TemplateError("no messages")
    s = lambda m: m["content"] if isinstance(m.get("content"), str) else ""      # a non-string content counts as ""
    n, out = len(messages), ""
    if tools:
        out += "<|im_start|>system\n" + (s(messages[0]) + "\n\n" if messages[0]["role"] == "system" else "")
        out += Q3_HEAD + "".join("\n" + py_json(t) for t in tools) + Q3_TAIL + "<|im_end|>\n"
    elif messages[0]["role"] == "system":
        out += "<|im_start|>system\n" + s(messages[0]) + "<|im_end|>\n"
    lq = next((i for i in range(n - 1, -1, -1) if messages[i]["role"] == "user" and isinstance(messages[i].get("content"), str) and is_query(messages[i]["content"])), n - 1)
    for i, m in enumerate(messages):                         # lq = index of the last real user query (none found: the last index, so no turn is "after" it)
        role, content = m["role"], s(m)
        if role == "user" or (role == "system" and i > 0):
            out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
        elif role == "assistant":
            reasoning, content = split_think(content, m.get("reasoning_content"))
            if i > lq and (i == n - 1 or reasoning):         # only turns after the last query keep a think block (the last one always gets one)
                out += "<|im_start|>assistant\n<think>\n" + reasoning.strip("\n") + "\n</think>\n\n" + content.lstrip("\n")
            else:
                out += "<|im_start|>assistant\n" + content
            for j, tc in enumerate(m.get("tool_calls") or []):
                tc = tc.get("function") or tc
                a = tc["arguments"]
                out += ("\n" if (j == 0 and content) or j > 0 else "") + '<tool_call>\n{"name": "' + tc["name"] + '", "arguments": ' + (a if isinstance(a, str) else py_json(a)) + "}\n</tool_call>"
            out += "<|im_end|>\n"
        elif role == "tool":                                 # consecutive tool results share one user turn
            out += ("<|im_start|>user" if i == 0 or messages[i - 1]["role"] != "tool" else "") + "\n<tool_response>\n" + content + "\n</tool_response>"
            out += "<|im_end|>\n" if i == n - 1 or messages[i + 1]["role"] != "tool" else ""
    if add_generation_prompt:
        out += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "")
    return out

Q35_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
Q35_TAIL = ("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n"
            "multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
            "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide "
            "optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, "
            "answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")

def render_qwen35(messages, tools=None, add_generation_prompt=True, enable_thinking=None, thinking_default=False):
    """thinking_default = what an ABSENT enable_thinking means: False in the 0.8B and 2B templates, True in the 4B and 9B ones (their only difference)."""
    if not messages:
        raise TemplateError("No messages provided.")
    n, out, T = len(messages), "", lambda m: text_of(m.get("content")).strip()      # EVERY content is trimmed in this template
    sys0 = T(messages[0]) if messages[0]["role"] == "system" else ""
    if tools:
        out += "<|im_start|>system\n" + Q35_HEAD + "".join("\n" + py_json(t) for t in tools) + Q35_TAIL + ("\n\n" + sys0 if sys0 else "") + "<|im_end|>\n"
    elif messages[0]["role"] == "system":
        out += "<|im_start|>system\n" + sys0 + "<|im_end|>\n"
    lq = next((i for i in range(n - 1, -1, -1) if messages[i]["role"] == "user" and is_query(T(messages[i]))), None)
    if lq is None:
        raise TemplateError("No user query found in messages.")
    for i, m in enumerate(messages):
        role, content = m["role"], T(m)
        if role == "system":
            if i > 0:
                raise TemplateError("System message must be at the beginning.")
        elif role == "user":
            out += "<|im_start|>user\n" + content + "<|im_end|>\n"
        elif role == "assistant":
            reasoning, content = split_think(content, m.get("reasoning_content"))
            out += "<|im_start|>assistant\n" + ("<think>\n" + reasoning.strip() + "\n</think>\n\n" if i > lq else "") + content   # every turn after the last query
            for j, tc in enumerate(m.get("tool_calls") or []):
                tc = tc.get("function") or tc
                out += ("\n\n" if j == 0 and content.strip() else "\n" if j > 0 else "") + "<tool_call>\n<function=" + tc["name"] + ">\n"
                a = tc.get("arguments") or {}
                if not isinstance(a, dict):
                    raise TemplateError("tool_call arguments must be an object")
                for k, v in a.items():                        # strings raw; objects and arrays as JSON; the rest as Python str(): True, None, 1.5
                    out += "<parameter=" + k + ">\n" + (py_json(v) if isinstance(v, (dict, list)) else str(v)) + "\n</parameter>\n"
                out += "</function>\n</tool_call>"
            out += "<|im_end|>\n"
        elif role == "tool":
            out += ("<|im_start|>user" if i > 0 and messages[i - 1]["role"] != "tool" else "") + "\n<tool_response>\n" + content + "\n</tool_response>"
            out += "<|im_end|>\n" if i == n - 1 or messages[i + 1]["role"] != "tool" else ""
        else:
            raise TemplateError("Unexpected message role.")
    if add_generation_prompt:                                 # "<think>\n" leaves the block OPEN: the model's first tokens are its reasoning
        out += "<|im_start|>assistant\n" + ("<think>\n" if (thinking_default if enable_thinking is None else enable_thinking) else "<think>\n\n</think>\n\n")
    return out

# ---------------------------------------------------------------------------------------------------------------- parsing model output
THINK_O, THINK_C, CALL_O, CALL_C = "<think>", "</think>", "<tool_call>", "</tool_call>"

def hold(buf, tags):
    """Length of the longest suffix of buf that is a proper prefix of one of the tags (that text must wait for more input)."""
    for k in range(min(len(buf), max(map(len, tags)) - 1), 0, -1):
        if any(len(t) > k and t.startswith(buf[len(buf) - k:]) for t in tags):
            return k
    return 0

def value_end(s, k):
    """Index just after the JSON value that starts at s[k], or None while the text ends inside it."""
    if s[k] not in '"{[':
        e = [x for x in (s.find(c, k) for c in ",}") if x >= 0]
        return min(e) if e else None
    depth, instr, esc = 0, False, False
    for i in range(k, len(s)):
        ch = s[i]
        if instr:
            esc, instr = not esc and ch == "\\", esc or ch != '"'
        elif ch == '"':
            instr = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if not instr and depth == 0 and ch in '"}]':
            return i + 1
    return None

def scan_json(s, closed, schemas):
    """Qwen3 body '{"name": N, "arguments": A}' -> (N or None, raw text of A so far, A complete?). Any key order; "parameters" counts as "arguments"."""
    name, args, done, i = None, "", False, s.find("{") + 1
    while i > 0:
        j = s.find('"', i)
        e = value_end(s, j) if j >= 0 else None
        c = s.find(":", e) if e else -1
        k = c + 1
        while 0 < k < len(s) and s[k] in " \t\r\n":
            k += 1
        if c < 0 or k >= len(s):
            break
        key, e = json.loads(s[j:e]), value_end(s, k)
        if key in ("arguments", "parameters"):
            args, done = s[k:e or len(s)], e is not None
        elif key == "name" and e:
            name = json.loads(s[k:e])
        if e is None:
            break
        i = e
    return name, args, done

def coerce(raw, typ):
    """Typed value of an XML parameter (vLLM's rules): 'null' is None for every type; text that does not fit its type stays a string."""
    t, r = str(typ).lower(), raw.strip()
    try:
        if r.lower() == "null": return None
        if t in ("integer", "int"): return int(r)
        if t in ("number", "float"): return int(float(r)) if float(r).is_integer() and not set(r.lower()) & set(".e") else float(r)
        if t in ("boolean", "bool"): return r.lower() == "true"
        if t in ("object", "array"): return json.loads(r)
    except ValueError:
        pass
    return raw

def scan_xml(s, closed, schemas):
    """Qwen3.5 body '<function=N>\\n<parameter=K>\\nV\\n</parameter>..</function>' -> (N or None, JSON text of the arguments so far, finished?).
    String values stream (an unfinished JSON string has no closing quote); other types appear once their end is seen."""
    i = s.find("<function=")
    j = s.find(">", i) if i >= 0 else -1
    if j < 0:
        return None, "", False
    name = s[i + 10:j].strip()
    props = (schemas.get(name) or {}).get("properties") or {}
    items, pos, finished, ends_at = [], j + 1, False, ("</parameter>", "<parameter=", "</function>")
    while True:
        p, f = s.find("<parameter=", pos), s.find("</function>", pos)
        if f >= 0 and (p < 0 or f < p):
            finished = True
            break
        q = s.find(">", p) if p >= 0 else -1
        if q < 0:
            break
        key, vs = s[p + 11:q].strip(), q + 1 + (s[q + 1:q + 2] == "\n")
        typ = (props.get(key) or {}).get("type", "string")
        found = [x for x in (s.find(t, vs) for t in ends_at) if x >= 0]
        raw = s[vs:min(found)] if found else s[vs:]
        if found or closed:                                   # the value is complete: its end tag came, or the call was cut and we make the best of it
            items.append(py_json(key) + ": " + py_json(coerce(raw[:-1] if raw.endswith("\n") else raw, typ)))
            if not found:
                break
            pos = min(found) + (12 if s.startswith("</parameter>", min(found)) else 0)
            continue
        if typ == "string":                                   # still streaming: hold back a possible end tag and one newline
            k = len(raw) - hold(raw, ends_at)
            k -= raw[:k].endswith("\n")
            items.append(py_json(key) + ": " + py_json(raw[:k])[:-1])
        break
    return name, "{" + ", ".join(items) + ("}" if finished or closed else ""), finished       # finished: </function> was really seen

class OutputParser:
    """Incremental parser; feed() takes any split of the text. Events: ('reason', s) ('content', s) ('call_start', i, name) ('call_args', i, fragment)
    ('call_end', i, complete). fmt 'json' = Qwen3 <tool_call>{..}</tool_call>, 'xml' = Qwen3.5 <function=..>. thinking_open: the prompt ended inside <think>."""
    def __init__(self, fmt, thinking_open=False, tools=None):
        self.fmt, self.schemas = fmt, {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools or []}
        self.mode, self.buf, self.ws, self.sent, self.started, self.n, self.cut = "reason" if thinking_open else "content", "", "", "", False, 0, True
        self.pre, self.back = "", self.mode                   # whitespace that preceded a tag; the mode a stray <tool_call> returns to

    def feed(self, text, final=False):
        self.buf, ev = self.buf + text, []
        while (self._call if self.mode == "call" else self._text)(ev, final):
            pass
        return ev

    def finish(self):
        return self.feed("", True)

    def _say(self, ev, s, boundary):                          # trailing whitespace waits: a tag, a tool call or the end may follow and swallow it
        s = self.ws + s
        self.ws, self.pre = ("", s[len(s.rstrip()):]) if boundary else (s[len(s.rstrip()):], "")
        if s.rstrip():
            ev.append((self.mode, s.rstrip()))

    def _text(self, ev, final):                               # modes 'content' and 'reason'
        tags = (THINK_O, CALL_O) if self.mode == "content" else (THINK_C, CALL_O)
        b = self.buf.lstrip("\n") if self.cut else self.buf   # newlines at the start of a section are dropped
        self.cut = self.cut and not b
        hits = [(b.find(t), t) for t in tags if t in b]
        if hits:
            i, t = min(hits)
            self._say(ev, b[:i], True)
            self.buf = b[i + len(t):]
            if t == CALL_O:                                   # also ends a reasoning block that was never closed
                self.back, self.mode, self.started, self.sent = self.mode, "call", False, ""
            else:
                self.mode, self.cut = ("reason" if t == THINK_O else "content"), True
            return True
        k = len(b) if final else len(b) - hold(b, tags)
        self._say(ev, b[:k], final)
        self.buf = b[k:]
        return False

    def _call(self, ev, final):
        b = self.buf
        end = b.find(CALL_C)
        closed = end >= 0 or final
        name, args, done = (scan_json if self.fmt == "json" else scan_xml)(b[:end] if end >= 0 else b, closed, self.schemas)
        if name is None and not closed:
            return False                                      # not even the name yet
        if name is None:                                      # not a call after all: keep the text, as Ember does
            ev.append((self.back, self.pre + CALL_O + (b[:end + len(CALL_C)] if end >= 0 else b)))
            self.buf, self.mode = b[end + len(CALL_C):] if end >= 0 else "", self.back
            return end >= 0
        if not self.started:
            ev.append(("call_start", self.n, name))
            self.started, self.n = True, self.n + 1
        if len(args) > len(self.sent):
            ev.append(("call_args", self.n - 1, args[len(self.sent):]))
            self.sent = args
        if closed:
            ev.append(("call_end", self.n - 1, done))
            self.buf, self.mode, self.cut = b[end + len(CALL_C):] if end >= 0 else "", "content", True
        return closed and end >= 0

def parse(fmt, text, cuts=(), **kw):
    """One-shot or chunked parse (text cut at the given offsets) -> {'reasoning', 'content', 'calls': [[name, arguments text]]}."""
    p, ev, pts = OutputParser(fmt, **kw), [], [0, *cuts, len(text)]
    for a, b in zip(pts, pts[1:]):
        ev += p.feed(text[a:b])
    out = {"reasoning": "", "content": "", "calls": []}
    for e in ev + p.finish():
        if e[0] in ("reason", "content"): out["reasoning" if e[0] == "reason" else "content"] += e[1]
        elif e[0] == "call_start": out["calls"].append([e[2], ""])
        elif e[0] == "call_args": out["calls"][e[1]][1] += e[2]
    return out

# ---------------------------------------------------------------------------------------------------------------- tests
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
TOOLS = [{"type": "function", "function": {"name": "run_python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "n": {"type": "integer"}, "ok": {"type": "boolean"}, "xs": {"type": "array"}}}}}]
J, X = "json", "xml"
CASES = [   # (format, parser options, model output, reasoning, answer, [(tool name, arguments or None when the text is not valid JSON)])
    (J, {}, 'I will do both.\n<tool_call>\n{"name": "a", "arguments": {"x": "}\\"{"}}\n</tool_call>\n<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>', "", "I will do both.", [("a", {"x": '}"{'}), ("b", {})]),
    (J, {}, '<think>\nplan\n</think>\n\nOK.\n\n<tool_call>\n{"arguments": {"a": 1}, "name": "k"}\n</tool_call>', "plan", "OK.", [("k", {"a": 1})]),
    (J, {}, '<tool_call>\n{"name": "k", "arguments": {"a": 1}}', "", "", [("k", {"a": 1})]),                       # no closing tag: the end of the output closes the call
    (J, {}, '<tool_call>\n{"name": "k", "arguments": {"a": [1, 2', "", "", [("k", None)]),                         # cut off by max_tokens: arguments stay invalid
    (J, {}, 'I would write <tool_call> to call a tool. a < b and 1<2 <b', "", "I would write <tool_call> to call a tool. a < b and 1<2 <b", []),
    (X, {"thinking_open": True}, 'Think.\n</think>\n\nHi!', "Think.", "Hi!", []),                                  # Qwen3.5 with thinking on: the block is already open
    (X, {"thinking_open": True}, 'Hmm.\n<tool_call>\n<function=f>\n</function>\n</tool_call>', "Hmm.", "", [("f", {})]),    # a tool call ends the reasoning
    (X, {"tools": TOOLS}, 'Sure.\n\n<tool_call>\n<function=run_python>\n<parameter=code>\nprint("hi")\nx = 1 < 2\n</parameter>\n</function>\n</tool_call>', "", "Sure.",
     [("run_python", {"code": 'print("hi")\nx = 1 < 2'})]),
    (X, {"tools": TOOLS}, '<tool_call>\n<function=run_python>\n<parameter=n>\n5\n</parameter>\n<parameter=ok>\nTrue\n</parameter>\n<parameter=xs>\n[1, "a"]\n</parameter>\n</function>\n</tool_call>',
     "", "", [("run_python", {"n": 5, "ok": True, "xs": [1, "a"]})]),                                             # the template prints Python's True/False: any case counts
    (X, {"tools": TOOLS}, '<tool_call>\n<function=run_python>\n<parameter=code>\nx\n<parameter=n>\n7', "", "", [("run_python", {"code": "x", "n": 7})]),   # missing end tags
    (X, {}, '<tool_call>\n<function=a>\n<parameter=p>\nv\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>', "", "", [("a", {"p": "v"}), ("b", {})]),
    (X, {}, '<tool_call>\ngarbage\n</tool_call>tail', "", "<tool_call>\ngarbage\n</tool_call>tail", []),
]

def check_parsers():
    bad = runs = 0
    for fmt, kw, text, reasoning, answer, calls in CASES:
        whole = parse(fmt, text, **kw)
        chunked = [parse(fmt, text, cuts=range(st, len(text), st), **kw) for st in (1, 2, 3, 5, 8)] + [parse(fmt, text, cuts=[i], **kw) for i in range(len(text) + 1)]
        ok = (whole["reasoning"], whole["content"], [(n, jl(a)) for n, a in whole["calls"]]) == (reasoning, answer, calls) and all(c == whole for c in chunked)
        runs, bad = runs + len(chunked), bad + (not ok)
        if not ok:
            print("PARSER FAIL (one-shot or chunked)", fmt, repr(text), "\n  got ", whole, "\n  want", reasoning, repr(answer), calls)
    p = OutputParser(X)                                                     # a call cut off by max_tokens is reported as incomplete: the engine must drop it
    cut = [e for e in p.feed("<tool_call>\n<function=f>\n<parameter=a>\nx") + p.finish() if e[0] == "call_end"]
    bad += cut != [("call_end", 0, False)]
    print(f"parser: {len(CASES) - bad}/{len(CASES)} cases ok, {runs} chunkings all give the one-shot result, truncated call flagged incomplete: {cut == [('call_end', 0, False)]}")
    return bad

def check_goldens():
    """Every golden case, from the stored messages and from the same conversation as a client sends it (arguments as a JSON string, content null)."""
    bad = 0
    for name, fn, fmt in (("smollm2", render_smollm2, None), ("qwen3", render_qwen3, J), ("qwen35", render_qwen35, X)):
        cases, b = json.load(open(os.path.join(GOLD, f"chat_{name}.json"), encoding="utf-8"))["cases"], 0
        for c in cases:
            wire = json.loads(json.dumps(c["messages"]))
            for m in wire:
                for tc in m.get("tool_calls") or []:
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"], separators=(",", ":"), ensure_ascii=False)
                m["content"] = None if m.get("tool_calls") and not m["content"] else m["content"]
            for msgs in (c["messages"], wire_to_template(wire)):
                got = fn(msgs, c["tools"], c["add_generation_prompt"], **c["kwargs"])
                b += got != c["text"]
                if got != c["text"]:
                    print(f"GOLDEN FAIL {name} {c['name']} gen={c['add_generation_prompt']} {c['kwargs']}\n  got  {got!r}\n  want {c['text']!r}")
            if fmt and c["name"] == "two_tool_calls" and not c["tools"] and not c["add_generation_prompt"]:     # the rendered assistant turn parses back into its message
                r, m = parse(fmt, c["text"].split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]), c["messages"][1]
                b += (r["content"], [(n, jl(a)) for n, a in r["calls"]]) != (m["content"], [(t["function"]["name"], t["function"]["arguments"]) for t in m["tool_calls"]])
        bad += b
        print(f"{name}: {len(cases)}/{len(cases)} golden renderings exact, each also from the OpenAI wire form" + (" and the tool-call turn parses back" if fmt else "") if not b else f"{name}: {b} FAILED")
    return bad

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    failures = check_goldens() + check_parsers()
    print("ALL OK" if not failures else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
