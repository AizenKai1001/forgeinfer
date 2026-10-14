"""A small web chat page for a Forge-1 model, served by the standard library only.

    python3 chat_proxy.py [--port 8191] [--engine http://127.0.0.1:8190] [--key-file PATH] [--run-dir DIR]

GET /  -> the chat page (one HTML document, phone friendly, streaming);  /v1/...  -> forwarded to the engine (`forge serve ... --host 127.0.0.1 --port 8190 --api-key KEY`), with the API key added
here, so that the page needs no key and the engine can stay on localhost. Nothing is stored: the conversation lives in the browser tab.
With --run-dir DIR (a training run directory) it also serves /status (a live status page of the run) and /status.json, from tools/f1_live_status.py next to this file.
With --models-file FILE (JSON: {"models": {"name": {"ckpt": PATH, "note": TEXT}}}), --engine-bin and --tokenizer the page gets a model selector: /api/models lists the checkpoints, POST /api/select {"name": ...}
restarts the engine (`forge serve`, 127.0.0.1:8190) on the chosen checkpoint (one engine slot of the GPU: the switch takes a few seconds).
"""
import argparse
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forge-1 chat</title>
<style>
:root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --text:#e8eaf0; --dim:#9aa3b2; --accent:#ff8a3d; --user:#243247; }
@media (prefers-color-scheme: light) { :root { --bg:#f6f7f9; --panel:#ffffff; --line:#d9dde4; --text:#1b1f27; --dim:#5b6575; --accent:#d9570b; --user:#e3ecf8; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:16px/1.45 system-ui, sans-serif; display:flex; flex-direction:column; height:100dvh; }
header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
header b { color:var(--accent); font-size:18px; white-space:nowrap; }
header span { color:var(--dim); font-size:13px; }
header select { max-width:46%; padding:4px 6px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--text); font-size:13px; }
#log { flex:1; overflow-y:auto; padding:12px 16px; }
.msg { max-width:760px; margin:0 auto 10px; padding:10px 12px; border-radius:12px; white-space:pre-wrap; word-wrap:break-word; border:1px solid var(--line); background:var(--panel); }
.msg.user { background:var(--user); }
.msg small { display:block; color:var(--dim); margin-bottom:2px; font-size:12px; }
#bar { border-top:1px solid var(--line); padding:10px 16px; background:var(--panel); }
#row { max-width:760px; margin:0 auto; display:flex; gap:8px; }
textarea { flex:1; resize:none; height:52px; padding:8px 10px; border-radius:10px; border:1px solid var(--line); background:var(--bg); color:var(--text); font:inherit; }
button { padding:0 16px; border-radius:10px; border:0; background:var(--accent); color:#fff; font:inherit; font-weight:600; }
button.sec { background:var(--line); color:var(--text); }
#opts { max-width:760px; margin:8px auto 0; display:flex; gap:16px; flex-wrap:wrap; color:var(--dim); font-size:13px; align-items:center; }
#opts input[type=range] { vertical-align:middle; width:110px; }
#stat { margin-left:auto; }
#chips { max-width:760px; margin:0 auto 10px; display:flex; flex-wrap:wrap; gap:8px; }
.chip { padding:7px 12px; border-radius:16px; border:1px solid var(--line); background:var(--panel); color:var(--text); font-size:14px; font-weight:400; }
#intro { max-width:760px; margin:0 auto 10px; color:var(--dim); font-size:14px; }
.think { color:var(--dim); font-size:13px; border-left:3px solid var(--line); padding:2px 0 2px 8px; margin-bottom:6px; }
.think summary { cursor:pointer; }
.think div { white-space:pre-wrap; }
.stats { color:var(--dim); font-size:12px; margin-top:6px; }
.stats:empty { display:none; }
.md { white-space:normal; }
.md p { margin:0 0 8px; white-space:pre-wrap; }
.md p:last-child, .md ul:last-child, .md ol:last-child, .md pre:last-child { margin-bottom:0; }
.md ul, .md ol { margin:4px 0 8px; padding-left:22px; }
.md li { margin:2px 0; white-space:pre-wrap; }
.md code { font:0.92em ui-monospace, Menlo, Consolas, monospace; background:var(--bg); padding:1px 4px; border-radius:4px; }
.md pre { margin:6px 0 8px; padding:8px 10px; background:var(--bg); border:1px solid var(--line); border-radius:8px; overflow-x:auto; white-space:pre; }
.md pre code { padding:0; background:none; }
.md .mdh { font-weight:700; margin:8px 0 4px; }
.md .mdq { border-left:3px solid var(--line); padding-left:8px; color:var(--dim); margin:4px 0 8px; white-space:pre-wrap; }
.md hr { border:0; border-top:1px solid var(--line); margin:8px 0; }
.md a { color:var(--accent); }
</style></head><body>
<header><b id="brand">Forge-1</b><span id="sub">loading...</span><select id="model" style="display:none"></select></header>
<div id="log"></div>
<div id="bar">
  <div id="row"><textarea id="inp" placeholder="Message Forge-1"></textarea><button id="send">Send</button><button id="stop" class="sec" style="display:none">Stop</button><button id="reset" class="sec">New</button></div>
  <div id="opts"><label>temperature <input id="temp" type="range" min="0" max="1.4" step="0.05" value="0.5"> <span id="tv">0.5</span></label>
  <label>max tokens <input id="maxt" type="range" min="32" max="600" step="16" value="400"> <span id="mv">400</span></label>
  <label id="thinkl" style="display:none"><input id="think" type="checkbox"> think</label><span id="stat"></span></div>
</div>
<script>
const NL = String.fromCharCode(10);
const log = document.getElementById("log"), inp = document.getElementById("inp");
const sendBtn = document.getElementById("send"), stat = document.getElementById("stat"), stopBtn = document.getElementById("stop");
const sel = document.getElementById("model"), thinkBox = document.getElementById("think");
let conv = {h: []}, busy = false, alias = "forge-1", ctrl = null, cfg = {};
const DEFAULT_SYSTEM = "You are a helpful assistant.";   // sent with every request: the chat fine-tunes saw it in training, and the patched model refuses harmful requests far more reliably with it (57 percent without, 82 percent with it, measured on the page)
const DEFAULT_INTRO = "A 150M-parameter language model trained from scratch (no pretrained weights), with a short chat fine-tune. It is small and early: expect simple answers, and expect it to be wrong often. Tap an example or type your own.";
const EXAMPLES = ["What is the capital of France?", "Explain photosynthesis in two sentences.", "Give me three tips for studying.", "Write a haiku about the sea.", "What is 17 + 25?"];
function title() { return cfg.title || "Forge-1"; }
function showIntro() {
  for (const id of ["intro", "chips"]) { const e = document.getElementById(id); if (e) e.remove(); }
  const w = document.createElement("div"); w.id = "intro";
  w.textContent = cfg.intro || DEFAULT_INTRO;
  if (!cfg.intro) { const a = document.createElement("a"); a.href = "/status"; a.textContent = " Training status of the base model."; a.style.color = "var(--accent)"; w.appendChild(a); }
  const c = document.createElement("div"); c.id = "chips";
  for (const t of (cfg.examples || EXAMPLES)) { const b = document.createElement("button"); b.className = "chip"; b.textContent = t; b.onclick = () => { inp.value = t; send(); }; c.appendChild(b); }
  log.appendChild(w); log.appendChild(c);
}
function applyCfg() {
  document.getElementById("brand").textContent = title();
  document.title = title() + " chat";
  inp.placeholder = "Message " + title();
  const t = document.getElementById("temp"), m = document.getElementById("maxt");
  t.value = cfg.temperature !== undefined ? cfg.temperature : 0.5; document.getElementById("tv").textContent = t.value;
  m.max = cfg.max_tokens_limit || 600; m.value = cfg.max_tokens || 400; document.getElementById("mv").textContent = m.value;
  document.getElementById("thinkl").style.display = cfg.thinking ? "" : "none"; thinkBox.checked = !!cfg.thinking_default;
}
document.getElementById("temp").oninput = e => document.getElementById("tv").textContent = e.target.value;
document.getElementById("maxt").oninput = e => document.getElementById("mv").textContent = e.target.value;
function mdEl(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function isAlnum(c) { return c !== undefined && c !== "" && ((c >= "0" && c <= "9") || (c >= "a" && c <= "z") || (c >= "A" && c <= "Z")); }
// inline markdown: `code`, **bold**, *italic* / _italic_, [text](https://link); everything else is text (the DOM is built node by node, nothing is ever parsed as HTML)
function mdInline(parent, s) {
  let i = 0, buf = "";
  const flush = () => { if (buf) { parent.appendChild(document.createTextNode(buf)); buf = ""; } };
  while (i < s.length) {
    const ch = s[i];
    if (ch === "`") {
      const j = s.indexOf("`", i + 1);
      if (j > i + 1) { flush(); const c = mdEl("code"); c.textContent = s.slice(i + 1, j); parent.appendChild(c); i = j + 1; continue; }
    } else if (ch === "*" && s[i + 1] === "*") {
      const j = s.indexOf("**", i + 2);
      if (j > i + 2) { flush(); const b = mdEl("strong"); mdInline(b, s.slice(i + 2, j)); parent.appendChild(b); i = j + 2; continue; }
    } else if ((ch === "*" || ch === "_") && i + 1 < s.length && s[i + 1] !== " " && (i === 0 || !isAlnum(s[i - 1]))) {
      const j = s.indexOf(ch, i + 1);
      if (j > i + 1 && s[j - 1] !== " " && !isAlnum(s[j + 1])) { flush(); const e = mdEl("em"); mdInline(e, s.slice(i + 1, j)); parent.appendChild(e); i = j + 1; continue; }
    } else if (ch === "[") {
      const j = s.indexOf("](", i + 1);
      const k = j > 0 ? s.indexOf(")", j + 2) : -1;
      if (j > i + 1 && k > j) {
        const url = s.slice(j + 2, k);
        if (url.startsWith("http://") || url.startsWith("https://")) { flush(); const a = mdEl("a"); a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; mdInline(a, s.slice(i + 1, j)); parent.appendChild(a); i = k + 1; continue; }
      }
    }
    buf += ch; i++;
  }
  flush();
}
function mdListItem(ln) {
  let k = 0; while (k < ln.length && ln[k] === " ") k++;
  const rest = ln.slice(k);
  if (rest.length > 2 && (rest.startsWith("- ") || rest.startsWith("* ") || rest.startsWith("+ "))) return {ordered: false, indent: k, text: rest.slice(2)};
  let d = 0; while (d < rest.length && rest[d] >= "0" && rest[d] <= "9") d++;
  if (d > 0 && d < 10 && (rest[d] === "." || rest[d] === ")") && rest[d + 1] === " ") return {ordered: true, indent: k, start: parseInt(rest.slice(0, d)), text: rest.slice(d + 2)};
  return null;
}
function mdHeading(t) {
  let n = 0; while (n < t.length && t[n] === "#") n++;
  return n > 0 && n <= 6 && t[n] === " " ? n : 0;
}
// block markdown: fenced code, headings, bullet and numbered lists, quotes, rules, paragraphs (a single newline inside a paragraph stays a line break)
function renderMd(root, text) {
  root.textContent = "";
  const lines = text.split(NL);
  const para = [];
  const flushPara = () => { if (para.length) { const p = mdEl("p"); mdInline(p, para.join(NL)); root.appendChild(p); para.length = 0; } };
  let i = 0;
  while (i < lines.length) {
    const ln = lines[i], t = ln.trim();
    if (t.startsWith("```")) {   // to the closing fence, or to the end of the text: a streaming answer has no closing fence yet
      flushPara();
      const pre = mdEl("pre"), code = mdEl("code"), body = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) { body.push(lines[i]); i++; }
      i++;
      code.textContent = body.join(NL); pre.appendChild(code); root.appendChild(pre);
      continue;
    }
    if (t === "") { flushPara(); i++; continue; }
    const h = mdHeading(t);
    if (h) { flushPara(); const d = mdEl("div", "mdh"); mdInline(d, t.slice(h + 1)); root.appendChild(d); i++; continue; }
    if (t === "---" || t === "***" || t === "___") { flushPara(); root.appendChild(mdEl("hr")); i++; continue; }
    if (t.startsWith(">")) {
      flushPara();
      const q = mdEl("div", "mdq"), qs = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) { qs.push(lines[i].trim().slice(1).trim()); i++; }
      mdInline(q, qs.join(NL)); root.appendChild(q);
      continue;
    }
    const li = mdListItem(ln);
    if (li) {
      flushPara();
      const list = mdEl(li.ordered ? "ol" : "ul");
      if (li.ordered && li.start !== 1) list.start = li.start;
      while (i < lines.length) {
        const m = mdListItem(lines[i]);
        if (!m || m.ordered !== li.ordered) break;
        const item = mdEl("li"), parts = [m.text];
        i++;
        while (i < lines.length && lines[i].trim() !== "" && !mdListItem(lines[i]) && lines[i].startsWith(" ")) { parts.push(lines[i].trim()); i++; }
        mdInline(item, parts.join(NL));
        if (m.indent > li.indent) item.style.marginLeft = (8 * (m.indent - li.indent)) + "px";
        list.appendChild(item);
      }
      root.appendChild(list);
      continue;
    }
    para.push(ln); i++;
  }
  flushPara();
}
function add(role, text) {
  const d = document.createElement("div"); d.className = "msg " + role;
  const s = document.createElement("small"); s.textContent = role === "user" ? "you" : title();
  const box = document.createElement("details"); box.className = "think"; box.style.display = "none";
  const sm = document.createElement("summary"); sm.textContent = "thinking"; const tx = document.createElement("div"); box.appendChild(sm); box.appendChild(tx);
  const b = document.createElement(role === "user" ? "span" : "div"); if (role !== "user") b.className = "md"; b.textContent = text;
  const st = document.createElement("div"); st.className = "stats";
  d.appendChild(s); d.appendChild(box); d.appendChild(b); d.appendChild(st); log.appendChild(d); log.scrollTop = log.scrollHeight;
  return {out: b, think: tx, box: box, stats: st};
}
async function loadModels() {
  try {
    const m = await (await fetch("/api/models")).json();
    const cur = m.models && m.models.find(x => x.name === m.current);
    cfg = (cur && cur.page) || {};
    if (m.models && m.models.length > 1) {
      sel.textContent = "";
      for (const x of m.models) { const o = document.createElement("option"); o.value = x.name; o.textContent = x.name; o.title = x.note; sel.appendChild(o); }
      sel.value = m.current; sel.style.display = "";
      if (cur && cur.note) document.getElementById("sub").textContent = cur.note;
    }
  } catch (e) {}
  applyCfg();
  if (conv.h.length === 0 && !busy) showIntro();
}
sel.onchange = async () => {
  if (ctrl) ctrl.abort();
  document.getElementById("sub").textContent = "loading " + sel.value + " ..."; sel.disabled = true;
  try {
    const r = await fetch("/api/select", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({name: sel.value})});
    if (!r.ok) throw new Error("http " + r.status);
    conv = {h: []}; log.textContent = ""; stat.textContent = "";
  } catch (e) { document.getElementById("sub").textContent = "could not switch: " + e; }
  sel.disabled = false; await init();
};
async function init() {
  try {
    const r = await fetch("/v1/models"); const j = await r.json();
    if (j.data && j.data.length) alias = j.data[0].id;
    document.getElementById("sub").textContent = alias + " - 150M parameters, trained from scratch (small: check what matters)";
  } catch (e) { document.getElementById("sub").textContent = "engine not reachable"; }
  await loadModels();
}
function seconds(ms) { return ms < 1000 ? ms.toFixed(0) + " ms" : (ms / 1000).toFixed(1) + " s"; }
function statsLine(t, n, tFirst, finish) {
  const parts = [];
  if (t && t.predicted_n) parts.push(t.predicted_n + " tokens at " + t.predicted_per_second.toFixed(0) + " tok/s"); else if (n) parts.push(n + " tokens");
  if (tFirst) parts.push("first token " + seconds(tFirst));
  if (t && t.prompt_n + t.cache_n > 0) parts.push("prompt " + (t.prompt_n + t.cache_n) + " tokens" + (t.cache_n ? " (" + t.cache_n + " reused)" : ""));
  if (t && t.draft_n) parts.push("draft head right " + Math.round(100 * t.draft_n_accepted / t.draft_n) + " percent");
  if (finish === "length") parts.push("stopped at the token limit");
  return parts.join(" - ");
}
async function send() {
  const text = inp.value.trim(); if (!text || busy) return;
  busy = true; sendBtn.style.display = "none"; stopBtn.style.display = ""; inp.value = "";
  for (const id of ["intro", "chips"]) { const e = document.getElementById(id); if (e) e.remove(); }
  const c = conv; add("user", text); c.h.push({role: "user", content: text});
  const ui = add("assistant", ""); const out = ui.out;
  let acc = "", think = "", n = 0, aborted = false, tFirst = 0, timings = null, finish = "", pending = false, done = false, reading = false;
  const draw = () => { pending = false; if (!done) { renderMd(out, acc); log.scrollTop = log.scrollHeight; } };
  const t0 = performance.now();
  ctrl = new AbortController();
  try {
    const body = {model: alias, messages: [{role: "system", content: cfg.system !== undefined ? cfg.system : DEFAULT_SYSTEM}].concat(c.h), stream: true, stream_options: {include_usage: true}, return_progress: true,
                  temperature: parseFloat(document.getElementById("temp").value), max_tokens: parseInt(document.getElementById("maxt").value)};
    body.repetition_penalty = cfg.repetition_penalty !== undefined ? cfg.repetition_penalty : (/[0-9]|python|code|function|script|program/i.test(text) ? 1.0 : 1.1);
    if (cfg.thinking) body.chat_template_kwargs = {enable_thinking: thinkBox.checked};
    const r = await fetch("/v1/chat/completions", {method: "POST", signal: ctrl.signal, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    if (!r.ok) { out.textContent = "error " + r.status + " " + (await r.text()); throw new Error("http"); }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const {value, done} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split(NL); buf = lines.pop();
      for (const ln of lines) {
        if (!ln.startsWith("data:")) continue;
        const p = ln.slice(5).trim(); if (p === "[DONE]") continue;
        try {
          const j = JSON.parse(p); const ch = j.choices && j.choices[0]; const d = ch && ch.delta;
          if (ch && ch.finish_reason) finish = ch.finish_reason;
          if (j.timings) timings = j.timings;
          const pp = j.prompt_progress;
          if (pp && pp.total && !tFirst) {
            // a long prompt: how far the engine has read it (the answer replaces this line with its first token)
            reading = true;
            out.textContent = "Reading the prompt: " + pp.processed.toLocaleString("en-US") + " of " + pp.total.toLocaleString("en-US") + " tokens (" + Math.round(100 * pp.processed / pp.total) + " percent)";
          }
          const rc = d && d.reasoning_content, ct = d && d.content;
          if (rc || ct) {
            if (!tFirst) { tFirst = performance.now() - t0; if (reading) { out.textContent = ""; reading = false; } }
            n += 1;
            if (rc) { think += rc; ui.think.textContent = think; ui.box.style.display = ""; ui.box.open = !acc; }
            if (ct) { acc += ct; ui.box.open = false; if (!pending) { pending = true; setTimeout(draw, 40); } }
            log.scrollTop = log.scrollHeight; stat.textContent = (n / ((performance.now() - t0) / 1000)).toFixed(0) + " tok/s";
          }
        } catch (e) {}
      }
    }
    done = true;
    if (acc) { c.h.push({role: "assistant", content: acc}); renderMd(out, acc); }
    else { out.textContent = think ? "(no answer yet: the thinking used up the token limit, raise max tokens)" : "(empty answer)"; c.h.pop(); }
  } catch (e) {
    aborted = e && e.name === "AbortError"; done = true;
    if (aborted && acc) { c.h.push({role: "assistant", content: acc}); renderMd(out, acc); }
    else { if (!out.textContent) out.textContent = aborted ? "(stopped)" : "(no answer: " + e + ")"; c.h.pop(); }
  }
  ui.stats.textContent = statsLine(timings, n, tFirst, finish);
  if (timings && timings.predicted_n) stat.textContent = timings.predicted_per_second.toFixed(0) + " tok/s";
  busy = false; ctrl = null; stopBtn.style.display = "none"; sendBtn.style.display = ""; inp.focus();
}
stopBtn.onclick = () => { if (ctrl) ctrl.abort(); };
sendBtn.onclick = send;
inp.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
document.getElementById("reset").onclick = () => { if (ctrl) ctrl.abort(); conv = {h: []}; log.textContent = ""; stat.textContent = ""; showIntro(); };
init();
showIntro();
</script></body></html>
"""


class Switcher:
    """The one engine of the GPU and the checkpoints it may serve (names from the models file only: no path comes from a request)."""

    def __init__(self, models_file, engine_bin, tokenizer, engine_url, key, current_ckpt=""):
        with open(models_file, encoding="utf-8") as f:
            cfg = json.load(f)
        self.models = cfg["models"]
        self.q8 = bool(cfg.get("q8"))  # "q8": true at the top of the models file = every model runs on Q8 weights (forge serve --q8) unless its own entry says "q8": false
        self.models_file = models_file
        self.mtime = os.stat(models_file).st_mtime
        self.bin, self.tok, self.url, self.key = engine_bin, tokenizer, engine_url, key
        self.port = int(engine_url.rsplit(":", 1)[1].split("/")[0])
        self.lock = threading.Lock()
        self.current = next((n for n, m in self.models.items() if m["ckpt"] == current_ckpt), cfg.get("default") or next(iter(self.models)))
        self.log = os.path.join(os.path.dirname(os.path.abspath(models_file)), "chat_engine.log")
        # "idle_minutes" in a model's engine block (the 27B has 20): after that long without a request the engine is stopped (its 7 GB of GPU memory and 10 GB of RAM go back to the machine's other services:
        # Jellyfin, Tdarr, the desktop) and the next request starts it again (about 25 seconds for the 27B). The model stays selected: the page keeps showing it.
        self.cnt_lock = threading.Lock()
        self.active = 0
        self.last_used = time.time()
        self.asleep = False
        threading.Thread(target=self._idle_watch, daemon=True).start()

    def begin(self):
        """A request is about to use the engine: count it and wake the engine if it was put to sleep. False when it could not be started."""
        with self.cnt_lock:
            self.active += 1
            self.last_used = time.time()
        if self.asleep:
            with self.lock:
                if self.asleep:
                    if not self.start(self.current):
                        self.end()
                        return False
                    self.asleep = False
        return True

    def end(self):
        with self.cnt_lock:
            self.active = max(0, self.active - 1)
            self.last_used = time.time()

    def _idle_watch(self):
        poll = float(os.environ.get("CHAT_IDLE_POLL", "30"))
        while True:
            time.sleep(poll)
            try:
                mins = float(((self.models.get(self.current) or {}).get("engine") or {}).get("idle_minutes") or 0)
                if mins <= 0 or self.asleep:
                    continue
                with self.lock:  # (a request that wakes or switches the engine holds this lock)
                    with self.cnt_lock:
                        quiet = self.active == 0 and time.time() - self.last_used > mins * 60
                    if quiet and not self.asleep and self.alive():
                        with self.cnt_lock:  # (the flag is set before the engine is stopped and under the counter's lock: a request that arrives from now on sees it, waits for the lock and starts the engine again)
                            quiet = self.active == 0 and time.time() - self.last_used > mins * 60
                            if quiet:
                                self.asleep = True
                        if quiet:
                            self.kill()
            except Exception:  # noqa: BLE001
                pass

    def reload(self):
        """Re-reads the models file when it has changed: tools/chat_add_model.py adds a model without restarting the proxy and the engine (a restart of both costs the page about ten seconds). New names appear in the
        list, the model that is running stays running; a file that cannot be read or has no models leaves the list that works."""
        try:
            mtime = os.stat(self.models_file).st_mtime
            if mtime == self.mtime:
                return
            with open(self.models_file, encoding="utf-8") as f:
                cfg = json.load(f)
            models = cfg["models"]
            self.mtime = mtime
            if models:
                self.models = models
                self.q8 = bool(cfg.get("q8"))
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def listing(self):
        self.reload()
        return {"current": self.current, "models": [{"name": n, "note": m.get("note", ""), "page": m.get("page") or {}} for n, m in self.models.items()]}

    def alive(self):
        try:
            req = urllib.request.Request(self.url + "/v1/models")
            if self.key:
                req.add_header("Authorization", "Bearer " + self.key)
            return urllib.request.urlopen(req, timeout=3).status == 200
        except Exception:  # noqa: BLE001
            return False

    def native(self, name):
        """A downloaded model in a GGUF file: its quantized blocks are read as they are (forge serve --native; the file is already quantized, the Q8 switch is for our own full-precision checkpoints)."""
        m = self.models[name]
        return bool((m.get("engine") or {}).get("native", str(m["ckpt"]).endswith(".gguf")))

    def wants_q8(self, name):
        """Q8 weights for this model: its own "q8" entry, else the default of the models file; never for a native (GGUF) model."""
        if self.native(name):
            return False
        return bool(self.models[name].get("q8", self.q8))

    def engine_command(self, name, q8=False, speculate=True):
        """The `forge serve` command of a model. An entry without an "engine" block is one of our own checkpoints (chatml, 1024 tokens of context, the Forge-1 tokenizer and sampling); the block of a downloaded model
        names its tokenizer, chat format, context, sampling and the draft head ("mtp") of Qwen3.5, whose speculative decoding `speculate=False` turns off."""
        m = self.models[name]
        e = m.get("engine") or {}
        # "nice": the small models run at nice 10 (the machine's other services come first); the 27B's CPU feed-forward needs every core at full speed and loses a lot at nice 5 to 10, so its entry sets 0
        cmd = ["nice", "-n", str(int(e.get("nice", 10))), self.bin, "serve", m["ckpt"], "--tokenizer", str(e.get("tokenizer", self.tok)), "--host", "127.0.0.1", "--port", str(self.port), "--ctx", str(e.get("ctx", 1024)),
               "--alias", str(e.get("alias", "forge-1")), "--chat", str(e.get("chat", "chatml")), "--temp", str(e.get("temp", 0.7)), "--top-k", str(e.get("top_k", 40)), "--top-p", str(e.get("top_p", 0.95)),
               "--repeat-penalty", str(e.get("repeat_penalty", 1.1))]
        if e.get("min_p") is not None:
            cmd += ["--min-p", str(e["min_p"])]
        if self.native(name):
            cmd.append("--native")
        elif q8:
            cmd.append("--q8")
        if e.get("mtp"):
            cmd += ["--mtp", str(e["mtp"])] if speculate else ["--no-speculate"]
        if self.key:
            cmd += ["--api-key", self.key]
        return cmd

    def launch(self, name, q8, speculate=True):
        """Starts the engine on a model and waits up to 45 s (the model's engine.wait) for it to answer (a 9B model from a cold disk needs 15 s); True when it does."""
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in ((self.models[name].get("engine") or {}).get("env") or {}).items()})  # e.g. FORGE_RECIPE=... for a model stitched from several GGUF files
        with open(self.log, "ab") as lf:
            proc = subprocess.Popen(self.engine_command(name, q8, speculate), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        wait_s = float((self.models[name].get("engine") or {}).get("wait", 45))  # a 27B model with weights in host memory needs about 25 s warm and 80 s cold
        for _ in range(int(wait_s * 2)):
            time.sleep(0.5)
            if self.alive():
                return True
            if proc.poll() is not None:  # the engine has already ended (a model it cannot take): no use in waiting
                return False
        return False

    def kill(self):
        subprocess.run(["pkill", "-f", "forge[^ ]* serve .* --port %d" % self.port], check=False)
        time.sleep(1.5)

    def start(self, name):
        """Starts the engine on a model: the first try, then the fallbacks. A checkpoint that the Q8 path cannot take (a width that is no multiple of 64, an n-gram memory) runs on fp16 weights, a model whose
        draft head does not load runs without it. True when an engine answers."""
        q8 = self.wants_q8(name)
        tries = [(q8, True)]
        if q8:
            tries.append((False, True))
        if (self.models[name].get("engine") or {}).get("mtp"):
            tries.append((False, False))
        for i, (use_q8, speculate) in enumerate(tries):
            if i:
                self.kill()
            if self.launch(name, use_q8) if speculate else self.launch(name, use_q8, False):
                return True
        return False

    def select(self, name):
        self.reload()
        if name not in self.models:
            raise KeyError(name)
        with self.lock:
            if name == self.current and self.alive():
                return
            before = self.current
            self.kill()
            if self.start(name):
                self.current = name
                self.asleep = False
                with self.cnt_lock:
                    self.last_used = time.time()
                return
            if before != name and before in self.models:  # the page must not be left without an engine (a model that does not fit the GPU next to what is there): the model that was running comes back
                self.kill()
                if self.start(before):
                    self.asleep = False
            raise RuntimeError("the engine did not come up on " + name)


class Handler(http.server.BaseHTTPRequestHandler):
    engine = "http://127.0.0.1:8190"
    key = ""
    run_dir = ""
    status = None
    switcher = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self):
        self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def _forward(self, method):
        sw = self.switcher
        if sw is not None and method == "GET":  # a poll of the model list (the page, a monitor) neither counts as activity nor wakes a sleeping engine
            if sw.asleep:
                alias = ((sw.models.get(sw.current) or {}).get("engine") or {}).get("alias", "forge-1")
                return self._send(json.dumps({"object": "list", "data": [{"id": alias, "object": "model"}]}).encode("utf-8"), "application/json")
            return self._forward_engine(method)
        if sw is not None and not sw.begin():  # (the engine was asleep and could not be started again)
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            body = b"the model could not be started (is the GPU memory taken by another program?)"
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            return self._forward_engine(method)
        finally:
            if sw is not None:
                sw.end()

    def _forward_engine(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n) if n else None
        req = urllib.request.Request(self.engine + self.path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", "Bearer " + self.key)
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        except Exception as e:  # noqa: BLE001
            body = ("engine not reachable: %s" % e).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        while True:
            chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._page()
        if self.status is not None and self.path in ("/status", "/status/"):
            return self._send(self.status.PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if self.status is not None and self.path == "/status.json":
            return self._send(self.status.status_json(self.run_dir), "application/json")
        if self.path == "/api/models":
            return self._send(json.dumps(self.switcher.listing() if self.switcher else {"current": "", "models": []}).encode("utf-8"), "application/json")
        if self.path.startswith("/v1/"):
            return self._forward("GET")
        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/select" and self.switcher is not None:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                name = json.loads(self.rfile.read(n) or b"{}").get("name", "")
                self.switcher.select(name)
                return self._send(json.dumps(self.switcher.listing()).encode("utf-8"), "application/json")
            except KeyError:
                self.send_error(400, "unknown model")
                return
            except Exception as e:  # noqa: BLE001
                self.send_error(500, str(e)[:100])
                return
        if self.path.startswith("/v1/"):
            return self._forward("POST")
        self.send_error(404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--engine", default="http://127.0.0.1:8190")
    ap.add_argument("--key-file", default="")
    ap.add_argument("--run-dir", default="", help="training run directory: enables /status and /status.json")
    ap.add_argument("--models-file", default="", help="JSON list of checkpoints for the model selector (with --engine-bin and --tokenizer)")
    ap.add_argument("--engine-bin", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--current-ckpt", default="", help="the checkpoint the engine was started with (to name the current model)")
    a = ap.parse_args()
    Handler.engine = a.engine
    if a.run_dir:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import f1_live_status  # noqa: E402
        Handler.status = f1_live_status
        Handler.run_dir = a.run_dir
    if a.key_file:
        Handler.key = open(a.key_file).read().strip()
    if a.models_file and a.engine_bin and a.tokenizer:
        Handler.switcher = Switcher(a.models_file, a.engine_bin, a.tokenizer, a.engine, Handler.key, a.current_ckpt)
    Server(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
