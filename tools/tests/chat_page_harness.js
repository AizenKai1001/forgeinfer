// Runs the script of the chat page (tools/chat_proxy.py, PAGE) against a minimal fake DOM and a scripted fetch, and prints what the page shows as one JSON document.
//   node chat_page_harness.js PAGE_SCRIPT.js
// Used by tools/tests/test_chat_proxy.py (PageBehaviourTest); no browser needed.
const fs = require("fs");
const vm = require("vm");

class El {
  constructor(tag, id) {
    this.tag = tag; this.id = id || ""; this.children = []; this.style = {}; this.className = ""; this._text = ""; this._value = ""; this._max = ""; this.checked = false; this.open = false;
    this.parent = null; this.scrollTop = 0; this.scrollHeight = 0; this.disabled = false; this.placeholder = ""; this.title = ""; this.href = ""; this.listeners = {};
  }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(""); }  // the text of the element, then its children
  set textContent(v) { this.children = []; this._text = String(v); }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(x => x !== this); this.parent = null; }
  get value() { return this._value; }
  set value(v) { this._value = String(v); }  // like the value of a real input: always a string
  get max() { return this._max; }
  set max(v) { this._max = String(v); }
  focus() {}
  addEventListener(n, f) { this.listeners[n] = f; }
}

function build() {
  const root = new El("body", "");
  const ids = {};
  for (const id of ["brand", "sub", "model", "log", "inp", "send", "stop", "reset", "temp", "tv", "maxt", "mv", "think", "thinkl", "stat"]) { ids[id] = new El("x", id); root.appendChild(ids[id]); }
  ids.temp.value = "0.5"; ids.maxt.value = "400"; ids.model.style.display = "none"; ids.thinkl.style.display = "none"; ids.stop.style.display = "none";
  const find = (n, id) => { if (n.id === id) return n; for (const c of n.children) { const f = find(c, id); if (f) return f; } return null; };
  const document = { getElementById: id => find(root, id), createElement: tag => new El(tag), createTextNode: t => { const e = new El("#text"); e._text = String(t); return e; }, title: "" };
  return {root, ids, document};
}

function sseFetch(routes, calls) {
  return async (url, opts) => {
    calls.push({url, body: opts && opts.body ? JSON.parse(opts.body) : null});
    const r = routes[url];
    if (!r) return {ok: false, status: 404, text: async () => "no route " + url};
    if (r.json !== undefined) return {ok: true, status: 200, json: async () => r.json};
    if (r.status) return {ok: false, status: r.status, text: async () => r.text || ""};
    const enc = new TextEncoder().encode(r.sse.join("\n\n") + "\n\n");
    const pieces = [];
    for (let i = 0; i < enc.length; i += 37) pieces.push(enc.slice(i, i + 37));
    let k = 0;
    return {ok: true, status: 200, body: {getReader: () => ({read: async () => k < pieces.length ? {value: pieces[k++], done: false} : {value: undefined, done: true}})}};
  };
}

function tagsOf(el, acc) {
  acc = acc || {};
  for (const c of el.children) { if (c.tag !== "#text") { acc[c.tag] = (acc[c.tag] || 0) + 1; if (c.className) acc["." + c.className] = (acc["." + c.className] || 0) + 1; tagsOf(c, acc); } }
  return acc;
}
function textsOf(el, tag, out) {
  out = out || [];
  for (const c of el.children) { if (c.tag === tag) out.push(c.textContent); textsOf(c, tag, out); }
  return out;
}
function linksOf(el, out) {
  out = out || [];
  for (const c of el.children) { if (c.tag === "a") out.push({href: c.href, rel: c.rel, text: c.textContent}); linksOf(c, out); }
  return out;
}
const chunk = (delta, extra) => "data: " + JSON.stringify(Object.assign({choices: [{index: 0, delta: delta}]}, extra || {}));
const script = fs.readFileSync(process.argv[2], "utf8");

async function scenario(current, modelsList, sse, typed) {
  const {root, ids, document} = build();
  const calls = [];
  const routes = {"/v1/models": {json: {data: [{id: "alias-x"}]}}, "/api/models": {json: {current: current, models: modelsList}}, "/v1/chat/completions": sse};
  const ctx = vm.createContext({document, fetch: sseFetch(routes, calls), performance, AbortController, TextDecoder, parseFloat, parseInt, String, JSON, Math, Promise, setTimeout, console});
  vm.runInContext(script + "\n;globalThis.__t = {send, init, get cfg() { return cfg; }, get conv() { return conv; }, get alias() { return alias; }};", ctx);
  await new Promise(r => setTimeout(r, 30));  // init() at the end of the script
  const out = {};
  out.brand = ids.brand.textContent; out.title = document.title; out.placeholder = ids.inp.placeholder; out.sub = ids.sub.textContent;
  out.thinkVisible = ids.thinkl.style.display === ""; out.thinkChecked = ids.think.checked; out.maxMax = ids.maxt.max; out.maxValue = ids.maxt.value; out.temp = ids.temp.value;
  out.selectVisible = ids.model.style.display === ""; out.options = ids.model.children.map(o => o.textContent);
  const intro = document.getElementById("intro"); out.intro = intro ? intro.textContent : null; out.chips = (document.getElementById("chips") || {children: []}).children.length;
  out.alias = ids.model && vm.runInContext("__t.alias", ctx);
  if (typed !== undefined) {
    if (typed.thinkChecked !== undefined) ids.think.checked = typed.thinkChecked;
    ids.inp.value = typed.text;
    await vm.runInContext("__t.send()", ctx);
    const msgs = ids.log.children.filter(c => c.className.startsWith("msg"));
    out.messages = msgs.map(m => ({cls: m.className, label: m.children[0].textContent, think: m.children[1].children[1].textContent, thinkShown: m.children[1].style.display !== "none", thinkOpen: m.children[1].open,
                                   answer: m.children[2].textContent, tags: tagsOf(m.children[2]), strong: textsOf(m.children[2], "strong"), em: textsOf(m.children[2], "em"), code: textsOf(m.children[2], "code"), links: linksOf(m.children[2]), stats: m.children[3].textContent}));
    out.stat = ids.stat.textContent; out.history = vm.runInContext("JSON.stringify(__t.conv.h)", ctx); out.request = calls.filter(c => c.url === "/v1/chat/completions").map(c => c.body)[0];
    out.introGone = document.getElementById("intro") === null;
  }
  return out;
}

(async () => {
  const forge = {name: "Forge-1 365M", note: "ours", page: {}};
  const qwen = {name: "Qwen", note: "pretrained", page: {title: "Qwen3.5-9B", intro: "Qwen intro text", thinking: true, thinking_default: false, repetition_penalty: 1.0, temperature: 0.7, max_tokens: 1024, max_tokens_limit: 4096}};
  const both = [forge, qwen];
  const result = {};
  // 1. our own model: the page as it always was
  result.forge = await scenario("Forge-1 365M", both, {sse: [chunk({role: "assistant", content: ""}), chunk({content: "Par"}), chunk({content: "is."}),
    chunk({}, {choices: [{index: 0, delta: {}, finish_reason: "stop"}]}), "data: " + JSON.stringify({choices: [], usage: {}, timings: {prompt_n: 12, cache_n: 30, predicted_n: 2, predicted_per_second: 765.4, draft_n: 0, draft_n_accepted: 0}}), "data: [DONE]"]},
    {text: "What is the capital of France?"});
  // 2. a downloaded model with a thinking switch, thinking on: reasoning first, then the answer, then the numbers
  result.qwenThinking = await scenario("Qwen", both, {sse: [chunk({role: "assistant", content: ""}), chunk({reasoning_content: "17 times 20 is 340"}), chunk({reasoning_content: ", plus 51 is 391."}), chunk({content: "391"}),
    chunk({}, {choices: [{index: 0, delta: {}, finish_reason: "stop"}]}), "data: " + JSON.stringify({choices: [], usage: {}, timings: {prompt_n: 20, cache_n: 0, predicted_n: 30, predicted_per_second: 112.34, draft_n: 20, draft_n_accepted: 18}}), "data: [DONE]"]},
    {text: "What is 17 times 23?", thinkChecked: true});
  // 3. the thinking used up the limit: no answer, the history does not keep the question
  result.qwenNoAnswer = await scenario("Qwen", both, {sse: [chunk({reasoning_content: "hmm"}), chunk({}, {choices: [{index: 0, delta: {}, finish_reason: "length"}]}), "data: [DONE]"]}, {text: "hard question", thinkChecked: true});
  // 4. an error of the engine
  result.engineError = await scenario("Forge-1 365M", both, {status: 500, text: "boom"}, {text: "hello"});
  // 5. an answer with markdown: bold, italic, code, lists, a code block, a link, a heading, a quote; markup of other kinds stays text
  const NLC = String.fromCharCode(10);
  const mdText = ["Here is **bold** and *italic* and `code`.", "", "- item one", "- item two", "", "1. first", "2. second", "", "```py", "print(1)", "```", "",
                  "A [link](https://example.com) and [bad](javascript:alert(1)) and <b>tag</b> and 2*3*4 and a * b.", "", "# Title", "", "> quote"].join(NLC);
  const md = [mdText.slice(0, 20), mdText.slice(20)];  // two stream chunks, the first one ends inside an emphasis
  result.markdown = await scenario("Qwen", both, {sse: [chunk({role: "assistant", content: ""})].concat(md.map(t => chunk({content: t}))).concat([chunk({}, {choices: [{index: 0, delta: {}, finish_reason: "stop"}]}), "data: [DONE]"])}, {text: "format please"});
  // 6. a page whose list has one model only: no selector
  result.single = await scenario("Forge-1 365M", [forge], {sse: ["data: [DONE]"]});
  console.log(JSON.stringify(result));
})().catch(e => { console.error("HARNESS ERROR", e && e.stack || e); process.exit(1); });
