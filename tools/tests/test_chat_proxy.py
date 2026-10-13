"""Tests of the model list handling of tools/chat_proxy.py (Switcher.reload): a model added to the models file shows up without a restart, and a broken file leaves the working list."""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("chat_proxy", os.path.join(TOOLS, "chat_proxy.py"))
chat_proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_proxy)


def write(path, models, default):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"models": models, "default": default}, f)
    # the file system's time stamp may be too coarse for two writes in one test: move it forward
    t = time.time() + write.tick
    write.tick += 2
    os.utime(path, (t, t))


write.tick = 0


class ReloadTest(unittest.TestCase):
    def make(self, d):
        path = os.path.join(d, "chat_models.json")
        write(path, {"a": {"ckpt": "/m/a.bin", "note": "first"}}, "a")
        return path, chat_proxy.Switcher(path, "forge", "tok.json", "http://127.0.0.1:8190", "", "/m/a.bin")

    def test_a_new_model_appears_without_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path, sw = self.make(d)
            self.assertEqual([m["name"] for m in sw.listing()["models"]], ["a"])
            write(path, {"a": {"ckpt": "/m/a.bin", "note": "first"}, "b": {"ckpt": "/m/b.bin", "note": "second"}}, "a")
            listing = sw.listing()
            self.assertEqual([m["name"] for m in listing["models"]], ["a", "b"])
            self.assertEqual(listing["current"], "a")  # the running model stays
            self.assertIn("b", sw.models)  # select() would accept it

    def test_a_broken_or_empty_file_keeps_the_list_that_works(self):
        with tempfile.TemporaryDirectory() as d:
            path, sw = self.make(d)
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            os.utime(path, (time.time() + 10, time.time() + 10))
            self.assertEqual([m["name"] for m in sw.listing()["models"]], ["a"])
            write(path, {}, "a")
            self.assertEqual([m["name"] for m in sw.listing()["models"]], ["a"])
            os.remove(path)
            self.assertEqual([m["name"] for m in sw.listing()["models"]], ["a"])

    def test_an_unchanged_file_is_not_read_again(self):
        with tempfile.TemporaryDirectory() as d:
            path, sw = self.make(d)
            before = sw.mtime
            sw.reload()
            self.assertEqual(sw.mtime, before)


if __name__ == "__main__":
    unittest.main()


class Q8LaunchTest(unittest.TestCase):
    """The engine command of the Switcher: Q8 weights (--q8) only when the models file or the model's own entry asks for them, and a fallback to fp16 weights when the Q8 engine does not come up."""

    def make(self, d, top=None, entry=None):
        path = os.path.join(d, "chat_models.json")
        model = {"ckpt": "/m/a.bin", "note": "first"}
        model.update(entry or {})
        cfg = {"models": {"a": model}, "default": "a"}
        cfg.update(top or {})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return chat_proxy.Switcher(path, "/bin/forge", "/t/tok.json", "http://127.0.0.1:18999", "key")

    def test_default_is_fp16_and_the_file_wide_flag_turns_q8_on(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d)
            self.assertFalse(s.wants_q8("a"))
            self.assertNotIn("--q8", s.engine_command("a", s.wants_q8("a")))
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d, top={"q8": True})
            self.assertTrue(s.wants_q8("a"))
            cmd = s.engine_command("a", True)
            self.assertIn("--q8", cmd)
            self.assertLess(cmd.index("serve"), cmd.index("--q8"))
            self.assertEqual(cmd[cmd.index("--api-key") + 1], "key")

    def test_a_model_entry_overrides_the_file_wide_flag(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self.make(d, top={"q8": True}, entry={"q8": False}).wants_q8("a"))
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self.make(d, entry={"q8": True}).wants_q8("a"))

    def test_the_flag_is_read_again_when_the_file_changes(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d)
            self.assertFalse(s.wants_q8("a"))
            path = os.path.join(d, "chat_models.json")
            write(path, {"a": {"ckpt": "/m/a.bin", "note": "first"}}, "a")
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["q8"] = True
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            t = time.time() + 20
            os.utime(path, (t, t))
            s.reload()
            self.assertTrue(s.wants_q8("a"))

    def test_a_q8_engine_that_does_not_come_up_is_followed_by_an_fp16_one(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d, top={"q8": True})
            launched = []
            s.launch = lambda name, q8: (launched.append(q8), not q8)[1]  # the Q8 engine never answers, the fp16 one does
            real_run, real_sleep = chat_proxy.subprocess.run, chat_proxy.time.sleep
            chat_proxy.subprocess.run = lambda *a, **k: None
            chat_proxy.time.sleep = lambda *_: None
            try:
                s.current = "other"
                s.select("a")
            finally:
                chat_proxy.subprocess.run, chat_proxy.time.sleep = real_run, real_sleep
            self.assertEqual(launched, [True, False])
            self.assertEqual(s.current, "a")


class DownloadedModelTest(unittest.TestCase):
    """Downloaded models (GGUF files with their own tokenizer, chat format, sampling and draft head) next to our own checkpoints on the same page."""

    GGUF = {"ckpt": "/m/Qwen3.5-9B-Q4_K_M.gguf", "note": "pretrained", "page": {"title": "Qwen3.5-9B", "thinking": True, "repetition_penalty": 1.0},
            "engine": {"tokenizer": "/m/tok9.json", "chat": "qwen35", "ctx": 8192, "alias": "qwen3.5-9b", "temp": 0.7, "top_k": 20, "top_p": 0.8, "repeat_penalty": 1.0, "mtp": "/m/mtp.safetensors"}}

    def make(self, d, top=None):
        path = os.path.join(d, "chat_models.json")
        cfg = {"models": {"own": {"ckpt": "/m/a.bin", "note": "ours"}, "gguf": dict(self.GGUF)}, "default": "own"}
        cfg.update(top or {})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return chat_proxy.Switcher(path, "/bin/forge", "/t/tok.json", "http://127.0.0.1:18999", "key")

    def test_our_own_checkpoints_get_the_command_they_always_had(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d)
            want = ["nice", "-n", "10", "/bin/forge", "serve", "/m/a.bin", "--tokenizer", "/t/tok.json", "--host", "127.0.0.1", "--port", "18999", "--ctx", "1024", "--alias", "forge-1", "--chat", "chatml",
                    "--temp", "0.7", "--top-k", "40", "--top-p", "0.95", "--repeat-penalty", "1.1", "--api-key", "key"]
            self.assertEqual(s.engine_command("own"), want)
            self.assertEqual(s.engine_command("own", True), want[:-2] + ["--q8", "--api-key", "key"])

    def test_a_gguf_model_is_served_native_with_its_own_settings_and_never_q8(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d, top={"q8": True})
            self.assertTrue(s.native("gguf") and not s.native("own"))
            self.assertTrue(s.wants_q8("own") and not s.wants_q8("gguf"))
            cmd = s.engine_command("gguf", s.wants_q8("gguf"))
            self.assertIn("--native", cmd)
            self.assertNotIn("--q8", cmd)
            for flag, value in (("--tokenizer", "/m/tok9.json"), ("--ctx", "8192"), ("--alias", "qwen3.5-9b"), ("--chat", "qwen35"), ("--temp", "0.7"), ("--top-k", "20"), ("--top-p", "0.8"),
                                ("--repeat-penalty", "1.0"), ("--mtp", "/m/mtp.safetensors"), ("--api-key", "key")):
                self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)
            self.assertNotIn("--no-speculate", cmd)
            off = s.engine_command("gguf", False, False)
            self.assertIn("--no-speculate", off)
            self.assertNotIn("--mtp", off)

    def test_the_page_settings_travel_with_the_list(self):
        with tempfile.TemporaryDirectory() as d:
            listing = self.make(d).listing()
            by_name = {m["name"]: m for m in listing["models"]}
            self.assertEqual(by_name["own"]["page"], {})
            self.assertEqual(by_name["gguf"]["page"]["title"], "Qwen3.5-9B")
            self.assertTrue(by_name["gguf"]["page"]["thinking"])

    def test_a_draft_head_that_does_not_load_is_followed_by_an_engine_without_it(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d)
            launched = []
            s.launch = lambda name, q8, speculate=True: (launched.append((q8, speculate)), not speculate)[1]  # only the engine without the draft head answers
            real_run, real_sleep = chat_proxy.subprocess.run, chat_proxy.time.sleep
            chat_proxy.subprocess.run = lambda *a, **k: None
            chat_proxy.time.sleep = lambda *_: None
            try:
                s.select("gguf")
            finally:
                chat_proxy.subprocess.run, chat_proxy.time.sleep = real_run, real_sleep
            self.assertEqual(launched, [(False, True), (False, False)])
            self.assertEqual(s.current, "gguf")

    def test_a_model_that_does_not_come_up_is_followed_by_the_one_that_was_running(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.make(d)  # "own" is running
            launched = []
            s.launch = lambda name, q8, speculate=True: (launched.append((name, q8, speculate)), name == "own")[1]  # the downloaded model never comes up (no room on the GPU), ours does
            real_run, real_sleep = chat_proxy.subprocess.run, chat_proxy.time.sleep
            chat_proxy.subprocess.run = lambda *a, **k: None
            chat_proxy.time.sleep = lambda *_: None
            try:
                with self.assertRaises(RuntimeError):
                    s.select("gguf")
            finally:
                chat_proxy.subprocess.run, chat_proxy.time.sleep = real_run, real_sleep
            self.assertEqual(launched, [("gguf", False, True), ("gguf", False, False), ("own", False, True)])
            self.assertEqual(s.current, "own")  # the page is not left without an engine


class PageTest(unittest.TestCase):
    def script(self):
        a = chat_proxy.PAGE.index("<script>") + len("<script>")
        return chat_proxy.PAGE[a:chat_proxy.PAGE.index("</script>")]

    def test_the_page_script_has_no_escape_sequences_and_parses(self):
        js = self.script()
        self.assertNotIn(chr(92), js)  # a backslash inside the Python string would be turned into an escape before the browser sees it
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed: no syntax check of the page script")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "page.js")
            with open(path, "w", encoding="utf-8") as f:
                f.write(js)
            r = subprocess.run([node, "--check", path], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_ids_the_script_uses_exist_in_the_page(self):
        html = chat_proxy.PAGE[:chat_proxy.PAGE.index("<script>")]
        for i in ("brand", "sub", "model", "log", "inp", "send", "stop", "reset", "temp", "tv", "maxt", "mv", "think", "thinkl", "stat"):
            self.assertIn('id="%s"' % i, html, i)


class PageBehaviourTest(unittest.TestCase):
    """The script of the page against a fake page and scripted answers (tools/tests/chat_page_harness.js, needs node): the settings of the selected model, the thinking box, the speed line, the limits."""

    @classmethod
    def setUpClass(cls):
        node = shutil.which("node")
        if node is None:
            raise unittest.SkipTest("node is not installed")
        cls.tmp = tempfile.TemporaryDirectory()
        a = chat_proxy.PAGE.index("<script>") + len("<script>")
        script = os.path.join(cls.tmp.name, "page.js")
        with open(script, "w", encoding="utf-8") as f:
            f.write(chat_proxy.PAGE[a:chat_proxy.PAGE.index("</script>")])
        r = subprocess.run([node, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_page_harness.js"), script], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise AssertionError(r.stderr)
        cls.out = json.loads(r.stdout)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_our_own_model_gets_the_page_it_always_had(self):
        o = self.out["forge"]
        self.assertEqual((o["brand"], o["placeholder"], o["thinkVisible"], o["maxMax"], o["maxValue"], o["temp"]), ("Forge-1", "Message Forge-1", False, "600", "400", "0.5"))
        self.assertTrue(o["selectVisible"])
        self.assertEqual(len(o["options"]), 2)
        self.assertTrue(o["intro"].startswith("A 150M-parameter language model"))
        self.assertEqual(o["chips"], 5)
        self.assertEqual(o["alias"], "alias-x")
        req = o["request"]
        self.assertEqual((req["model"], req["stream"], req["repetition_penalty"]), ("alias-x", True, 1.1))
        self.assertTrue(req["stream_options"]["include_usage"])
        self.assertNotIn("chat_template_kwargs", req)
        self.assertEqual(req["messages"][0], {"role": "system", "content": "You are a helpful assistant."})
        m = o["messages"]
        self.assertEqual([x["cls"] for x in m], ["msg user", "msg assistant"])
        self.assertEqual((m[1]["label"], m[1]["answer"], m[1]["thinkShown"]), ("Forge-1", "Paris.", False))
        self.assertIn("2 tokens at 765 tok/s", m[1]["stats"])
        self.assertIn("prompt 42 tokens (30 reused)", m[1]["stats"])
        self.assertTrue(o["introGone"])
        self.assertEqual(json.loads(o["history"]), [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}])

    def test_a_downloaded_model_brings_its_own_settings_and_its_thinking_box(self):
        o = self.out["qwenThinking"]
        self.assertEqual((o["brand"], o["title"], o["placeholder"], o["thinkVisible"], o["thinkChecked"], o["maxMax"], o["maxValue"], o["temp"], o["intro"]),
                         ("Qwen3.5-9B", "Qwen3.5-9B chat", "Message Qwen3.5-9B", True, False, "4096", "1024", "0.7", "Qwen intro text"))
        req = o["request"]
        self.assertEqual((req["repetition_penalty"], req["max_tokens"], req["temperature"], req["chat_template_kwargs"]), (1.0, 1024, 0.7, {"enable_thinking": True}))
        a = o["messages"][1]
        self.assertEqual((a["label"], a["think"], a["answer"], a["thinkShown"], a["thinkOpen"]), ("Qwen3.5-9B", "17 times 20 is 340, plus 51 is 391.", "391", True, False))  # collapsed once the answer starts
        self.assertIn("30 tokens at 112 tok/s", a["stats"])
        self.assertIn("draft head right 90 percent", a["stats"])
        self.assertIn("prompt 20 tokens", a["stats"])
        self.assertEqual(o["stat"], "112 tok/s")

    def test_thinking_that_uses_up_the_limit_is_reported_and_not_kept_in_the_history(self):
        o = self.out["qwenNoAnswer"]
        a = o["messages"][1]
        self.assertIn("no answer yet", a["answer"])
        self.assertTrue(a["thinkShown"])
        self.assertIn("stopped at the token limit", a["stats"])
        self.assertEqual(json.loads(o["history"]), [])

    def test_an_engine_error_is_shown_and_the_question_is_dropped(self):
        o = self.out["engineError"]
        self.assertEqual(o["messages"][1]["answer"], "error 500 boom")
        self.assertEqual(json.loads(o["history"]), [])

    def test_markdown_in_an_answer_is_built_from_nodes_and_other_markup_stays_text(self):
        a = self.out["markdown"]["messages"][1]
        tags = a["tags"]
        self.assertEqual((tags["strong"], tags["em"], tags["code"], tags["ul"], tags["ol"], tags["li"], tags["pre"], tags["a"], tags[".mdh"], tags[".mdq"]), (1, 1, 2, 1, 1, 4, 1, 1, 1, 1))
        self.assertEqual((a["strong"], a["em"], a["code"]), (["bold"], ["italic"], ["code", "print(1)"]))
        self.assertEqual(a["links"], [{"href": "https://example.com", "rel": "noopener noreferrer", "text": "link"}])  # the javascript: link is no link
        self.assertIn("[bad](javascript:alert(1))", a["answer"])
        self.assertIn("<b>tag</b>", a["answer"])  # tags are text, never HTML
        self.assertNotIn("b", tags)
        self.assertIn("2*3*4 and a * b.", a["answer"])  # a product is no emphasis
        self.assertNotIn("**", a["answer"])
        self.assertIn("Title", a["answer"])
        self.assertEqual(self.out["markdown"]["stat"].endswith("tok/s"), True)

    def test_one_model_means_no_selector(self):
        self.assertFalse(self.out["single"]["selectVisible"])
