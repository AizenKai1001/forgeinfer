"""Tests of the in-session autotune of cloud/kaggle_tpu_train.py: the decision (choose_config), the plumbing around a benchmark process (run_autotune with a stand-in bench_step.py: which files are used, what is passed,
what comes back when something is missing or fails) and, when jax is installed, a CPU run of the real tpu/bench_step.py on a tiny model with 8 fake devices.

    <repo>/tpu/venv/Scripts/python.exe cloud/test_autotune.py      (the last test needs jax; it takes a minute or two)
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_TMP = tempfile.mkdtemp(prefix="autotune_test_")
os.environ["FORGE_KAGGLE_IN"] = str(Path(_TMP) / "in")
os.environ["FORGE_KAGGLE_OUT"] = str(Path(_TMP) / "out")
Path(os.environ["FORGE_KAGGLE_OUT"]).mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("kaggle_tpu_train", HERE / "kaggle_tpu_train.py")
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)


def row(name, tps, first=10.4, final=10.39, status="ok", flags="", env=None, step=65536):
    r = {"variant": name, "status": status, "extra_flags": flags, "env": env or {}}
    if status == "ok":
        r.update(tokens_per_s=tps, first_loss=first, final_loss=final, tokens_per_step=step)
    return r


BASE = row("base", 170000.0)


class ChooseTest(unittest.TestCase):
    def test_fastest_valid_candidate_wins(self):
        d = K.choose_config([BASE, row("muon-shard", 205000.0, flags="--muon-shard"), row("ms+attn", 212000.0, flags="--muon-shard --remat-policy dots+attn", env={"A": "b"}), row("slow", 169000.0, flags="--x")])
        self.assertEqual(d["name"], "ms+attn")
        self.assertEqual(d["flags"], "--muon-shard --remat-policy dots+attn")
        self.assertEqual(d["env"], {"A": "b"})
        self.assertAlmostEqual(d["gain_pct"], 24.7, places=1)

    def test_a_candidate_with_another_function_is_rejected_however_fast(self):
        d = K.choose_config([BASE, row("wrong-first", 250000.0, first=10.5, flags="--a"), row("wrong-final", 250000.0, final=10.5, flags="--b"), row("other-batch", 250000.0, step=131072, flags="--c"),
                             row("ok", 180000.0, first=10.4004, final=10.391, flags="--d")])
        self.assertEqual(d["name"], "ok")
        self.assertIn("wrong-first", d["reason"])
        self.assertIn("other-batch", d["reason"])

    def test_nothing_faster_or_nothing_valid_keeps_the_settings(self):
        for rows, text in (([BASE, row("a", 171000.0, flags="--a")], "only"), ([BASE, row("a", 0, status="oom"), row("b", 0, status="error")], "oom"), ([BASE], "no candidate")):
            d = K.choose_config(rows)
            self.assertEqual((d["name"], d["flags"], d["env"]), ("base", "", {}))
            self.assertIn(text, d["reason"])

    def test_without_a_measured_base_there_is_no_decision(self):
        d = K.choose_config([row("base", 0, status="oom"), row("a", 250000.0, flags="--a")])
        self.assertEqual(d["name"], "base")
        self.assertIn("not measured", d["reason"])

    def test_the_fewer_flags_win_a_tie(self):
        d = K.choose_config([BASE, row("long", 200000.0, flags="--muon-shard --remat-policy dots+attn"), row("short", 200000.0, flags="--muon-shard")])
        self.assertEqual(d["name"], "short")

    def test_threshold_is_a_parameter(self):
        rows = [BASE, row("a", 172000.0, flags="--a")]
        self.assertEqual(K.choose_config(rows)["name"], "base")
        self.assertEqual(K.choose_config(rows, min_gain_pct=0.5)["name"], "a")


FAKE_BENCH = '''import json, sys, os
a = sys.argv
out = a[a.index("--out") + 1]
vj = json.load(open(a[a.index("--variants-json") + 1]))
args = [x for x in a if x.startswith("--args=")]
rows = [{"variant": v["name"], "status": "ok", "extra_flags": v.get("flags", ""), "env": v.get("env", {}), "tokens_per_s": 100000.0 * (1.3 if "muon" in v["name"] else 1.0),
         "first_loss": 10.4, "final_loss": 10.39, "tokens_per_step": 65536} for v in vj]
json.dump({"table": rows, "seen_args": args, "seen_argv": a}, open(os.path.join(out, "bench.json"), "w"))
'''


FAKE_BENCH_PICK = FAKE_BENCH.replace('"first_loss": 10.4,', '"attn_pick": ("512,1" if v["name"] == "base" else "256,0"), "first_loss": 10.4,')  # base chose 512,1 (the others carry their own pick, which must not matter)


class AttnPickTest(unittest.TestCase):
    def test_the_pick_of_the_base_row_becomes_a_trainer_flag(self):
        self.assertEqual(K.attn_pick_flag([dict(BASE, attn_pick="512,1")], ["--layers", "24"]), "--attn-pick 512,1")
        self.assertEqual(K.attn_pick_flag([dict(BASE, attn_pick="1024,0")], []), "--attn-pick 1024,0")

    def test_no_flag_when_there_is_no_usable_pick(self):
        self.assertEqual(K.attn_pick_flag([BASE], []), "")  # the row has no pick (an older bench_step.py)
        self.assertEqual(K.attn_pick_flag([dict(BASE, attn_pick="xla")], []), "")  # the benchmark fell back to XLA's attention: the trainer selects for itself
        self.assertEqual(K.attn_pick_flag([row("base", 0, status="oom")], []), "")  # base failed
        self.assertEqual(K.attn_pick_flag([row("a", 1.0, flags="--a")], []), "")  # no base row
        for bad in ("512", "512,2", "512,1;ls", "a,b", "", None, "512,1 --x"):
            self.assertEqual(K.attn_pick_flag([dict(BASE, attn_pick=bad)], []), "", bad)

    def test_the_picks_of_other_rows_do_not_count(self):
        self.assertEqual(K.attn_pick_flag([dict(BASE, attn_pick="512,1"), dict(row("x", 2.0, flags="--x"), attn_pick="256,0")], []), "--attn-pick 512,1")
        self.assertEqual(K.attn_pick_flag([BASE, dict(row("x", 2.0, flags="--x"), attn_pick="256,0")], []), "")

    def test_the_settings_choice_of_the_attention_wins(self):
        rows = [dict(BASE, attn_pick="512,1")]
        self.assertEqual(K.attn_pick_flag(rows, ["--attn", "xla"]), "")
        self.assertEqual(K.attn_pick_flag(rows, ["--attn", "splash"]), "")
        self.assertEqual(K.attn_pick_flag(rows, ["--attn-pick", "256,1"]), "")


class RunTest(unittest.TestCase):
    def folder(self, root, name, trainer=b"print('trainer')\n", with_bench=True, bench=None):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "forge_jax.py").write_bytes(trainer)
        if with_bench:
            (d / "bench_step.py").write_text(bench or FAKE_BENCH)
        return d

    def test_the_attention_pick_of_the_base_row_goes_to_the_trainer(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.folder(root, "forge-jax", bench=FAKE_BENCH_PICK)
            K.OUT = Path(root) / "out"
            K.OUT.mkdir()
            flags, env, note = K.run_autotune({"candidates": [{"name": "muon-shard", "flags": "--muon-shard"}]}, d / "forge_jax.py", ["--layers", "24"], 4)
            self.assertEqual((flags, env, note), ("--muon-shard --attn-pick 512,1", {}, "muon-shard"))
            # no winner (nothing is faster than base): the pick still goes to the trainer, which then has no other new flag
            flags, env, note = K.run_autotune({"candidates": [{"name": "slow", "flags": "--q"}]}, d / "forge_jax.py", ["--layers", "24"], 4)
            self.assertEqual((flags, note), ("--attn-pick 512,1", "base"))
            # the settings choose the attention themselves: no pick
            flags, env, note = K.run_autotune({"candidates": [{"name": "muon-shard", "flags": "--muon-shard"}]}, d / "forge_jax.py", ["--layers", "24", "--attn", "splash"], 4)
            self.assertEqual(flags, "--muon-shard")

    def test_uses_the_folder_of_the_trainer_and_adds_the_flags_of_the_winner(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.folder(root, "forge-jax")
            K.OUT = Path(root) / "out"
            K.OUT.mkdir()
            flags, env, note = K.run_autotune({"candidates": [{"name": "muon-shard", "flags": "--muon-shard"}, {"name": "tiny-gain", "flags": "--q", "env": {}}]}, d / "forge_jax.py", ["--layers", "24", "--lr", "1e-3"], 4)
            self.assertEqual((flags, env, note), ("--muon-shard", {}, "muon-shard"))
            seen = json.loads((K.OUT / "autotune" / "bench.json").read_text())
            self.assertIn("--args=--layers 24 --lr 1e-3 --micro 4", seen["seen_args"])
            self.assertTrue((K.OUT / "autotune" / "decision.json").exists())

    def test_a_bench_next_to_another_trainer_is_not_used(self):
        with tempfile.TemporaryDirectory() as root:
            real = self.folder(root, "forge-jax", with_bench=False)
            self.folder(root, "forge-jax-code", trainer=b"print('another version')\n")
            os.environ["FORGE_KAGGLE_IN"] = root
            K.IN = Path(root)
            K.OUT = Path(root) / "out"
            K.OUT.mkdir()
            flags, env, note = K.run_autotune({}, real / "forge_jax.py", [], 4)
            self.assertEqual((flags, env), ("", {}))
            self.assertIn("skipped", note)

    def test_the_code_dataset_is_used_when_it_holds_the_same_trainer(self):
        with tempfile.TemporaryDirectory() as root:
            real = self.folder(root, "forge-jax", with_bench=False)
            self.folder(root, "forge-jax-code")
            K.IN = Path(root)
            K.OUT = Path(root) / "out"
            K.OUT.mkdir()
            flags, env, note = K.run_autotune({}, real / "forge_jax.py", ["--layers", "2"], 2)
            self.assertEqual(flags, "--muon-shard")  # the default candidate list
            self.assertEqual(note, "muon-shard")

    def test_a_missing_table_leaves_the_settings(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.folder(root, "forge-jax")
            (d / "bench_step.py").write_text("import sys\nsys.exit(3)\n")
            K.OUT = Path(root) / "out"
            K.OUT.mkdir()
            flags, env, note = K.run_autotune({}, d / "forge_jax.py", [], 4)
            self.assertEqual((flags, env), ("", {}))
            self.assertIn("no bench.json", note)


class TrainerEnvTest(unittest.TestCase):
    """main() of the runner with a stand-in trainer that reports its environment: FORGE_NO_PIN (no second jax probe in the trainer) is set when the runner has settled the jax version, and not after a pin that failed here."""

    def run_main(self, backend, pinned, settings_extra=None):
        with tempfile.TemporaryDirectory() as root:
            inp, outp = Path(root) / "in", Path(root) / "out"
            for name in ("forge-jax", "forge-tpujob", "forge-tokens"):
                (inp / name).mkdir(parents=True)
            outp.mkdir()
            (inp / "forge-jax" / "forge_jax.py").write_text("import os, sys\nprint('TRAINER_ENV no_pin=' + os.environ.get('FORGE_NO_PIN', '-') + ' marker=' + os.environ.get('MARK', '-') + ' argv=' + ' '.join(sys.argv[1:]))\n")
            settings = {"args": "--layers 2", "data": "forge-tokens", "micro": [2], "require_accelerator": False, "session_s": 4000, "end_margin_s": 60}
            settings.update(settings_extra or {})
            (inp / "forge-tpujob" / "settings.json").write_text(json.dumps(settings))
            saved = (K.IN, K.OUT, K.backend, K.pinned_jax, K.sh)
            K.IN, K.OUT = inp, outp
            K.backend = lambda env=None: backend
            K.pinned_jax = lambda version: pinned
            K.sh = lambda cmd, timeout=600, env=None: ""
            try:
                K.main()
            finally:
                K.IN, K.OUT, K.backend, K.pinned_jax, K.sh = saved
            return (outp / "train.log").read_text()

    def test_the_jax_version_is_settled(self):
        log = self.run_main("tpu 8 TPU v5 lite (jax 0.11.0)", dict(os.environ, MARK="pinned"))  # the pin was installed and works
        self.assertIn("no_pin=1 marker=pinned", log)
        log = self.run_main("tpu 8 TPU v5 lite (jax 0.11.2)", None)  # already the right version
        self.assertIn("no_pin=1", log)
        log = self.run_main("tpu 8 TPU v5 lite (jax 0.11.0)", None, {"jax_pin": ""})  # the settings want the preinstalled jax
        self.assertIn("no_pin=1", log)
        log = self.run_main("gpu 2 fake (jax 0.4)", None)  # no TPU: the trainer pins only on a TPU
        self.assertIn("no_pin=1", log)

    def test_a_failed_pin_leaves_the_trainers_own_second_try(self):
        log = self.run_main("tpu 8 TPU v5 lite (jax 0.11.0)", None)
        self.assertIn("no_pin=- ", log)


def have_jax():
    return importlib.util.find_spec("jax") is not None


@unittest.skipUnless(have_jax(), "jax is not installed in this Python")
class RealBenchOnCpuTest(unittest.TestCase):
    def test_tiny_model_on_fake_devices(self):
        with tempfile.TemporaryDirectory() as root:
            tpu = ROOT / "tpu"
            d = Path(root) / "forge-jax"
            d.mkdir()
            for n in ("forge_jax.py", "bench_step.py"):
                (d / n).write_bytes((tpu / n).read_bytes())
            K.OUT = Path(root) / "out"
            K.OUT.mkdir()
            cands = [{"name": "muon-shard", "flags": "--muon-shard"}, {"name": "fuse-qkv", "flags": "--fuse-qkv"}]
            flags, env, note = K.run_autotune({"tiny": True, "candidates": cands, "budget_s": 400, "per_variant_s": 150, "steps": 3, "min_gain_pct": -1000.0}, d / "forge_jax.py", [], 4)
            rows = json.loads((K.OUT / "autotune" / "bench.json").read_text())["table"]
            by = {r["variant"]: r for r in rows}
            self.assertEqual(by["base"]["status"], "ok", by["base"])
            self.assertIn("first_loss", by["base"])
            for n in ("muon-shard", "fuse-qkv"):
                self.assertEqual(by[n]["status"], "ok", by[n])
                # the same function: the first-step loss agrees with base to float noise (the check the autotune relies on)
                self.assertAlmostEqual(by[n]["first_loss"], by["base"]["first_loss"], delta=2e-3, msg=n)
            self.assertIn(note, ("base", "muon-shard", "fuse-qkv"))


@unittest.skipUnless(have_jax(), "jax is not installed in this Python")
class SettingsTest(unittest.TestCase):
    def parse(self, args):
        import shlex

        sys.path.insert(0, str(ROOT / "tpu"))
        import forge_jax as fj

        return fj.parse_args(["train", "--data", "d", "--out", "o"] + shlex.split(args))

    def test_every_settings_file_parses_with_the_trainers_arguments(self):
        names = sorted((HERE / "tpu_settings").glob("*.json"))
        self.assertGreaterEqual(len(names), 4)
        for p in names:
            settings = json.loads(p.read_text())
            all_args = [settings["args"]] if "args" in settings else [r["args"] for r in settings["runs"]]  # the SFT grid keeps its arguments per run
            for args in all_args:
                self.assertGreater(self.parse(args).steps, 0, p.name)
                for cand in (settings.get("autotune") or {}).get("candidates", []):
                    self.parse(args + " " + cand.get("flags", ""))

    def test_the_365m_settings_use_the_new_host_side_options(self):
        for name in ("m365_ve8.json", "m365_ve8_auto.json"):
            a = self.parse(json.loads((HERE / "tpu_settings" / name).read_text())["args"])
            self.assertEqual((a.eval_every, a.eval_early_until), (250, 1000), name)
        a = self.parse(json.loads((HERE / "tpu_settings" / "m365_ve8_auto.json").read_text())["args"])
        self.assertEqual((a.async_steps, a.async_io), (2, True))


if __name__ == "__main__":
    unittest.main()
