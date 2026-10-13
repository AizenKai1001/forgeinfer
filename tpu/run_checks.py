"""Runs every check of tpu/test_forge_jax.py one after the other and reports each, instead of stopping at the first failure (test_forge_jax.py itself stops at the first failed assertion).

    python run_checks.py [--skip check_safety_nets,...] [--only check_fuse_qkv,...]

Use it where one check is known to fail for an environment reason (on the homelab's jax[tpu] virtualenv the no-op resume run of check_safety_nets ends with SIGABRT at interpreter exit, with the old trainer too) and the
others must still be seen. Exit code 0 only when every check that ran passed. The list below follows the `__main__` block of test_forge_jax.py (keep the two in step when a check is added).
"""
import argparse
import sys
import time
import traceback

import test_forge_jax as t

VARIANTS = ("plain", "qknorm", "untied", "relu2", "softcap", "lambdas", "skips", "valemb")
CHECKS = ("check_muon", "check_stacked_muon", "check_muon_shard", "check_bench_step", "check_kaggle_bench_runner", "check_checkpoint_roundtrip", "check_window_checkpoint", "check_window_attention",
          "check_window_layer_rule", "check_window_off_is_identical", "check_window_model", "check_training_runs", "check_plan_steps", "check_plan_with_stalls", "check_end_to_end", "check_auto_steps",
          "check_attention_fallback", "check_window_end_to_end", "check_pin_wanted", "check_safety_nets", "check_splash_attention", "check_remat_policy", "check_remat_acts", "check_fuse_qkv", "check_accum1_step")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    skip = {x for x in a.skip.split(",") if x}
    only = [x for x in a.only.split(",") if x]
    jobs = [("check_variant(%s)" % v, (lambda v=v: t.check_variant(v))) for v in VARIANTS] + [(c, getattr(t, c)) for c in CHECKS]
    results = []
    for name, fn in jobs:
        if name in skip or (only and not any(name == o or name.startswith(o + "(") for o in only)):
            continue
        t0 = time.time()
        try:
            fn()
            results.append((name, "PASS", time.time() - t0, ""))
        except BaseException as e:  # noqa: BLE001 (every check is reported, whatever it raises)
            results.append((name, "FAIL", time.time() - t0, "%s: %s" % (type(e).__name__, str(e)[:300])))
            traceback.print_exc()
    print("\n=== summary ===")
    for name, status, secs, msg in results:
        print(f"{status:4s} {secs:7.1f} s  {name}" + (f"   {msg}" if msg else ""))
    failed = [r for r in results if r[1] != "PASS"]
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
