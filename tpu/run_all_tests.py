"""Runs every test of the Python side (tpu/, cloud/, tools/tests/) in parallel worker processes and prints one summary: the dev loop of a merge (it took about 25 minutes one test file after the other).

    python run_all_tests.py [--jobs N] [--log-dir DIR] [--only TEXT[,TEXT...]] [--skip TEXT[,TEXT...]] [--list]

The jobs are one per check of tpu/test_forge_jax.py (the list of tpu/run_checks.py, which runs one check per process with --only), one per check of every other tpu/test_*.py whose `__main__` block calls check_*()
functions one after the other (name FILE::check_x; a file built differently, with unittest.main() or a main(), is one job), cloud/test_autotune.py and the unittest files of tools/tests (one job). Every job runs in its own
process with its output in LOG_DIR/<job>.log, so that a failure shows its own lines and nothing is mixed. The jobs that take longest start first. Exit code 0 only when every job passed (a process that dies at exit, for
example with SIGABRT, is a failure even when its checks printed ok).

--only / --skip take texts that must occur in the job name (--only check_variant: the eight variant checks). Needs only the standard library: run it with the interpreter that has jax (on the homelab ~/forgeinfer/tpuc/bin/python).
Use it where the machine is free: every job is a JAX CPU compile, so --jobs 4 uses 4 cores for several minutes. Default --jobs: half the cores, at most 4.
"""
import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# the jobs that took longest in the measured serial run start first (a pool finishes sooner when the long jobs are not the last ones); names that are not in the list keep their order
HEAVY_FIRST = ("test_layers2.py", "check_bench_step", "test_async_loop.py::check_guards_with_lag", "check_remat_acts", "test_opt_tune.py::check_shard", "test_async_loop.py::check_lag_changes_nothing",
               "test_premortem.py::check_resume_continuity", "test_async_loop.py::check_async_io", "test_async_loop.py::check_time_limit_with_pipeline", "test_async_loop.py::check_time_plan_with_pipeline",
               "check_safety_nets", "check_auto_steps")  # measured 2026-10-12 on the homelab with 4 jobs at a time (176 s down to 41 s)


def literal_tuple(path, name):
    """The tuple literal `name = (...)` of a module, read without importing it (run_checks imports jax)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise KeyError(f"{name} not found in {path}")


def main_calls(path):
    """The functions that the `if __name__ == "__main__":` block of a test file calls one after the other (bare calls of check_*() without arguments), read without importing the file; [] when the file is built
    differently (unittest.main(), a main() ...) and has to run as one job."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
            return [st.value.func.id for st in node.body
                    if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name) and not st.value.args and not st.value.keywords and st.value.func.id.startswith("check_")]
    return []


def job_list():
    """[(name, argument list, working directory)] in the order of the files."""
    py = sys.executable
    runner = os.path.join(HERE, "run_checks.py")
    jobs = []
    for v in literal_tuple(runner, "VARIANTS"):
        jobs.append((f"check_variant({v})", [py, runner, "--only", f"check_variant({v})"], HERE))
    for c in literal_tuple(runner, "CHECKS"):
        jobs.append((c, [py, runner, "--only", c], HERE))
    for f in sorted(os.listdir(HERE)):
        if re.fullmatch(r"test_.*\.py", f) and f != "test_forge_jax.py":
            calls = main_calls(os.path.join(HERE, f))
            if len(calls) >= 2:  # a file of independent checks: one job per check (a fresh process each, as when the check is run alone)
                jobs.extend((f"{f}::{c}", [py, "-c", f"import {f[:-3]} as m; m.{c}()"], HERE) for c in calls)
            else:
                jobs.append((f, [py, os.path.join(HERE, f)], HERE))
    cloud = os.path.join(ROOT, "cloud")
    if os.path.exists(os.path.join(cloud, "test_autotune.py")):
        jobs.append(("cloud/test_autotune.py", [py, os.path.join(cloud, "test_autotune.py")], cloud))
    tests = os.path.join(ROOT, "tools", "tests")
    if os.path.isdir(tests):
        jobs.append(("tools/tests (unittest)", [py, "-m", "unittest", "discover", "-s", tests, "-p", "test_*.py"], ROOT))
    return jobs


def order_heavy_first(jobs):
    rank = {n: i for i, n in enumerate(HEAVY_FIRST)}
    return sorted(jobs, key=lambda j: rank.get(j[0], len(rank)))  # sorted() is stable: the others keep their order


def last_lines(path, n=4):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [ln.rstrip() for ln in f.read().splitlines() if ln.strip()][-n:]
    except OSError:
        return []


def run_pool(jobs, njobs, log_dir):
    """Runs the jobs with at most njobs processes at a time; returns {name: (status, seconds, exit code, log path)}."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    pending = list(jobs)
    running = []
    results = {}
    while pending or running:
        while pending and len(running) < njobs:
            name, cmd, cwd = pending.pop(0)
            log = os.path.join(log_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".log")
            fh = open(log, "w", encoding="utf-8")
            running.append((name, subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT), time.time(), log, fh))
        time.sleep(0.3)
        for item in list(running):
            name, proc, t0, log, fh = item
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            running.remove(item)
            results[name] = ("PASS" if rc == 0 else "FAIL", time.time() - t0, rc, log)
            print(f"{results[name][0]:4s} {results[name][1]:7.1f} s  {name}" + ("" if rc == 0 else f"   exit code {rc}, log {log}"), flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    only = [x for x in a.only.split(",") if x]
    skip = [x for x in a.skip.split(",") if x]
    jobs = [j for j in job_list() if (not only or any(o in j[0] for o in only)) and not any(s in j[0] for s in skip)]
    if not jobs:
        print("no job selected")
        return 2
    if a.list:
        for name, cmd, cwd in jobs:
            print(name)
        print(f"{len(jobs)} jobs")
        return 0
    log_dir = a.log_dir or tempfile.mkdtemp(prefix="forge_tests_")
    os.makedirs(log_dir, exist_ok=True)
    print(f"{len(jobs)} jobs, {a.jobs} at a time, logs in {log_dir}", flush=True)
    t0 = time.time()
    results = run_pool(order_heavy_first(jobs), max(1, a.jobs), log_dir)
    print("\n=== summary (in the order of the files) ===")
    failed = []
    for name, cmd, cwd in jobs:
        status, secs, rc, log = results[name]
        print(f"{status:4s} {secs:7.1f} s  {name}")
        if status != "PASS":
            failed.append(name)
            for ln in last_lines(log):
                print(f"       | {ln[:220]}")
    print(f"{len(jobs) - len(failed)} passed, {len(failed)} failed, {time.time() - t0:.0f} s wall clock (the jobs add up to {sum(r[1] for r in results.values()):.0f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
