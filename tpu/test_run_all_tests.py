"""Tests of tpu/run_all_tests.py (standard library only): the job list follows run_checks.py and the test files, the pool reports passes, failures and a process that dies after it printed ok, runs jobs side by side, and
--only / --skip / --list select by name."""
import contextlib
import io
import os
import sys
import tempfile
import time

import run_all_tests as r

PY = sys.executable


def check_job_list():
    jobs = r.job_list()
    names = [j[0] for j in jobs]
    assert len(names) == len(set(names)), "a job name is used twice"
    for v in r.literal_tuple(os.path.join(r.HERE, "run_checks.py"), "VARIANTS"):
        assert f"check_variant({v})" in names, v
    for c in r.literal_tuple(os.path.join(r.HERE, "run_checks.py"), "CHECKS"):
        assert c in names, c
    assert "test_forge_jax.py" not in names  # its checks are jobs of their own
    assert "test_run_all_tests.py::check_job_list" in names and "test_async_loop.py::check_scalars" in names and "test_hlo_cost.py" in names  # test_hlo_cost.py runs unittest.main(): one job
    assert any(n.startswith("cloud/") for n in names) or not os.path.exists(os.path.join(r.ROOT, "cloud", "test_autotune.py"))
    for name, cmd, cwd in jobs:
        assert os.path.isdir(cwd), (name, cwd)
    print(f"job list: {len(names)} jobs, one per check of test_forge_jax.py, one per other test file: ok")


def check_main_calls():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        with open(p, "w") as f:
            f.write('import sys\n\ndef check_a():\n    pass\n\nif __name__ == "__main__":\n    if len(sys.argv) > 1:\n        sys.exit(0)\n    check_a()\n    check_b()\n    helper()\n    check_c(1)\n    print("done")\n')
        assert r.main_calls(p) == ["check_a", "check_b"]  # bare check_*() calls only: not helper(), not a call with arguments, not the if block
        with open(p, "w") as f:
            f.write('import unittest\nif __name__ == "__main__":\n    unittest.main()\n')
        assert r.main_calls(p) == []
        with open(p, "w") as f:
            f.write("x = 1\n")
        assert r.main_calls(p) == []
    print("main block of a test file: only the bare check_*() calls count: ok")


def check_pool():
    with tempfile.TemporaryDirectory() as d:
        jobs = [("fine", [PY, "-c", "print('fine')"], d),
                ("fails", [PY, "-c", "import sys; print('boom'); sys.exit(3)"], d),
                ("dies at exit", [PY, "-c", "import os; print('PASS looking line'); os._exit(7)"], d)]
        with contextlib.redirect_stdout(io.StringIO()):
            res = r.run_pool(jobs, 2, d)
        assert res["fine"][0] == "PASS" and res["fine"][2] == 0
        assert res["fails"][0] == "FAIL" and res["fails"][2] == 3 and "boom" in open(res["fails"][3]).read()
        assert res["dies at exit"][0] == "FAIL" and res["dies at exit"][2] == 7  # printed ok, died at exit: still a failure
        # side by side: two one-second jobs with two workers take about one second, with one worker two
        sleepers = [(f"sleep{i}", [PY, "-c", "import time; time.sleep(1.0)"], d) for i in range(2)]
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r.run_pool(sleepers, 2, d)
        wide = time.time() - t0
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r.run_pool(sleepers, 1, d)
        narrow = time.time() - t0
        assert narrow >= 2.0 and wide < narrow - 0.5, (wide, narrow)
    print(f"pool: pass, failure, death at exit, side by side ({wide:.1f} s with two workers, {narrow:.1f} s with one): ok")


def check_selection():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = r.main(["--list", "--only", "check_variant"])
    lines = buf.getvalue().splitlines()
    assert rc == 0 and lines[-1].endswith("jobs") and sum(1 for ln in lines if ln.startswith("check_variant(")) == len(r.literal_tuple(os.path.join(r.HERE, "run_checks.py"), "VARIANTS")), lines
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r.main(["--list", "--skip", "check_variant,check_safety_nets"])
    assert "check_safety_nets" not in buf.getvalue() and "check_variant(" not in buf.getvalue() and "check_muon" in buf.getvalue()
    with contextlib.redirect_stdout(io.StringIO()):
        assert r.main(["--list", "--only", "no such job"]) == 2
    print("--only / --skip / --list: ok")


if __name__ == "__main__":
    check_job_list()
    check_main_calls()
    check_pool()
    check_selection()
    print("run_all_tests: all ok")
