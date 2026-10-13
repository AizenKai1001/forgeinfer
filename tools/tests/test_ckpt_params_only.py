"""Tests of tools/ckpt_params_only.py on a tiny fake forgeckpt1 file."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TOOLS, "ckpt_params_only.py")
N = 10


def make_ckpt(path, n=N, magic="forgeckpt1", moments=True):
    header = (json.dumps({"magic": magic, "n_params": n, "step": 7}) + "\n").encode()
    params = struct.pack(f"<{n}f", *[float(i) + 0.5 for i in range(n)])
    adam = struct.pack(f"<{2 * n}f", *[-float(i) for i in range(2 * n)]) if moments else b""
    with open(path, "wb") as f:
        f.write(header + params + adam)
    return header, params


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


class CkptParamsOnlyTest(unittest.TestCase):
    def test_cut_keeps_header_and_parameter_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            header, params = make_ckpt(src)
            r = run(src, dst, "--check")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("check ok", r.stdout)
            self.assertEqual(os.path.getsize(dst), len(header) + 4 * N)
            self.assertEqual(os.path.getsize(src), len(header) + 12 * N)
            with open(dst, "rb") as f:
                data = f.read()
            self.assertEqual(data[: len(header)], header)
            self.assertEqual(data[len(header):], params)
            with open(src, "rb") as f:
                self.assertEqual(data, f.read()[: len(data)])
            self.assertFalse(os.path.exists(dst + ".part"))

    def test_input_without_moments_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            header, _ = make_ckpt(src, moments=False)
            r = run(src, dst)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("absent or partial", r.stdout)
            self.assertEqual(os.path.getsize(dst), len(header) + 4 * N)

    def test_wrong_magic_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            make_ckpt(src, magic="other")
            r = run(src, dst)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not a forgeckpt1", r.stderr)
            self.assertFalse(os.path.exists(dst))

    def test_truncated_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            make_ckpt(src)
            with open(src, "r+b") as f:
                f.truncate(os.path.getsize(src) - 12 * N + 8)  # cuts into the parameters
            r = run(src, dst)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(os.path.exists(dst))

    def test_verify_accepts_complete_files_and_rejects_broken_ones(self):
        with tempfile.TemporaryDirectory() as d:
            full, cut = os.path.join(d, "full.bin"), os.path.join(d, "cut.bin")
            header, _ = make_ckpt(full)
            r = run(full, "--verify")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("with both moments", r.stdout)
            self.assertEqual(run(full, cut).returncode, 0)
            r = run(cut, "--verify")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("weights only", r.stdout)
            with open(full, "r+b") as f:  # a download that broke off inside the moments
                f.truncate(len(header) + 12 * N - 4)
            r = run(full, "--verify")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("INCOMPLETE", r.stderr)
            with open(full, "r+b") as f:  # and one that broke off inside the parameters
                f.truncate(len(header) + 4 * N - 4)
            self.assertNotEqual(run(full, "--verify").returncode, 0)

    def test_verify_rejects_a_file_that_is_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            junk, empty = os.path.join(d, "junk.bin"), os.path.join(d, "empty.bin")
            with open(junk, "wb") as f:
                f.write(b"not json at all\n" + b"\0" * 64)
            open(empty, "wb").close()
            self.assertNotEqual(run(junk, "--verify").returncode, 0)
            self.assertNotEqual(run(empty, "--verify").returncode, 0)

    def test_wrong_argument_count_prints_usage(self):
        r = run("only_one.bin")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ckpt_params_only.py", r.stderr)


if __name__ == "__main__":
    unittest.main()
