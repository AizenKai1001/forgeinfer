#!/usr/bin/env python3
"""test_contamination.py - tests of tools/contamination_check.py.

    python tools/test_contamination.py [-v] [--tokenizer FILE] [--slow]

Needs numpy (and `tokenizers` plus a tokenizer_f1.json for the real-tokenizer tests, which are skipped otherwise: they look in --tokenizer, $F1_TOKENIZER,
~/forgeinfer/data/tokenizer_f1.json and ./tokenizer_f1.json).  Runs on the homelab (venv python) and on the PC.  Everything happens in a temporary directory.

What is checked (each test says why it matters)
  * the prefix-sum rolling hash equals the per-window polynomial hash (and is the same wherever the window sits)
  * a planted eval text is found at both window sizes, at the start of a file, in the middle, at the very end, and ACROSS a chunk boundary; the result is bit-identical for every chunk size
  * the result equals a brute-force oracle (a Python dict of every eval window against every corpus window), so no hit is missed and none is invented, also for near-misses (15 of 16 tokens equal)
  * random tokens are not flagged
  * a deliberately weak hash (10 bits) creates thousands of hash matches and the exact verification rejects every false one
  * the whole-text match for texts shorter than the window (anchor 8 + exact extension), with a near-miss that must be rejected
  * bit 15 (SFT loss mask) is cleared before hashing when asked, and not otherwise
  * O_DIRECT reads (Linux) give the same tokens as buffered reads, including a file whose size is not a multiple of the page
  * end to end on a toy evaluation set: index -> scan -> report; item counts, strong hits, per-source counts, the correct-versus-wrong analysis, the clean/dirty subset files and their manifest
  * with the real tokenizer: the leading-space and the typography variants of a text are found; digits and special tokens behave
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contamination_check as cc  # noqa: E402

ARGS = {"tokenizer": None, "slow": False}
QUIET = lambda msg: None  # noqa: E731


class MockTok:
    """one token per whitespace-separated word; ids 16..32015 (never 0: the document separator)"""

    path = None
    sha256 = "mock"

    @staticmethod
    def _id(w):
        return 16 + zlib.crc32(w.encode("utf-8")) % 32000

    def encode_many(self, strings):
        ids, lens = [], []
        for s in strings:
            t = [self._id(w) for w in re.findall(r"\S+", s)]
            ids += t
            lens.append(len(t))
        return np.asarray(ids, dtype=np.uint16), np.asarray(lens, dtype=np.int64)

    def decode(self, ids):
        return " ".join(f"<{int(i)}>" for i in ids)


def rwords(rng, n, tag="w"):
    return " ".join(f"{tag}{rng.randrange(10 ** 6)}" for _ in range(n))


def make_eval(root, rng, n_mc=24, n_hs=8, n_lam=8):
    """a toy evaluation set in the ladder layout; returns the records"""
    for d in ("mc", "hellaswag", "lambada"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    mc = []
    for i in range(n_mc):
        q = rwords(rng, rng.randrange(24, 40), "q")
        ch = [{"ctx": f"Question: {q}\nAnswer:", "cont": " " + rwords(rng, rng.randrange(5, 12), "a")} for _ in range(4)]
        mc.append({"choices": ch, "label": rng.randrange(4)})
    # a short item (whole text 10 tokens) and a very short one (6 tokens)
    mc.append({"choices": [{"ctx": "Question: s1 s2 s3 s4\nAnswer:", "cont": " " + rwords(rng, 4, "sa")} for _ in range(2)], "label": 0})
    mc.append({"choices": [{"ctx": "Question: t1\nAnswer:", "cont": " " + rwords(rng, 2, "ta")} for _ in range(2)], "label": 1})
    with open(os.path.join(root, "mc", "toyqa.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in mc:
            f.write(json.dumps(r) + "\n")
    hs = []
    for i in range(n_hs):
        ca = rwords(rng, 22, "h")
        hs.append({"ind": i, "activity_label": "Toy", "ctx_a": ca, "ctx_b": "he " + rwords(rng, 4, "hb"), "ctx": "", "endings": [rwords(rng, rng.randrange(14, 22), "e") for _ in range(4)], "label": str(rng.randrange(4))})
        hs[-1]["ctx"] = hs[-1]["ctx_a"] + " " + hs[-1]["ctx_b"]
    with open(os.path.join(root, "hellaswag", "hellaswag_val.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in hs:
            f.write(json.dumps(r) + "\n")
    lam = [{"ctx": rwords(rng, 50, "l"), "cont": " lastword%d" % i} for i in range(n_lam)]
    with open(os.path.join(root, "lambada", "toy_lambada.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in lam:
            f.write(json.dumps(r) + "\n")
    man = {"schema": 1, "order": ["hellaswag", "toyqa", "toy_lambada"], "tokenizer": {"file": "tokenizer_f1.json", "sha256": "x", "vocab": 32768}, "tasks": {
        "hellaswag": {"title": "HS", "kind": "hellaswag", "file": "hellaswag/hellaswag_val.jsonl", "examples": n_hs, "choices": 4, "chance": 25.0, "headline": "acc_norm", "quick": 4, "full_limit": n_hs, "seq": 256, "bytes": 0, "sha256": ""},
        "toyqa": {"title": "toy", "kind": "mc", "file": "mc/toyqa.jsonl", "examples": len(mc), "choices": 4, "chance": 25.0, "headline": "acc", "quick": 10, "full_limit": len(mc), "seq": 64, "bytes": 0, "sha256": ""},
        "toy_lambada": {"title": "lam", "kind": "lambada", "file": "lambada/toy_lambada.jsonl", "examples": n_lam, "choices": 1, "chance": 0.0, "headline": "acc", "quick": 4, "full_limit": n_lam, "seq": 128, "bytes": 0, "sha256": ""}}}
    json.dump(man, open(os.path.join(root, "MANIFEST.json"), "w"))
    return mc, hs, lam


class Idx:
    """convenience view of an index directory"""

    def __init__(self, out):
        self.z = np.load(os.path.join(out, "index", "index.npz"))
        self.tok = self.z["tokens"]
        self.vs, self.vl, self.vt, self.vty = self.z["v_start"], self.z["v_len"], self.z["v_text"], self.z["v_type"]
        self.meta = json.load(open(os.path.join(out, "index", "index.json")))
        self.tk, self.ti, self.tc = self.z["t_kind"], self.z["t_item"], self.z["t_choice"]
        self.ib, self.iidx = self.z["item_bench"], self.z["item_idx"]
        self.benches = self.meta["benchmarks"]

    def text(self, bench, item_idx, kind, choice=-1, vtype=0):
        bi = self.benches.index(bench)
        items = np.flatnonzero((self.ib == bi) & (self.iidx == item_idx))
        assert len(items) == 1, (bench, item_idx, items)
        for t in np.flatnonzero((self.ti == items[0]) & (self.tk == cc.KIND_ID[kind]) & (self.tc == choice)):
            v = np.flatnonzero((self.vt == t) & (self.vty == vtype))
            if len(v):
                return int(t), self.tok[self.vs[v[0]]:self.vs[v[0]] + self.vl[v[0]]].copy()
        raise KeyError((bench, item_idx, kind, choice, vtype))

    def eval_windows(self, N):
        d = {}
        for v in range(len(self.vl)):
            a, l = int(self.vs[v]), int(self.vl[v])
            for i in range(l - N + 1):
                w = self.tok[a + i:a + i + N]
                if not (w == 0).any():
                    d[w.tobytes()] = True
        return d


def oracle(idx, corpus, N, mask15=False):
    """brute force: every corpus window that is an eval window -> (count, first position)"""
    ev = idx.eval_windows(N)
    x = (corpus & np.uint16(0x7FFF)) if mask15 else corpus
    res = {}
    for p in range(len(x) - N + 1):
        k = x[p:p + N].tobytes()
        if k in ev:
            c = res.get(k)
            res[k] = (c[0] + 1, c[1]) if c else (1, p)
    return res


def tool_result(out, name):
    """{window bytes: (count, first position)} summed over the scan files of the directory (one file per call here)"""
    meta, hasher, ixs = cc.load_index(out, log=QUIET)
    ix = [i for i in ixs if i.name == name][0]
    res = {}
    for jp in sorted(os.listdir(os.path.join(out, "scan"))):
        if not jp.endswith(".npz") or jp.endswith(".eos.npz"):
            continue
        with np.load(os.path.join(out, "scan", jp)) as r:
            r = {k: r[k] for k in r.files}
        for g, c, p in zip(r[f"{name}_g"], r[f"{name}_c"], r[f"{name}_p"]):
            k = ix.rep[g].tobytes()
            assert k not in res
            res[k] = (int(c), int(p))
    return res


def write_corpus(path, arr):
    np.asarray(arr, dtype="<u2").tofile(path)


def background(rng, n):
    return np.array([rng.randrange(16, 32768) for _ in range(n)], dtype=np.uint16)


class HashTests(unittest.TestCase):
    def test_powers_and_inverse(self):
        h = cc.Hasher(7)
        h.ensure(50)
        for i in (0, 1, 2, 17, 49):
            self.assertEqual(int(h.pw[i]), pow(h.B, i, 1 << 64))
            self.assertEqual(int(h.ipw[i]), pow(h.Binv, i, 1 << 64))
            self.assertEqual((int(h.pw[i]) * int(h.ipw[i])) & cc.MASK64, 1)
        self.assertEqual(h.B % 2, 1)

    def test_prefix_windows_equal_naive(self):
        rng = np.random.default_rng(3)
        x = rng.integers(0, 65536, size=5000, dtype=np.uint16)
        for weak in (64, 12):
            h = cc.Hasher(11, weak_bits=weak)
            cs = h.prefix(x)
            for N in (1, 2, 8, 10, 16, 33):
                m = len(x) - N + 1
                w = h.windows(cs, N, m)
                nv, _ = h.naive(x, np.arange(m), N)
                self.assertTrue(np.array_equal(w, nv), (weak, N))
                # a shorter evaluation range of the same chunk gives the same prefix of hashes
                w2 = h.windows(cs, N, 100)
                self.assertTrue(np.array_equal(w2, w[:100]))

    def test_position_independence_and_sensitivity(self):
        rng = np.random.default_rng(4)
        win = rng.integers(1, 32768, size=16, dtype=np.uint16)
        x = np.concatenate([rng.integers(1, 32768, size=300, dtype=np.uint16), win, rng.integers(1, 32768, size=500, dtype=np.uint16), win])
        h = cc.Hasher(5)
        w = h.windows(h.prefix(x), 16, len(x) - 15)
        self.assertEqual(int(w[300]), int(w[300 + 16 + 500]))
        y = x.copy()
        y[310] ^= 1
        w3 = h.windows(h.prefix(y), 16, len(y) - 15)
        self.assertNotEqual(int(w3[300]), int(w[300]))
        self.assertEqual(int(w3[0]), int(w[0]))

    def test_few_collisions_on_random_windows(self):
        rng = np.random.default_rng(6)
        x = rng.integers(0, 65536, size=400_000, dtype=np.uint16)
        h = cc.Hasher(9)
        w = h.windows(h.prefix(x), 16, len(x) - 15)
        self.assertEqual(len(np.unique(w)), len(w))
        # the top bits (used by the bitmap) are spread evenly: 2^12 cells, 400k windows, every cell used
        cells = np.bincount((w >> np.uint64(52)).astype(np.int64), minlength=4096)
        self.assertGreater(cells.min(), 40)


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="contam_test_")
        self.rng = random.Random(1234)
        self.eval = os.path.join(self.tmp, "eval")
        self.mc, self.hs, self.lam = make_eval(self.eval, self.rng)
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, hasher=None):
        return cc.build_index(self.eval, MockTok(), self.out, Ns=(16, 10), anchor=8, hasher=hasher or cc.Hasher(21), log=QUIET)

    def scan(self, path, chunk=2048, mask15=False, direct=False, label="t"):
        cc.run_scan(self.out, [f"{label}={path}"], chunk=chunk, mask15_labels={label} if mask15 else (), direct=direct, log=QUIET)

    def test_index_basics(self):
        meta = self.build()
        self.assertEqual(meta["benchmarks"], ["hellaswag", "toy_lambada", "toyqa"])
        self.assertEqual(meta["items_per_bench"], {"hellaswag": 8, "toyqa": 26, "toy_lambada": 8})
        idx = Idx(self.out)
        # the toy short item has a join text of exactly 4 + 4 = 8 tokens ("s1 s2 s3 s4 sa.. sa.. sa.. sa..") and a "full" text of 10 tokens: both are below 16
        t, tok = idx.text("toyqa", 24, "join", 0)
        self.assertEqual(len(tok), 8)
        # variants: a leading-space variant exists for natural kinds only
        self.assertTrue(((idx.vt == t) & (idx.vty == 1)).any())
        t_full, _ = idx.text("toyqa", 0, "full", 0)
        self.assertFalse(((idx.vt == t_full) & (idx.vty == 1)).any())
        self.assertTrue((idx.tok != 0).all())  # the mock tokenizer never produces the separator token
        # HellaSwag texts: the processed text and the raw text (the raw ctx keeps the lower-case "he" of ctx_b) are both there
        kinds = set(int(k) for k in idx.tk[idx.ib[idx.ti] == idx.benches.index("hellaswag")])
        for k in ("ctx", "end", "join", "raw_ctx", "raw_join"):
            self.assertIn(cc.KIND_ID[k], kinds, k)

    def test_planted_texts_match_oracle_all_chunk_sizes(self):
        self.build()
        idx = Idx(self.out)
        rng = random.Random(77)
        plants = [("toyqa", 3, "join", 2), ("toyqa", 5, "ctx", -1), ("hellaswag", 2, "join", 1), ("toy_lambada", 4, "passage", -1)]
        arrs = [idx.text(b, i, k, c)[1] for b, i, k, c in plants]
        near = arrs[2].copy()
        near[7] = (int(near[7]) + 1) % 32768 or 17  # a near miss: one token differs
        corpus = [background(rng, 3000)]
        corpus += [arrs[0], background(rng, 1000), arrs[1], background(rng, 500), arrs[1], background(rng, 700), near, background(rng, 800), arrs[3]]
        corpus = np.concatenate(corpus + [background(rng, 100)])
        # also: the first text at the very start of the file and another at the very end
        corpus = np.concatenate([arrs[2], corpus, arrs[0]])
        path = os.path.join(self.tmp, "c.u16")
        write_corpus(path, corpus)
        for N in (16, 10):
            want = oracle(idx, corpus, N)
            self.assertGreater(len(want), 100)
            ref = None
            for chunk in (64, 100, 1000, 2048, 4096, 1 << 20):
                # every chunk size: results identical to the oracle (and so to each other)
                shutil.rmtree(os.path.join(self.out, "scan"), ignore_errors=True)
                self.scan(path, chunk=chunk)
                got = tool_result(self.out, f"n{N}")
                self.assertEqual(got, want, f"N={N} chunk={chunk}")
                ref = ref or got
        # the planted item must be found with ALL of its windows
        for N in (16, 10):
            got = tool_result(self.out, f"n{N}")
            for a in arrs:
                for i in range(len(a) - N + 1):
                    self.assertIn(a[i:i + N].tobytes(), got)

    def test_chunk_boundary_straddle(self):
        self.build()
        idx = Idx(self.out)
        rng = random.Random(5)
        _, a = idx.text("toyqa", 7, "join", 1)
        chunk = 1024
        for off in (chunk - 1, chunk - 5, chunk - 15, chunk - 16, chunk, chunk + 1, 2 * chunk - len(a) // 2):
            corpus = np.concatenate([background(rng, off), a, background(rng, 3000)])
            path = os.path.join(self.tmp, "b.u16")
            write_corpus(path, corpus)
            shutil.rmtree(os.path.join(self.out, "scan"), ignore_errors=True)
            self.scan(path, chunk=chunk)
            for N in (16, 10):
                got = tool_result(self.out, f"n{N}")
                self.assertEqual(got, oracle(idx, corpus, N), (off, N))
                self.assertEqual(len(set(a[i:i + N].tobytes() for i in range(len(a) - N + 1))), len(a) - N + 1)  # the windows of the planted text are distinct ...
                for i in range(len(a) - N + 1):
                    self.assertEqual(got[a[i:i + N].tobytes()], (1, off + i))  # ... and every one is found exactly once, at its true position

    def test_random_corpus_has_no_hits(self):
        self.build()
        rng = random.Random(8)
        corpus = background(rng, 200_000)
        path = os.path.join(self.tmp, "r.u16")
        write_corpus(path, corpus)
        self.scan(path, chunk=4096)
        for N in (16, 10):
            self.assertEqual(tool_result(self.out, f"n{N}"), {})
        st = json.load(open(os.path.join(self.out, "scan", "t__r.u16.json")))
        self.assertEqual(st["ngrams"]["n16"]["verified"], 0)

    def test_weak_hash_collisions_are_rejected(self):
        """a 30-bit hash: random corpus windows whose hash equals an eval window's hash (but whose tokens differ) exist in numbers; the exact verification must reject every one of them"""
        hasher = cc.Hasher(31, weak_bits=30)
        self.build(hasher=hasher)
        idx = Idx(self.out)
        meta, hs, ixs = cc.load_index(self.out, log=QUIET)
        uniq = [i for i in ixs if i.name == "n16"][0].uniq
        nrng = np.random.default_rng(5)
        forged = []
        for _ in range(40):  # find corpus windows that collide with an eval window by brute force (about 2 per batch of 400k windows)
            x = nrng.integers(16, 32768, size=400_015, dtype=np.uint16)
            w = hasher.windows(hasher.prefix(x), 16, 400_000)
            for p in np.flatnonzero(np.isin(w, uniq)):
                forged.append(x[p:p + 16].copy())
            if len(forged) >= 6:
                break
        self.assertGreaterEqual(len(forged), 3)
        rng = random.Random(9)
        _, a = idx.text("hellaswag", 1, "ctx", -1)
        parts = [background(rng, 300)]
        for f in forged:
            parts += [f, background(rng, 100)]
        parts += [a, background(rng, 300)]
        corpus = np.concatenate(parts)
        path = os.path.join(self.tmp, "w.u16")
        write_corpus(path, corpus)
        self.scan(path, chunk=4096)
        st = json.load(open(os.path.join(self.out, "scan", "t__w.u16.json")))
        self.assertGreaterEqual(st["ngrams"]["n16"]["collisions"], len(forged))  # the forged windows passed the hash comparison ...
        for N in (16, 10):
            self.assertEqual(tool_result(self.out, f"n{N}"), oracle(idx, corpus, N), N)  # ... and none of them was counted: only the planted text is reported
        self.assertEqual(len(tool_result(self.out, "n16")), len(a) - 15)

    def test_short_texts_whole_match(self):
        self.build()
        idx = Idx(self.out)
        rng = random.Random(10)
        t8, a8 = idx.text("toyqa", 24, "join", 0)  # "s1 s2 s3 s4" + 4 answer words = 8 tokens: eligible (anchor 8)
        t6, a6 = idx.text("toyqa", 25, "join", 0)  # 3 tokens ("t1 ta ta"): below the anchor, not testable
        t10, a10 = idx.text("toyqa", 24, "full", 1)  # "Question: s1 s2 s3 s4 Answer: + 4 words" = 10 tokens
        self.assertEqual((len(a8), len(a6), len(a10)), (8, 3, 10))
        near10 = a10.copy()
        near10[9] = 17 if near10[9] != 17 else 18  # the first 8 tokens (the anchor) are equal, the last two differ in one token
        corpus = np.concatenate([background(rng, 500), a8, background(rng, 500), a10, background(rng, 500), near10, background(rng, 500), a6, background(rng, 100)])
        path = os.path.join(self.tmp, "s.u16")
        write_corpus(path, corpus)
        self.scan(path, chunk=2048)
        with np.load(os.path.join(self.out, "scan", "t__s.u16.npz")) as r:
            vid, cnt, pos = r["short_g"], r["short_c"], r["short_p"]
        found = {}
        for v, c, p in zip(vid, cnt, pos):
            found.setdefault(int(idx.vt[v]), []).append((int(c), int(p)))
        self.assertIn(t8, found)
        self.assertIn(t10, found)
        self.assertEqual(sum(c for c, _ in found[t10]), 1)  # the near miss with an equal anchor is NOT counted: one occurrence only
        self.assertEqual(sorted(p for _, p in found[t10]), [500 + 8 + 500])
        self.assertEqual(sorted(p for _, p in found[t8]), [500] * len(found[t8]))  # (the leading-space variant has the same tokens: it is found at the same place)
        self.assertNotIn(t6, found)
        st = json.load(open(os.path.join(self.out, "scan", "t__s.u16.json")))
        self.assertGreaterEqual(st["ngrams"]["short"]["hash_match"], 3)  # the anchors of both whole texts and of the near miss matched ...
        self.assertEqual(st["ngrams"]["short"]["verified"], int(cnt.sum()))  # ... but only the real ones survive the extension
        self.assertEqual(st["ngrams"]["short"]["verified"], len(found[t8]) + 1)

    def test_bit15_mask(self):
        self.build()
        idx = Idx(self.out)
        rng = random.Random(11)
        _, a = idx.text("toy_lambada", 2, "passage", -1)
        masked = a.copy()
        masked[: len(a) // 2] |= np.uint16(0x8000)
        corpus = np.concatenate([background(rng, 1000), masked, background(rng, 1000)])
        path = os.path.join(self.tmp, "m.u16")
        write_corpus(path, corpus)
        self.scan(path, chunk=2048, mask15=False)
        self.assertLess(len(tool_result(self.out, "n16")), len(a) - 15)  # the windows that contain a flagged token are not equal to the eval windows
        shutil.rmtree(os.path.join(self.out, "scan"), ignore_errors=True)
        self.scan(path, chunk=2048, mask15=True)
        got = tool_result(self.out, "n16")
        self.assertEqual(got, oracle(idx, corpus, 16, mask15=True))
        self.assertEqual(len(got), len(a) - 15)

    @unittest.skipUnless(hasattr(os, "O_DIRECT") and hasattr(os, "preadv"), "no O_DIRECT on this OS")
    def test_direct_reads_equal_buffered(self):
        rng = np.random.default_rng(2)
        n = 3 * 2048 * 5 + 1234  # not a multiple of the page
        x = rng.integers(0, 65536, size=n, dtype=np.uint16)
        path = os.path.join(self.tmp, "d.u16")
        write_corpus(path, x)
        for direct in (True, False):
            rd = cc.Reader(path, 4096 + 15, direct=direct)
            try:
                for start in range(0, n, 4096):
                    got = rd.read(start, 4096 + 15)
                    self.assertTrue(np.array_equal(got, x[start:start + 4096 + 15]), (direct, start))
                self.assertEqual(len(rd.read(n - 10, 4096)), 10)
                self.assertEqual(len(rd.read(n, 4096)), 0)
            finally:
                rd.close()

    @unittest.skipUnless(hasattr(os, "O_DIRECT") and hasattr(os, "preadv"), "no O_DIRECT on this OS")
    def test_scan_with_direct_equals_buffered(self):
        self.build()
        idx = Idx(self.out)
        rng = random.Random(12)
        _, a = idx.text("toyqa", 4, "join", 3)
        corpus = np.concatenate([background(rng, 2048 - 8), a, background(rng, 9000)])
        path = os.path.join(self.tmp, "x.u16")
        write_corpus(path, corpus)
        results = []
        for direct in (True, False):
            shutil.rmtree(os.path.join(self.out, "scan"), ignore_errors=True)
            self.scan(path, chunk=2048, direct=direct)
            results.append(tool_result(self.out, "n16"))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], oracle(idx, corpus, 16))

    def test_document_separators(self):
        """token 0 positions are recorded for every chunk size, also across chunk boundaries and with the loss-mask bit set; the eos-only pass makes the same file"""
        self.build()
        rng = random.Random(14)
        corpus = background(rng, 20_000)
        zeros = [0, 1, 1023, 1024, 1025, 2047, 2048, 4095, 4096, 9999, 19_999]
        corpus[zeros] = 0
        corpus[[500, 3000]] = 0x8000  # a masked separator (SFT shards): token 0 once the bit is cleared
        path = os.path.join(self.tmp, "z.u16")
        write_corpus(path, corpus)
        for chunk, mask in ((1024, False), (2048, False), (1000, True), (1 << 20, True)):
            shutil.rmtree(os.path.join(self.out, "scan"), ignore_errors=True)
            self.scan(path, chunk=chunk, mask15=mask)
            want = sorted(zeros + ([500, 3000] if mask else []))
            with np.load(os.path.join(self.out, "scan", "t__z.u16.eos.npz")) as got:
                self.assertEqual(got["eos"].tolist(), want, (chunk, mask))
                self.assertEqual(int(got["size"]), len(corpus))
        # the fast pass: only the separators, no index needed, and it skips a file that already has them
        os.remove(os.path.join(self.out, "scan", "t__z.u16.eos.npz"))
        cc.run_scan(self.out, [f"t={path}"], chunk=2048, mask15_labels={"t"}, direct=False, eos_only=True, log=QUIET)
        with np.load(os.path.join(self.out, "scan", "t__z.u16.eos.npz")) as got:
            self.assertEqual(got["eos"].tolist(), sorted(zeros + [500, 3000]))
        # a finished scan whose separators are missing gets only them added (the hits are not recomputed)
        jp = os.path.join(self.out, "scan", "t__z.u16.json")
        m1 = os.path.getmtime(jp)
        os.remove(os.path.join(self.out, "scan", "t__z.u16.eos.npz"))
        self.scan(path, chunk=1 << 20, mask15=True)  # the chunk size of the last full scan above
        self.assertTrue(os.path.exists(os.path.join(self.out, "scan", "t__z.u16.eos.npz")))
        self.assertEqual(os.path.getmtime(jp), m1)

    def test_resume_skips_finished_files(self):
        self.build()
        rng = random.Random(13)
        path = os.path.join(self.tmp, "u.u16")
        write_corpus(path, background(rng, 5000))
        self.scan(path, chunk=2048)
        jp = os.path.join(self.out, "scan", "t__u.u16.json")
        m1 = os.path.getmtime(jp)
        self.scan(path, chunk=2048)
        self.assertEqual(os.path.getmtime(jp), m1)

    def test_parts_and_sources(self):
        files = []
        for i, sz in enumerate((100, 90, 80, 10, 5)):
            p = os.path.join(self.tmp, f"p{i}.u16")
            write_corpus(p, np.zeros(sz, np.uint16))
            files.append(("lab", p))
        a, b = cc.assign_part(files, (0, 2)), cc.assign_part(files, (1, 2))
        self.assertEqual(sorted(a + b), sorted(files))
        self.assertEqual(set(a) & set(b), set())
        self.assertEqual(cc.source_of("main", "/x/fwedu_003.u16"), "main/fwedu")
        self.assertEqual(cc.source_of("main", "/x/zzz_val.u16"), "main/zzz_val")
        self.assertEqual(cc.source_of("cool", "/x/cool_011.u16"), "cool/cool")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="contam_e2e_")
        self.rng = random.Random(99)
        self.eval = os.path.join(self.tmp, "eval")
        self.mc, self.hs, self.lam = make_eval(self.eval, self.rng)
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_report_numbers(self):
        tok = MockTok()
        cc.build_index(self.eval, tok, self.out, Ns=(16, 10), anchor=8, hasher=cc.Hasher(3), log=QUIET)
        idx = Idx(self.out)
        rng = random.Random(1)
        # planted in "web": the correct-choice join text of toyqa item 3 (full item present), the question of item 5 only, and 20 tokens of a hellaswag join
        label3 = self.mc[3]["label"]
        _, join3 = idx.text("toyqa", 3, "join", label3)
        _, ctx5 = idx.text("toyqa", 5, "ctx", -1)
        _, hsj = idx.text("hellaswag", 2, "join", 0)
        _, lam1 = idx.text("toy_lambada", 1, "passage", -1)
        web = np.concatenate([background(rng, 3000), join3, background(rng, 2000), ctx5, background(rng, 2000), hsj[:20], background(rng, 3000)])
        # in "books": the lambada passage of item 1
        books = np.concatenate([background(rng, 1000), lam1, background(rng, 1000)])
        wp, bp = os.path.join(self.tmp, "web_000.u16"), os.path.join(self.tmp, "books_000.u16")
        write_corpus(wp, web)
        write_corpus(bp, books)
        cc.run_scan(self.out, [f"main={wp}", f"main={bp}"], chunk=2048, direct=False, log=QUIET)
        rdir, sdir = os.path.join(self.tmp, "rep"), os.path.join(self.tmp, "subsets")
        rep = cc.make_report(self.out, tok=tok, eval_dir=self.eval, report_dir=rdir, subset_dir=sdir, top=3, log=QUIET)
        b = rep["benchmarks"]
        self.assertEqual(b["toyqa"]["n16"]["items_hit"], 2)  # items 3 and 5
        self.assertEqual(b["toyqa"]["n16"]["items_strong_text"], 2)
        self.assertEqual(b["toyqa"]["n16"]["by_source"]["main/web"]["items_hit"], 2)
        self.assertEqual(b["toyqa"]["n16"]["by_source"]["main/books"]["items_hit"], 0)
        self.assertEqual(b["hellaswag"]["n16"]["items_hit"], 1)  # 20 tokens of a join text: a few windows, below half of the item's windows
        self.assertEqual(b["hellaswag"]["n16"]["items_strong_text"], 0)
        self.assertEqual(b["hellaswag"]["n10"]["items_hit"], 1)
        self.assertEqual(b["toy_lambada"]["n16"]["items_hit"], 1)
        self.assertEqual(b["toyqa"]["present"]["present"], 2)  # the union over 16, 10 and the short whole-text match: items 3 and 5
        self.assertEqual(b["hellaswag"]["present"]["present"], 1)  # 20 of the 27 tokens of the context: half of the 10-token windows are there
        self.assertEqual(b["toy_lambada"]["present"]["present"], 1)
        self.assertEqual(b["toy_lambada"]["n16"]["by_source"]["main/books"]["items_hit"], 1)
        self.assertEqual(b["toy_lambada"]["n16"]["by_source"]["main/web"]["items_hit"], 0)
        # correct-versus-wrong: item 3's correct choice text is present, nothing of the wrong ones (the question of item 5 is shared by all of its choices, so it is not choice-specific)
        c = b["toyqa"]["n16"]["choices"]
        self.assertEqual(c["correct_hit"], 1)
        self.assertEqual(c["wrong_hit"], 0)
        self.assertEqual(c["only_correct"], 1)
        # the short items are too short for 16 tokens and are listed as such
        self.assertEqual(b["toyqa"]["n16"]["too_short_items"], 2)
        self.assertEqual(b["toyqa"]["short"]["items_too_short"], 2)
        # examples decode and carry their provenance
        ex = rep["examples"]["toyqa"]["top"]
        self.assertEqual(len(ex), 2)
        self.assertEqual(ex[0]["source"], "main/web")
        self.assertIn("[[", ex[0]["context"])
        # subsets: dirty16 holds exactly the two hit toyqa records; clean16 the rest; manifest adjusted
        dirty = [json.loads(ln) for ln in open(os.path.join(sdir, "dirty16", "mc", "toyqa.jsonl"))]
        clean = [json.loads(ln) for ln in open(os.path.join(sdir, "clean16", "mc", "toyqa.jsonl"))]
        self.assertEqual(len(dirty), 2)
        self.assertEqual(len(clean), len(self.mc) - 2)
        self.assertIn(self.mc[3], dirty)
        self.assertIn(self.mc[5], dirty)
        man = json.load(open(os.path.join(sdir, "dirty16", "MANIFEST.json")))
        self.assertEqual(man["tasks"]["toyqa"]["examples"], 2)
        man = json.load(open(os.path.join(sdir, "clean16", "MANIFEST.json")))
        self.assertEqual(man["tasks"]["toyqa"]["examples"], len(self.mc) - 2)
        self.assertTrue(os.path.getsize(os.path.join(sdir, "clean16", "mc", "toyqa.jsonl")) == man["tasks"]["toyqa"]["bytes"])
        for f in ("contamination_report.md", "contamination_results.json"):
            self.assertTrue(os.path.isfile(os.path.join(rdir, f)))

    def test_documents_with_many_items(self):
        """three items of one benchmark in one document and one more item in another: the first document is reported as holding 3 items"""
        tok = MockTok()
        cc.build_index(self.eval, tok, self.out, Ns=(16, 10), anchor=8, hasher=cc.Hasher(3), log=QUIET)
        idx = Idx(self.out)
        rng = random.Random(4)
        q = [idx.text("toyqa", i, "ctx", -1)[1] for i in (2, 4, 6, 8)]
        corpus = np.concatenate([background(rng, 300), np.array([0], np.uint16), background(rng, 200), q[0], background(rng, 50), q[1], background(rng, 50), q[2], background(rng, 200), np.array([0], np.uint16),
                                 background(rng, 100), q[3], background(rng, 100), np.array([0], np.uint16)])
        path = os.path.join(self.tmp, "web_000.u16")
        write_corpus(path, corpus)
        cc.run_scan(self.out, [f"main={path}"], chunk=2048, direct=False, log=QUIET)
        for with_eos in (True, False):  # with the recorded separator positions, and without them (the document is found by reading around the hit)
            if not with_eos:
                os.remove(os.path.join(self.out, "scan", "web_000.u16".join(["main__", ".eos.npz"])))
            rep = cc.make_report(self.out, tok=tok, eval_dir=self.eval, report_dir=os.path.join(self.tmp, "rep"), subset_dir=None, top=2, log=QUIET)
            d = rep["documents"]["n16"]["per_benchmark"]["toyqa"]
            self.assertEqual(d["documents_with_strong_items"], 2, with_eos)
            self.assertEqual(d["documents_with_min_items"], 1)
            self.assertEqual(d["top"][0]["items"], 3)
            self.assertEqual(d["top"][0]["start"], 301)
            self.assertEqual(d["top"][0]["tokens"], 200 + len(q[0]) + 50 + len(q[1]) + 50 + len(q[2]) + 200)
            self.assertEqual(d["items_in_documents_with_5_or_more_items"], 0)
            self.assertIn("<", d["top"][0]["start_of_document"])
        self.assertIn("Documents that hold the strong hits", open(os.path.join(self.tmp, "rep", "contamination_report.md"), encoding="utf-8").read())

    def test_doc_exact_count(self):
        """the exact count in a dump document: question found, correct answer line after it, all choice lines after it"""
        tok = MockTok()
        cc.build_index(self.eval, tok, self.out, Ns=(16, 10), anchor=8, hasher=cc.Hasher(3), log=QUIET)
        idx = Idx(self.out)
        rng = random.Random(6)
        nl = np.array([26], np.uint16)  # the newline token of the real tokenizer (doc_exact looks for choice text followed by it)

        def block(i, skip_correct=False):
            parts = [idx.text("toyqa", i, "ctx", -1)[1]]
            for c in range(4):
                if skip_correct and c == self.mc[i]["label"]:
                    continue
                parts += [idx.text("toyqa", i, "end", c)[1], nl]
            return np.concatenate(parts)

        corpus = np.concatenate([background(rng, 100), block(2), background(rng, 30), block(4), background(rng, 30), block(6), background(rng, 30), block(8, skip_correct=True), background(rng, 100)])
        path = os.path.join(self.tmp, "web_000.u16")
        write_corpus(path, corpus)
        cc.run_scan(self.out, [f"main={path}"], chunk=2048, direct=False, log=QUIET)
        R = cc.Results(self.out)
        ex = cc.doc_exact(R, R.files[0], 0, len(corpus), "toyqa")
        self.assertEqual(ex, {"questions_in_document": 4, "correct_answer_line_after": 3, "all_choice_lines_after": 3})

    def test_item_specific_windows(self):
        """two items that share their question: a hit on the shared question counts as a hit, but not as an item-specific one"""
        for p in ("mc",):
            path = os.path.join(self.eval, p, "toyqa.jsonl")
            recs = [json.loads(ln) for ln in open(path)]
            recs[1]["choices"] = [dict(c, ctx=recs[0]["choices"][0]["ctx"]) for c in recs[1]["choices"]]  # item 1 shares item 0's question
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        tok = MockTok()
        cc.build_index(self.eval, tok, self.out, Ns=(16, 10), anchor=8, hasher=cc.Hasher(3), log=QUIET)
        idx = Idx(self.out)
        _, q = idx.text("toyqa", 0, "ctx", -1)
        rng = random.Random(2)
        path = os.path.join(self.tmp, "web_000.u16")
        write_corpus(path, np.concatenate([background(rng, 500), q, background(rng, 500)]))
        cc.run_scan(self.out, [f"main={path}"], chunk=2048, direct=False, log=QUIET)
        rep = cc.make_report(self.out, tok=tok, eval_dir=self.eval, report_dir=os.path.join(self.tmp, "rep"), subset_dir=None, top=2, log=QUIET)
        r = rep["benchmarks"]["toyqa"]["n16"]
        self.assertEqual(r["items_hit"], 2)
        self.assertEqual(r["items_hit_specific"], 0)


QUESTION = "Which of the following statements about the lunar eclipse that astronomers observed from the northern hemisphere last winter is the most accurate description?"


class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("CONTAM_TEST_REAL_TOKENIZER"):
            raise unittest.SkipTest("real tokenizer not available (pass --tokenizer FILE; needs the `tokenizers` package)")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="contam_real_")
        self.tok = cc.HFTok(os.environ["CONTAM_TEST_REAL_TOKENIZER"])
        os.makedirs(os.path.join(self.tmp, "eval", "mc"))
        recs = [{"choices": [{"ctx": f"Question: {QUESTION}\nAnswer:", "cont": " It happens when the Earth passes between the Sun and the Moon."},
                             {"ctx": f"Question: {QUESTION}\nAnswer:", "cont": " It happens at noon."}], "label": 0},
                {"choices": [{"ctx": "Question: What’s the name of the “red planet” that astronomers have studied for centuries with telescopes?\nAnswer:", "cont": " Mars"},
                             {"ctx": "Question: What’s the name of the “red planet” that astronomers have studied for centuries with telescopes?\nAnswer:", "cont": " Venus"}], "label": 0}]
        with open(os.path.join(self.tmp, "eval", "mc", "toyreal.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.out = os.path.join(self.tmp, "out")
        cc.build_index(os.path.join(self.tmp, "eval"), self.tok, self.out, Ns=(16, 10), anchor=8, hasher=cc.Hasher(5), log=QUIET)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def encode(self, s):
        flat, lens = self.tok.encode_many([s])
        return flat

    def found_items(self, doc_text):
        rng = random.Random(3)
        ids = self.encode(doc_text)
        corpus = np.concatenate([background(rng, 2000), ids, background(rng, 2000)])
        path = os.path.join(self.tmp, "doc.u16")
        write_corpus(path, corpus)
        shutil.rmtree(os.path.join(self.out, "scan"), ignore_errors=True)
        cc.run_scan(self.out, [f"main={path}"], chunk=2048, direct=False, log=QUIET)
        rep = cc.make_report(self.out, tok=self.tok, eval_dir=os.path.join(self.tmp, "eval"), report_dir=os.path.join(self.tmp, "rep"), top=2, log=QUIET)
        return rep["benchmarks"]["toyreal"]["n16"]["items_hit"], rep

    def test_leading_space_variant(self):
        for prefix in ("", " ", "\n", "Quiz time. ", "(a) "):
            n, _ = self.found_items(prefix + QUESTION + " (A) one (B) two")
            self.assertEqual(n, 1, repr(prefix))

    def test_typography_variant(self):
        # the corpus has straight quotes where the benchmark has curly ones, and the other way round
        n, _ = self.found_items("What's the name of the \"red planet\" that astronomers have studied for centuries with telescopes?")
        self.assertEqual(n, 1)
        n, _ = self.found_items("What’s the name of the “red planet” that astronomers have studied for centuries with telescopes?")
        self.assertEqual(n, 1)

    def test_unrelated_text_is_clean(self):
        n, _ = self.found_items("The quick brown fox jumps over the lazy dog while the astronomers were asleep in their warm beds on a cold winter night in Norway.")
        self.assertEqual(n, 0)

    def test_encoder_agrees_with_decoder(self):
        s = "Hello 123 world, the Sun’s corona."
        ids = self.encode(s)
        self.assertEqual(self.tok.decode(ids), s)
        self.assertTrue((ids != 0).all())


def find_tokenizer(explicit=None):
    for c in (explicit, os.environ.get("F1_TOKENIZER"), os.path.expanduser("~/forgeinfer/data/tokenizer_f1.json"), os.path.join(os.getcwd(), "tokenizer_f1.json"),
              os.path.join(HERE, "tokenizer_f1.json")):
        if c and os.path.isfile(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a, rest = ap.parse_known_args()
    tokp = find_tokenizer(a.tokenizer)
    if tokp:
        try:
            import tokenizers  # noqa: F401

            os.environ["CONTAM_TEST_REAL_TOKENIZER"] = tokp
            print(f"real tokenizer tests: {tokp}")
        except ImportError:
            print("real tokenizer tests skipped: the `tokenizers` package is not installed")
    else:
        print("real tokenizer tests skipped: no tokenizer_f1.json found (use --tokenizer FILE)")
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2 if a.verbose else 1).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
