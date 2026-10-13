"""An executable SPECIFICATION of byte-level BPE TRAINING (the algorithm our Rust trainer ports): count pre-tokens, keep inverted pair indexes, merge the most frequent pair, update only the
affected words. Deterministic (ties broken by the smaller pair). Compared with the Hugging Face trainer on the same corpus (an ORACLE): compression (tokens per byte) must be within about 1 %.
    python3 tools/ref_bpe_train.py <text file> [vocab size=2048]       (homelab; the HF comparison needs the venv)
"""
import heapq
import sys
import time
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import ref_tokenizer as R


def train(text, vocab_size):
    # 1. pre-tokenize with the same scanner as the encoder, count words as byte tuples
    t0 = time.time()
    counts = defaultdict(int)
    for piece in pretokenize_all(text):
        counts[tuple(piece.encode("utf-8"))] += 1
    words = [list(w) for w in counts]
    freq = [counts[tuple(w)] for w in words]
    print(f"{len(words)} distinct pre-tokens from {sum(freq)} total ({time.time() - t0:.1f} s)")
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    pair_count = defaultdict(int)
    where = defaultdict(set)
    for wi, w in enumerate(words):
        for a, b in zip(w, w[1:]):
            pair_count[(a, b)] += freq[wi]
            where[(a, b)].add(wi)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)
    next_id = 256
    while next_id < vocab_size and heap:
        negc, pair = heapq.heappop(heap)
        if pair_count.get(pair, 0) != -negc or -negc <= 0:                     # stale entry
            if pair_count.get(pair, 0) > 0:
                heapq.heappush(heap, (-pair_count[pair], pair))
            continue
        a, b = pair
        new = next_id
        next_id += 1
        vocab[new] = vocab[a] + vocab[b]
        merges.append(pair)
        touched = set()
        for wi in list(where[pair]):
            w = words[wi]
            f = freq[wi]
            i, out = 0, []
            # remove this word's old pair counts, rebuild its symbols, add the new pair counts
            for x, y in zip(w, w[1:]):
                pair_count[(x, y)] -= f
                touched.add((x, y))
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for x, y in zip(out, out[1:]):
                pair_count[(x, y)] += f
                where[(x, y)].add(wi)
                touched.add((x, y))
        where[pair] = set()
        for p in touched:
            if pair_count.get(p, 0) > 0:
                heapq.heappush(heap, (-pair_count[p], p))
        if len(merges) % 500 == 0:
            print(f"  {len(merges)} merges, last {vocab[new]!r} ({-negc} occurrences)", flush=True)
    return vocab, merges


def pretokenize_all(text):
    out = []
    for part in R.split_digits_individual(text):
        out.extend(R.split_gpt2(part))
    return out


def encode(vocab_rank, text):
    """BPE encode with the merge ranks of a trained table (byte-level, same scanner)."""
    ids = []
    for piece in pretokenize_all(text):
        syms = list(piece.encode("utf-8"))
        while len(syms) > 1:
            best, at = None, -1
            for i in range(len(syms) - 1):
                r = vocab_rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best, at = (r, vocab_rank[(syms[i], syms[i + 1])]), i
            if best is None:
                break
            syms[at:at + 2] = [256 + best[0]]
        ids.extend(syms)
    return ids


def main():
    path = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    text = open(path, encoding="utf-8", errors="replace").read()
    t0 = time.time()
    vocab, merges = train(text, vocab_size)
    print(f"trained {len(merges)} merges in {time.time() - t0:.1f} s")
    rank = {m: i for i, m in enumerate(merges)}
    held = text[-200_000:]                                                   # compression on the last 200 KB (also in the training text: a smoke test, not a generalization claim)
    ids = encode(rank, held)
    back = b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
    print(f"round trip exact: {back == held}; tokens per byte on the held-out slice: {len(ids) / len(held.encode('utf-8')):.4f}")
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
        tk.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
        tk.train_from_iterator([text], tr)
        n = len(tk.encode(held).ids)
        print(f"Hugging Face trainer (oracle), same vocab size: tokens per byte {n / len(held.encode('utf-8')):.4f}")
    except Exception as e:
        print("no oracle comparison:", e)


if __name__ == "__main__":
    main()
