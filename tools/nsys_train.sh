#!/bin/bash
# Kernel-time breakdown of a few F125 training steps (Nsight Systems, on the homelab). Run it on an idle GPU.
#   tools/nsys_train.sh [steps] [extra forge-train flags]
# Prints the GPU kernels by total time (top 40) and the sum per kernel family.
KEY="$HOME/.ssh/id_ed25519"
HOST="user@homelab"
STEPS="${1:-3}"; shift
EXTRA="$*"
exec ssh -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 "$HOST" ". ~/.cargo/env 2>/dev/null; export TMPDIR=\$HOME/forgeinfer/tmp; mkdir -p \$TMPDIR; cd \$HOME/forgeinfer; \
  \$HOME/cuda-13.2.0/bin/nsys profile -o \$TMPDIR/prof_train --force-overwrite true --trace=cuda --sample=none --cpuctxsw=none \
  ./src/target/release/forge train --data data/tokens --out \$TMPDIR/prof_run_\$(date +%s) --layers 30 --dim 576 --heads 9 --kv 3 --ffn 1536 --seq 2048 --micro 2 --tokens-per-step 8192 --steps $STEPS --lr 1e-3 --warmup 1 --eval-every 1000 --save-every 100000 --qk-norm $EXTRA > \$TMPDIR/prof_run.log 2>&1; \
  tail -n 4 \$TMPDIR/prof_run.log; \
  \$HOME/cuda-13.2.0/bin/nsys stats --report cuda_gpu_kern_sum --format csv --force-export true \$TMPDIR/prof_train.nsys-rep 2>/dev/null | python3 -c \"
import csv, sys, collections
rows = list(csv.reader(l for l in sys.stdin if l.strip() and not l.startswith('Generating') and not l.startswith('Processing')))
hdr = None
data = []
for r in rows:
    if r and r[0].startswith('Time'):
        hdr = r; continue
    if hdr and len(r) == len(hdr): data.append(dict(zip(hdr, r)))
tot = sum(float(d['Total Time (ns)']) for d in data)
print('total GPU kernel time %.1f ms over %d kernel names' % (tot / 1e6, len(data)))
fam = collections.defaultdict(float)
for d in data:
    n = d['Name'].split('(')[0]
    f = n.split('<')[0]
    for key in ('gemm_', 'flash_', 'attn_', 'rmsnorm', 'swiglu', 'ce_', 'embed', 'qkv', 'adamw', 'grad_', 'f32_to_f16', 'mean', 'reduce', 'fill', 'muon'):
        if f.startswith(key) or key in f:
            fam[key] += float(d['Total Time (ns)']); break
    else:
        fam['other'] += float(d['Total Time (ns)'])
for k, v in sorted(fam.items(), key=lambda kv: -kv[1]): print('  family %-12s %8.1f ms  %5.1f %%' % (k, v / 1e6, 100 * v / tot))
print()
for d in sorted(data, key=lambda d: -float(d['Total Time (ns)']))[:40]:
    print('%8.2f ms %5.1f %%  n=%-5s avg %9.1f us  %s' % (float(d['Total Time (ns)']) / 1e6, 100 * float(d['Total Time (ns)']) / tot, d['Instances'], float(d['Avg (ns)']) / 1e3, d['Name'][:90]))
\""
