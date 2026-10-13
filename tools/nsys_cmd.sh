#!/bin/bash
# Kernel-time breakdown of any forge command (Nsight Systems, on the homelab; pause the trainer around it with gpu_window.sh):
#   tools/hl.sh "bash tools/gpu_window.sh bash tools/nsys_cmd.sh ./target/release/forge prefill-bench MODEL --native --tokens 512"
# Prints the GPU kernels by total time (top 30, with the number of calls and the average time).
export TMPDIR=$HOME/forgeinfer/tmp
mkdir -p $TMPDIR
cd $HOME/forgeinfer/src
$HOME/cuda-13.2.0/bin/nsys profile -o $TMPDIR/prof_cmd --force-overwrite true --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none "$@" > $TMPDIR/prof_cmd.log 2>&1
tail -n 2 $TMPDIR/prof_cmd.log
$HOME/cuda-13.2.0/bin/nsys stats --report cuda_gpu_kern_sum --format csv --force-export true $TMPDIR/prof_cmd.nsys-rep 2>/dev/null | python3 -c "
import csv, sys
rows = list(csv.reader(l for l in sys.stdin if l.strip() and not l.startswith('Generating') and not l.startswith('Processing')))
hdr = None
data = []
for r in rows:
    if r and r[0].startswith('Time'):
        hdr = r; continue
    if hdr and len(r) == len(hdr): data.append(dict(zip(hdr, r)))
tot = sum(float(d['Total Time (ns)']) for d in data)
print('total GPU kernel time %.1f ms over %d kernel names' % (tot / 1e6, len(data)))
for d in sorted(data, key=lambda d: -float(d['Total Time (ns)']))[:32]:
    print('%8.2f ms %5.1f %%  n=%-5s avg %9.1f us  %s' % (float(d['Total Time (ns)']) / 1e6, 100 * float(d['Total Time (ns)']) / tot, d['Instances'], float(d['Avg (ns)']) / 1e3, d['Name'][:90]))
"
