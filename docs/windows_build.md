# Native Windows build (RTX 4060)

`forge` (trainer, engine, CLI) builds natively on Windows 11 with MSVC, the CUDA toolkit and the Rust MSVC toolchain; no WSL. Branch `win-build`, written and checked on the user's PC on
2026-10-10. **Only the build and the CPU tests were run. Nothing has been started on the GPU yet**: the commands under "Benchmark" are what to run once the user agrees to use the 4060.

## Prerequisites (what this PC has)

| Tool | On this PC | Needed |
|---|---|---|
| Rust | 1.96.0 stable, host `x86_64-pc-windows-msvc` | stable MSVC toolchain (`rustup`), no crates are downloaded |
| CUDA toolkit | 12.9 (V12.9.41), `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`, `CUDA_PATH` set | 12.x or newer (nvcc is only used at build time) |
| C++ host compiler | Visual Studio 2022 Build Tools 17.14, MSVC 14.44.35207 (also VS 2022 Community) | **Visual Studio 2022** with "Desktop development with C++" (MSVC v143 + Windows SDK) |
| Windows SDK | 10.0.26100.0 | any recent one (comes with the workload) |
| NVIDIA driver | 617.42 (supports CUDA 13.4) | any driver that runs the RTX 4060; the binary loads `nvcuda.dll` at run time |

Visual Studio 2026 (MSVC 14.51) is also installed, but CUDA 12.9 refuses it (`crt/host_config.h` accepts `_MSC_VER` below 1950 only), so the build uses Visual Studio 2022.
WSL, CMake and Git are not needed.

## Build (one command)

```powershell
powershell -ExecutionPolicy Bypass -File <repo>-wt\win-build\tools\build_windows.ps1
```

(Paths in this file are for the `win-build` worktree; in another checkout, e.g. after a merge, replace `<repo>-wt\win-build` with that folder.)

The script finds Visual Studio 2022 with `vswhere`, loads its `vcvars64.bat` environment, checks nvcc, cargo and the driver DLL (file version only, the driver is not loaded), and runs
`cargo build --release -p forge-cli` with at most 6 jobs, the target folder `<repo>\target` and temp files under `<repo>\target\tmp`. It ends with the path of the binary:
`<repo>-wt\win-build\target\release\forge.exe`.

Options: `-Test` also runs the CPU-only tests and compiles (does not run) the GPU test binaries; `-CudaArchs none` leaves out the extra Ada cubin (`-CudaArchs 86,89` adds
more); `-Jobs 4`; `-TargetDir X:/somewhere`.

Measured on this PC (i5-12600K): a clean build with the script takes 5 min 22 s (5 min 3 s with plain cargo; 3 min 37 s without the extra sm_89 cubin): nvcc compiles `all.cu`
once per kernel image, about one minute each, one at a time. The binary is 178 MB = 170 MiB (145 MB without sm_89) because the kernel images are embedded. It imports only Windows
system DLLs, `VCRUNTIME140.dll` and the Universal CRT (`dumpbin /dependents`); `nvcuda.dll` is opened at run time with `LoadLibraryA`.

A plain `cargo build --release -p forge-cli` from any shell also works: when `cl.exe` is not on the PATH, `crates/forge-kernels/build.rs` finds the MSVC of Visual Studio 2022 itself and
passes it to nvcc through `NVCC_CCBIN` (nvcc then sets up the compiler environment on its own).

### Kernel images and the Ada switch

| Image | Built | Used on |
|---|---|---|
| sm_75 cubin | always | Turing (homelab 2060 SUPER, T4) |
| sm_80 cubin | always (unless `FORGE_NO_SM80=1`) | every 8.x GPU, including the 4060 (8.9) when no native sm_89 cubin is present |
| compute_80 / compute_75 PTX | always | 9.x and newer (JIT) / fallback |
| sm_89 cubin | when `FORGE_CUDA_ARCHS` contains 89 (the Windows script sets it by default) | exactly compute capability 8.9 (RTX 40xx, L4) |

`FORGE_CUDA_ARCHS` (build time) takes a list such as `86,89`; unset, nothing changes (the Linux build is as before). At run time `FORGE_NO_EXTRA_CUBIN=1` makes the same binary load the
portable sm_80 cubin instead of the native one, for an A/B. When a native cubin is involved, the program says so on stderr (`[kernels] native sm_89 cubin ...`).
Both images contain the Ampere GEMMs (`cp.async`, `mma.m16n8k16`); their dynamic shared memory (51 KB) is below the 99 KB per block that Ada allows.

## Benchmark on the 4060 (only after the user says yes)

**Warning: this runs the RTX 4060 at full load for about two minutes: fans up, the desktop may stutter, and games or other GPU programs compete for its 8 GB.** Close games, and
preferably browsers with video, first. It reads one local token shard (`<dev>\kaggle\stage\forge-data-web\fwedu_000.u16`, 1.58 GB, about 790 M tokens) and writes a log plus a
checkpoint of about 0.8 GB at the end.

The run is the 100-step version of the proxy of `cloud/kaggle_stage.py` (12 layers, d 384, 6/2 heads, seq 2048, 65,536 tokens per step, Muon) with the v2 recipe at proxy size
(`L8_combo`: ReLU² MLP of 1536, lambdas, value embeddings in layers 0-3 and 8-11, table rate 100x, embedding rate 3x): about 65 M parameters by my count, half of them value-embedding
tables (the trainer prints the exact number), 6.6 M tokens in all.

```powershell
# 1. build (and CPU tests)
powershell -ExecutionPolicy Bypass -File <repo>-wt\win-build\tools\build_windows.ps1 -Test

# 2. 100 training steps on the 4060 (native sm_89 image)
& <repo>-wt\win-build\target\release\forge.exe train --data <dev>\kaggle\stage\forge-data-web --out <dev>\forge-runs\4060-proxy --layers 12 --dim 384 --heads 6 --kv 2 --ffn 1536 --seq 2048 --micro 4 --tokens-per-step 65536 --steps 100 --warmup 30 --eval-every 100 --eval-batches 8 --save-every 100000 --lr 2e-3 --optim muon --muon-lr 2e-3 --muon-wd 0.05 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb "0,1,2,3,8,9,10,11" --lr-table-mult 100 --lr-emb-mult 3

# 3. average speed of steps 11-100
Get-Content <dev>\forge-runs\4060-proxy\log.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.step -gt 10 } | Measure-Object -Property tokens_per_s -Average -Minimum -Maximum
```

(The quotes around the layer list are not required in Windows PowerShell 5.1, which passes `0,1,2,3,8,9,10,11` through unchanged, but they keep the command safe in other shells.)

What to look at:

- The first lines: `[kernels] native sm_89 cubin (FORGE_CUDA_ARCHS) ...`, then `[train] NVIDIA GeForce RTX 4060 (... parameters, Ampere GEMMs) on <dev>\forge-runs\4060-proxy: ...`
  ("Ampere GEMMs" = the `cp.async` / `mma.m16n8k16` path; the A100 smoke run on Modal printed the same words).
- One line per step: `step N/100 loss L lr ... gnorm ... scale ... T tok/s (S s)`. The first two or three steps are slower (module load, first use); command 3 averages steps 11-100.
  The loss must fall from about 10.4 (ln 32768); for comparison the homelab's 150M v1 run was at 5.83 (val 5.96) after 100 steps on the same kind of data. A few
  `non-finite gradients, update skipped, loss scale -> ...` lines early on are normal, one on every step is a failure.
- `val` on the last line (step 100), and the checkpoint message.
- References for this proxy shape (500-step runs, `scratch/report_all.md` in the main checkout): homelab RTX 2060 SUPER about 55-64 k tokens/s for single options, Kaggle T4 about
  34-41 k (39.5 k for value embeddings in 8 layers). Nothing has been measured on the 4060 yet; its memory bandwidth (272 GB/s) is lower than the 2060 SUPER's (448 GB/s), its
  tensor throughput higher, so the result is open.

How to stop: `Ctrl+C` in the window (the process ends at once and the driver frees the GPU memory; no checkpoint unless the run had finished), or from another window
`Stop-Process -Name forge`.

Optional A/B of the two images with the same binary (loss values should agree closely, the speed may differ):

```powershell
$env:FORGE_NO_EXTRA_CUBIN = "1"; & <repo>-wt\win-build\target\release\forge.exe train --data <dev>\kaggle\stage\forge-data-web --out <dev>\forge-runs\4060-proxy-sm80 --layers 12 --dim 384 --heads 6 --kv 2 --ffn 1536 --seq 2048 --micro 4 --tokens-per-step 65536 --steps 100 --warmup 30 --eval-every 100 --eval-batches 8 --save-every 100000 --lr 2e-3 --optim muon --muon-lr 2e-3 --muon-wd 0.05 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb "0,1,2,3,8,9,10,11" --lr-table-mult 100 --lr-emb-mult 3; Remove-Item Env:FORGE_NO_EXTRA_CUBIN
```

Optional speed check at the production v2 shape (30 layers, d 576, ReLU² 2304, value embeddings in 0-3 and 26-29). References from the research notes: the T4 does 12.6-13.2 k tokens/s
with this recipe, the homelab 2060 SUPER trains the v1 run (same shape, four value-embedding layers) at about 15.1 k. 20 steps, each probably 5-7 s (estimate); it writes a checkpoint
of about 2 GB at the end. If it reports out of memory, use `--micro 1`:

```powershell
& <repo>-wt\win-build\target\release\forge.exe train --data <dev>\kaggle\stage\forge-data-web --out <dev>\forge-runs\4060-v2shape --layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --micro 2 --tokens-per-step 65536 --steps 20 --warmup 100 --eval-every 1000 --eval-batches 8 --save-every 100000 --lr 1.5e-3 --optim muon --muon-lr 1.5e-3 --muon-wd 0.05 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb "0,1,2,3,26,27,28,29" --lr-table-mult 100 --lr-emb-mult 3
```

`forge kbench` (per-kernel timings of one training step) exists on the branch `kernel-opt` only; after that branch is merged it runs the same way (`forge.exe kbench --reps 10`) and
also picks up the sm_89 image.

## What changed for Windows (branch win-build)

- `crates/forge-core/src/mmap.rs`: a real read-only mapping on Windows (`CreateFileMappingW` + `MapViewOfFile`). Before, every non-Unix target read the whole file into memory, which
  for the training shards (`<dev>\kaggle\stage\forge-tokens` holds 12.7 GB) meant that much RAM and a full read at start-up. Test: `crates/forge-core/tests/mmap.rs`.
- `crates/forge-kernels/build.rs`: the MSVC host compiler is found with `vswhere` when `cl.exe` is not on the PATH; `FORGE_CUDA_ARCHS` builds extra native cubins.
- `crates/forge-kernels/src/lib.rs`: `image_for` prefers an extra cubin built for exactly the device; `FORGE_NO_EXTRA_CUBIN=1` switches it off. CPU tests for both.
- Nothing else was needed: `forge-cuda` already opened `nvcuda.dll` with `LoadLibraryA`/`GetProcAddress` on Windows (all 45 driver entry points it resolves, on main and on
  the three agent branches, are in the export table of this PC's `nvcuda.dll`: `dumpbin /exports`, read from the file), and the crates the trainer uses (forge-core, forge-tok,
  forge-cuda, forge-kernels, forge-train, forge-cli) have no other Linux-only code (no `/proc`, signals, `libc` or shell calls; checkpoints use write-then-rename, which replaces
  the old file on Windows too).

## Known differences and risks (not verified without the GPU)

- **WDDM.** A GeForce card on Windows runs under the display driver model (TCC is not available): kernel launches are queued and batched by the driver and cost more than on Linux,
  and the GPU is shared with the desktop. A training step of the proxy is hundreds of small kernels, so some loss against the same card under Linux is possible; how much is unknown.
  "Hardware-accelerated GPU scheduling" in the Windows graphics settings may help.
- **TDR.** Windows resets the GPU when one kernel runs longer than about 2 s ("Display driver stopped responding and has recovered", `CUDA_ERROR_LAUNCH_TIMEOUT`). The training
  kernels take milliseconds, so this is not expected; if it happens, report it rather than raising `TdrDelay`.
- **VRAM.** The desktop and other programs hold part of the 8 GB, unlike the headless homelab card. Out of memory: close GPU programs or lower `--micro` (the tokens per step stay
  the same, the micro-steps per step grow).
- **Compiler.** The homelab builds with CUDA 13.2, this PC with CUDA 12.9: same source, different compiler. The build prints `warning #221-D: floating-point value does not fit in
  required floating-point type` for every `-INFINITY` in the kernels, because the Windows CRT defines `INFINITY` as `(float)1e+300`. It is harmless: the generated PTX holds the
  exact constant -inf (`0fFF800000`, 191 times) and no clamped value (no `0fFF7FFFFF`).
- **Untested images.** The sm_80 cubin has run on an A100 (Modal smoke tests); the sm_89 cubin has never run anywhere. The first benchmark run plus the A/B above is the check.
  The two images are not the same code (`cuobjdump --dump-resource-usage`): the GEMMs and the attention backward have the same register counts, but for sm_89 ptxas gives the
  training flash forward more registers (`flash_fwd_train_64` 128 -> 142, `flash_fwd_train_128` 163 -> 172) and several element-wise kernels 4-8 more; neither spills. Which one
  is faster on the 4060 is unknown until measured.

## Troubleshooting

| Message | Fix |
|---|---|
| `nvcc fatal : Cannot find compiler 'cl.exe' in PATH` | install Visual Studio 2022 Build Tools with "Desktop development with C++", or build with the script |
| `unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported!` | a Visual Studio 2026 prompt is active: use the script (it loads VS 2022) or a VS 2022 prompt |
| `cannot open the CUDA driver library (libcuda.so.1 / nvcuda.dll): no NVIDIA driver?` | no NVIDIA driver installed |
| `CUDA error in cuMemAlloc...: CUDA_ERROR_OUT_OF_MEMORY` | close GPU programs, lower `--micro` |
| the script does not start (execution policy) | run it with `powershell -ExecutionPolicy Bypass -File ...` as above |
