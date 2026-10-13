"""Stage the Kaggle datasets and the script kernel for the Forge-1 sweeps (nothing is uploaded here; <dev>\\kaggle\\kg.sh does the calls with the API token, account YOUR_KAGGLE_USER).

    python cloud/kaggle_stage.py bin --binary scratch/forge-linux                  # <dev>\\kaggle\\stage\\forge-bin: the Linux build of `forge` (built on Ubuntu 22.04 inside the Modal image)
    python cloud/kaggle_stage.py job --arms round1 [--max-hours 4]                  # <dev>\\kaggle\\stage\\forge-job: settings.json with the arms of a sweep (see ARMS below)
    python cloud/kaggle_stage.py kernel [--name forge-ab]                           # <dev>\\kaggle\\stage\\forge-ab-kernel: kernel-metadata.json + cloud/kaggle_kernel.py
The token shards go by hand into <dev>\\kaggle\\stage\\forge-data-web (dataset id YOUR_KAGGLE_USER/forge-data-web).
Then:  bash <dev>/kaggle/kg.sh datasets create -p <folder>   (first time) or   datasets version -p <folder> -m "text"      and      kernels push -p <kernel folder>
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(r"<dev>\kaggle\stage")
OWNER = "YOUR_KAGGLE_USER"

# the 31M proxy of tools/f1_ab.sh (12 layers, d 384, 6/2 heads, ffn 1024, vocab 32768, seq 2048), 500 steps x 65,536 tokens, Muon 2e-3
PROXY = ("--layers 12 --dim 384 --heads 6 --kv 2 --ffn 1024 --seq 2048 --micro 4 --tokens-per-step 65536 --steps 500 --warmup 30 --eval-every 100 --eval-batches 8 --save-every 100000 "
         "--lr 2e-3 --optim muon --muon-lr 2e-3 --muon-wd 0.05 --qk-norm")
ARMS = {
    # the arms of round 1 on the homelab (tools/f1_round1.sh): every option alone, then all together
    "round1": [
        ("base", ""), ("base_s2", "--seed 2"), ("relu2", "--relu2 --ffn 1536"), ("untied", "--untied"), ("lambdas", "--lambdas"), ("skips", "--skips"),
        ("ve4", "--value-emb 0,1,10,11"), ("softcap20", "--softcap 20"), ("all", "--untied --relu2 --ffn 1536 --softcap 20 --lambdas --skips --value-emb 0,1,10,11"),
    ],
    # round 2a (does not depend on the results of round 1): the n-gram memory, learning-rate multipliers of the tables and scalars, a third seed for the noise floor
    "round2a": [
        ("base_s3", "--seed 3"),
        ("ngram23_b17", "--ngram 2,3 --ngram-bits 17 --ngram-dim 128"),
        ("ngram2_b17", "--ngram 2 --ngram-bits 17 --ngram-dim 128"),
        ("ngram23_b15", "--ngram 2,3 --ngram-bits 15 --ngram-dim 64"),
        ("ve4_t5", "--value-emb 0,1,10,11 --lr-table-mult 5"),
        ("ve4_t20", "--value-emb 0,1,10,11 --lr-table-mult 20"),
        ("lambdas_s10", "--lambdas --lr-scalar-mult 10"),
        ("skips_s10", "--skips --lr-scalar-mult 10"),
        ("untied_t3", "--untied --lr-table-mult 3"),
        ("base_e3", "--lr-emb-mult 3"),
    ],
    # round 2b: ReLU squared at the parameter-matched width (the round-1 arm kept ffn 1024: 26.7 M parameters) and every other option on top of it
    "round2b": [
        ("relu2_1536", "--relu2 --ffn 1536"),
        ("relu2_2048", "--relu2 --ffn 2048"),
        ("r2_untied_t3", "--relu2 --ffn 1536 --untied --lr-table-mult 3"),
        ("r2_lambdas_s10", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10"),
        ("r2_skips_s10", "--relu2 --ffn 1536 --skips --lr-scalar-mult 10"),
        ("r2_ve4_t5", "--relu2 --ffn 1536 --value-emb 0,1,10,11 --lr-table-mult 5"),
        ("r2_ngram23", "--relu2 --ffn 1536 --ngram 2,3 --ngram-bits 17 --ngram-dim 128"),
        ("r2_softcap20", "--relu2 --ffn 1536 --softcap 20"),
    ],
    # round 3 (optimizer and schedule on the plain proxy with ReLU squared, ffn 1536): the learning rates of Muon and AdamW, Muon's weight decay, Adam's beta2, the warmup
    "round3": [
        ("h_muon1e3", "--relu2 --ffn 1536 --muon-lr 1e-3"),
        ("h_muon3e3", "--relu2 --ffn 1536 --muon-lr 3e-3"),
        ("h_muon5e3", "--relu2 --ffn 1536 --muon-lr 5e-3"),
        ("h_adam1e3", "--relu2 --ffn 1536 --lr 1e-3"),
        ("h_adam5e3", "--relu2 --ffn 1536 --lr 5e-3"),
        ("h_mwd0", "--relu2 --ffn 1536 --muon-wd 0.0"),
        ("h_mwd2", "--relu2 --ffn 1536 --muon-wd 0.2"),
        ("h_warm100", "--relu2 --ffn 1536 --warmup 100"),
    ],
    # round 5 (after round 2: value embeddings with a 20x table learning rate -0.065 and a 3x embedding learning rate -0.043 were the biggest effects): how far do the multipliers go, and do the gains add up?
    "round5": [
        ("e10", "--lr-emb-mult 10"),
        ("e30", "--lr-emb-mult 30"),
        ("ve4_t50", "--value-emb 0,1,10,11 --lr-table-mult 50"),
        ("ve4_t100", "--value-emb 0,1,10,11 --lr-table-mult 100"),
        ("comboA", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,10,11 --lr-table-mult 20 --lr-emb-mult 3"),
        ("comboB", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --skips --value-emb 0,1,10,11 --lr-table-mult 20 --lr-emb-mult 3"),
        ("comboC", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,10,11 --ngram 2,3 --ngram-bits 17 --ngram-dim 128 --lr-table-mult 20 --lr-emb-mult 3"),
        ("ngram23_t20", "--ngram 2,3 --ngram-bits 17 --ngram-dim 128 --lr-table-mult 20"),
    ],
    # round 7 (after round 5: value embeddings with a table learning rate of 5, 20, 50, 100 times gave -0.026, -0.065, -0.075, -0.101): how far does it go, and how many layers want a table?
    "round7": [
        ("ve4_t200", "--value-emb 0,1,10,11 --lr-table-mult 200"),
        ("ve4_t400", "--value-emb 0,1,10,11 --lr-table-mult 400"),
        ("ve8_t100", "--value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100"),
        ("ve12_t100", "--value-emb 0,1,2,3,4,5,6,7,8,9,10,11 --lr-table-mult 100"),
        ("ve2_t100", "--value-emb 0,11 --lr-table-mult 100"),
        ("ve4last_t100", "--value-emb 8,9,10,11 --lr-table-mult 100"),
        ("ve4_t100_e3", "--value-emb 0,1,10,11 --lr-table-mult 100 --lr-emb-mult 3"),
        ("ve4_t100_s2", "--value-emb 0,1,10,11 --lr-table-mult 100 --seed 2"),
    ],
    # round 8 (2000 steps, while round 4 runs): more value-embedding layers, a higher table rate, and seeds of two round-4 arms to know the noise at this length
    "round8": [
        ("L8_t200", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 200 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L12_t100", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,4,5,6,7,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L_base_s2", "--seed 2 --steps 2000 --warmup 60"),
        ("L_comboA100_s2", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,10,11 --lr-table-mult 100 --lr-emb-mult 3 --seed 2 --steps 2000 --warmup 60"),
    ],
    # round 9 (after round 4: at 2000 steps the 100x table rate no longer beats 20x with four layers, the gain of the recipe is -0.074 for v1's): is it the number of layers that helps, and which rate?
    "round9": [
        ("L8_t20", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 20 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L8_t50", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 50 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L4_t50", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,10,11 --lr-table-mult 50 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L_comboA20_s2", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,10,11 --lr-table-mult 20 --lr-emb-mult 3 --seed 2 --steps 2000 --warmup 60"),
    ],
    # round 10 (efficiency directive, needs no new binary): the cheapest sample-efficiency test is a smaller batch at equal tokens (more optimizer steps for the same GPU time); the reference is the best 2000-step recipe
    # L8_combo (value embeddings in 8 of 12 layers, table rate 100x: 3.5986 on Lightning, seed noise about 0.003)
    "round10": [
        ("L8_combo_s2", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --seed 2 --steps 2000 --warmup 60"),
        ("L8_bs32", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --tokens-per-step 32768 --steps 4000 --warmup 120"),
        ("L8_bs16", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --tokens-per-step 16384 --steps 8000 --warmup 240"),
        ("L8_bs32_lr07", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --tokens-per-step 32768 --steps 4000 --warmup 120 --lr 1.4e-3 --muon-lr 1.4e-3"),
    ],
    # rounds 11 and 12 (binary of the recipe-opt branch, dataset forge-bin2): the sample-efficiency options of the recipe-opt agent on the best 2000-step proxy recipe (L8_combo); every arm also keeps a parameter average
    # (--ema 0.98 --ema-every 2: it never changes the training, it adds a val_ema column for free). The options are off by default, so RL_base is the same run as L8_combo (3.5986, seed 1) with the new binary.
    "round11": [
        ("RL_base", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2"),
        ("RL_seq", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --seq-schedule 512:600,1024:1200"),
        ("RL_normuon", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --normuon"),
        ("RL_docmask", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --doc-mask"),
        ("RL_peri", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --peri-norm --emb-norm"),
        ("RL_stack", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --seq-schedule 512:600,1024:1200 --normuon --doc-mask --muon-polar --cautious-wd"),
    ],
    "round12": [
        ("RL_sqrt", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --decay-shape sqrt"),
        ("RL_polar", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --muon-polar"),
        ("RL_cwd", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --cautious-wd --cautious-wd-adam --muon-wd 0.1"),
        ("RL_gate12", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --attn-gate 12"),
        ("RL_base_s2", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --seed 2"),
        ("RL_gate384", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --attn-gate 384"),
    ],
    # round 13 (binary of the kernel-opt branch, dataset forge-bin3): sliding-window attention in 3 of every 4 layers (--window W --window-layers auto: the 4th layer and the last one stay dense) on the best 2000-step
    # proxy recipe. It makes attention cheaper (46 % of dense at W = 512, 2048 tokens) and the step 7 to 9 % faster; the question is what it costs in loss at equal tokens. RW_base is the same run as L8_combo
    # (3.5986 seed 1) with the kernel-opt build: it also checks that the faster kernels train to the same loss.
    "round13": [
        ("RW_base", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("RW_win512", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --window 512 --window-layers auto"),
        ("RW_win1024", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --window 1024 --window-layers auto"),
        ("RW_win256", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --window 256 --window-layers auto"),
    ],
    # round 14 (after round 10: at equal tokens a SMALLER batch was worse, 32k +0.010, 16k +0.075 against 64k, and a 0.7x learning rate did not matter): is a LARGER batch better still? 128k and 256k tokens per
    # step (1000 and 500 steps; warmup scaled with the step count), at the learning rate of the 64k recipe and 1.5x higher
    "round14": [
        ("L8_bs128", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --tokens-per-step 131072 --steps 1000 --warmup 30"),
        ("L8_bs128_lr3", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --tokens-per-step 131072 --steps 1000 --warmup 30 --lr 3e-3 --muon-lr 3e-3"),
        ("L8_bs256", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --tokens-per-step 262144 --steps 500 --warmup 15"),
        ("L8_bs256_lr3", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --tokens-per-step 262144 --steps 500 --warmup 15 --lr 3e-3 --muon-lr 3e-3"),
    ],
    # round 12b (after round 11: seq schedule -0.014 and NorMuon -0.008 against 3.5992, Peri-LN +0.002, document mask not comparable (masked val 3.5904, unmasked 3.6223); binary forge-bin4 = the
    # production build, which has every option and the window attention): the decay shape (it matters for the decay of the real run at step 40,500), a second seed of the base, the two winners together,
    # the two winners with a window of 512 (a free speedup at the proxy scale), Polar Express and cautious decay
    "round12b": [
        ("RL_sqrt", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --decay-shape sqrt"),
        ("RL_base_s2", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --seed 2"),
        ("RL_seq_normuon", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --seq-schedule 512:600,1024:1200 --normuon"),
        ("RL_sn_win512", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --seq-schedule 512:600,1024:1200 --normuon --window 512 --window-layers auto"),
        ("RL_polar", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --muon-polar"),
        ("RL_cwd", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,8,9,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60 --ema 0.98 --ema-every 2 --cautious-wd --cautious-wd-adam --muon-wd 0.1"),
    ],
    # round 15 (the real 150M model, data forge-tokens, binary forge-bin4): does the proxy's gain of the sequence schedule + NorMuon (+ window attention) hold on the real model? The arms use the flags of the Kaggle chain hop 1
    # (v2 recipe, dense attention, seed 1, 65,536 tokens per step, constant learning rate until step 40,500; hop 1's log is the dense reference at the same data and steps: hop 1 evaluated 32 windows (micro 4 x --eval-batches 8, rank 0 only; its val 3.1691 at step 2000),
    # these arms evaluate 64 (micro 4 x 16): compare with hop 1 by the paired mean training loss of steps >= 1200 (tools/hop_compare.py), not by val (+0.062 for the window sets alone).
    # One arm per GPU, stopped by the time limit (8000 s in the staged settings; 10800 here) after about step 2000. A leading "!" = a complete command line.
    "round15": [
        ("R150_seq_normuon_win512", "!--layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --micro 4 --tokens-per-step 65536 --steps 45000 --warmup 100 --decay-frac 0.1 --min-lr-frac 0.05 --eval-every 100 --eval-batches 16 --save-every 100000 --qk-norm --optim muon --lr 1.5e-3 --muon-lr 1.5e-3 --muon-wd 0.05 --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,26,27,28,29 --lr-table-mult 100 --lr-emb-mult 3 --ema 0.98 --ema-every 10 --seq-schedule 512:600,1024:1200 --normuon --window 512 --window-layers auto --time-limit-s 10800"),
        ("R150_seq_normuon", "!--layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --micro 4 --tokens-per-step 65536 --steps 45000 --warmup 100 --decay-frac 0.1 --min-lr-frac 0.05 --eval-every 100 --eval-batches 16 --save-every 100000 --qk-norm --optim muon --lr 1.5e-3 --muon-lr 1.5e-3 --muon-wd 0.05 --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,26,27,28,29 --lr-table-mult 100 --lr-emb-mult 3 --ema 0.98 --ema-every 10 --seq-schedule 512:600,1024:1200 --normuon --time-limit-s 10800"),
    ],
    # round 6: seeds of the two big effects of round 2 (value embeddings with a 20x table rate, a 3x embedding rate) and of the baseline, to know how much of them is luck
    "round6": [
        ("ve4_t20_s2", "--value-emb 0,1,10,11 --lr-table-mult 20 --seed 2"),
        ("ve4_t20_s3", "--value-emb 0,1,10,11 --lr-table-mult 20 --seed 3"),
        ("base_e3_s2", "--lr-emb-mult 3 --seed 2"),
        ("base_e3_s3", "--lr-emb-mult 3 --seed 3"),
        ("base_s4", "--seed 4"),
        ("base_s5", "--seed 5"),
    ],
    # round 4 (redefined after round 5): the recipe of the running Forge-1 v1 on a 4x longer schedule (2000 steps = 131 M tokens): does the gain of the 500-step runs persist, and is a 100x table learning rate
    # still better than 20x? (comboA20 is v1's recipe at proxy size; the value-embedding layers 10,11 stand for v1's 28,29: the last two layers)
    "round4": [
        ("L_base", "--steps 2000 --warmup 60"),
        ("L_e3", "--lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L_comboA20", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,10,11 --lr-table-mult 20 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L_comboA100", "--relu2 --ffn 1536 --lambdas --lr-scalar-mult 10 --value-emb 0,1,10,11 --lr-table-mult 100 --lr-emb-mult 3 --steps 2000 --warmup 60"),
        ("L_ve4_t100", "--value-emb 0,1,10,11 --lr-table-mult 100 --steps 2000 --warmup 60"),
    ],
}


def merge_args(base: str, extra: str) -> str:
    """The command line of an arm: the proxy's flags with every flag that `extra` sets again removed from them (`forge train` takes the FIRST occurrence of a flag, and older binaries must keep working)."""
    import shlex

    def pairs(tokens):
        out, i = [], 0
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("--") and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out.append((t, tokens[i + 1]))
                i += 2
            else:
                out.append((t, None))
                i += 1
        return out

    over = {k for k, _ in pairs(shlex.split(extra))}
    kept = [(k, v) for k, v in pairs(shlex.split(base)) if k not in over]
    return " ".join(" ".join(x for x in (k, v) if x is not None) for k, v in kept + pairs(shlex.split(extra)))


def meta(folder: Path, slug: str, title: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": f"{OWNER}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=1), encoding="utf-8")


def fresh(name: str) -> Path:
    folder = STAGE / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bin", "job", "kernel", "trainjob", "ckpt", "trainkernel"])
    ap.add_argument("--binary")
    ap.add_argument("--bin-slug", default="forge-bin", help="dataset slug of the build the arms run (bin: where it is staged; job: written to settings.json; kernel: attached as a source)")
    ap.add_argument("--arms", default="round1")
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--name", default="forge-ab")
    ap.add_argument("--data", default="forge-data-web")
    ap.add_argument("--job-slug", default="forge-job")
    ap.add_argument("--train-args", default="")
    ap.add_argument("--hours", type=float, default=11.2)
    ap.add_argument("--files", default="")
    ap.add_argument("--from-dir", default="")
    ap.add_argument("--with-ckpt", action="store_true")
    a = ap.parse_args()
    if a.what == "bin":
        out = fresh(a.bin_slug)
        shutil.copy2(a.binary, out / "forge")
        meta(out, a.bin_slug, a.bin_slug.replace("-", " ").title())
        print(out)
    elif a.what == "job":
        out = fresh(a.job_slug)
        arms = [{"name": n, "args": (extra[1:] if extra.startswith("!") else merge_args(PROXY, extra))} for n, extra in ARMS[a.arms]]  # an arm whose text starts with "!" is a complete command line (no proxy base)
        settings = {"data": a.data, "arms": arms, "max_hours": a.max_hours}
        if a.bin_slug != "forge-bin":
            settings["bin"] = a.bin_slug
        (out / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
        meta(out, a.job_slug, a.job_slug.replace("-", " ").title())
        print(out, len(arms), "arms")
    elif a.what == "trainjob":
        out = fresh("forge-trainjob")
        settings = {"args": a.train_args, "data": "forge-tokens", "hours": a.hours}
        if a.files:
            settings["files"] = a.files.split(",")
        (out / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
        meta(out, "forge-trainjob", "Forge trainjob")
        print(out)
    elif a.what == "ckpt":
        out = fresh("forge-ckpt")
        src = Path(a.from_dir)
        shutil.copy2(src / "ckpt_latest.bin", out / "ckpt_latest.bin")
        if (src / "log.jsonl").exists():
            shutil.copy2(src / "log.jsonl", out / "log.jsonl")
        meta(out, "forge-ckpt", "Forge ckpt")
        print(out)
    elif a.what == "trainkernel":
        out = fresh(f"{a.name}-kernel")
        shutil.copy2(ROOT / "cloud" / "kaggle_train.py", out / "kaggle_train.py")
        sources = [f"{OWNER}/forge-bin", f"{OWNER}/forge-tokens", f"{OWNER}/forge-trainjob"] + ([f"{OWNER}/forge-ckpt"] if a.with_ckpt else [])
        (out / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{OWNER}/{a.name}", "title": a.name, "code_file": "kaggle_train.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": sources, "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1), encoding="utf-8")
        print(out)
    else:
        out = fresh(f"{a.name}-kernel")
        shutil.copy2(ROOT / "cloud" / "kaggle_kernel.py", out / "kaggle_kernel.py")
        (out / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{OWNER}/{a.name}", "title": a.name, "code_file": "kaggle_kernel.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [f"{OWNER}/{a.bin_slug}", f"{OWNER}/{a.data}", f"{OWNER}/{a.job_slug}"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
