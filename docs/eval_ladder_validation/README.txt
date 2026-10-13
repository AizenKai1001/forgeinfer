Validation runs of the Forge-1 evaluation ladder (2026-10-10, helper). Research note: "ForgeInfer - Evaluation Ladder".
All numbers were produced on CPU (homelab, 3 threads, nice 19; the GPU was not used) with tools/f1_eval_hf_ref.py on the ladder files in /mnt/data/forgeinfer-data/eval (MANIFEST.json, seed 20261010).

gpt2_a.json, gpt2_b.json, gpt2_c.json      GPT-2 124M (openai-community/gpt2), fp32: winogrande openbookqa arc_easy arc_challenge piqa lambada_openai (n 1000, openbookqa 500), sciq mmlu_cloze (n 500), hellaswag (n 1000)
pythia160_a.json .. pythia160_c.json       Pythia-160M (EleutherAI/pythia-160m), fp32: winogrande arc_easy piqa arc_challenge (full sets), lambada_openai sciq (n 1000), openbookqa hellaswag (n 500)
gpt2_ppl.json, gpt2_ppl_fixed.json, pythia160_ppl.json
                                           bits per byte of the two models on the ladder's 7 domain files, 64 windows of 513 Forge-1 tokens each, decoded to text and rescored with the model's own tokenizer
                                           (gpt2_ppl.json is the first run: windows longer than 1024 GPT-2 tokens were counted with the bytes of the full window, which understates python and books;
                                           gpt2_ppl_fixed.json counts only the bytes of the tokens that were read: use the fixed file)
f125_expected_a.json, f125_expected_b.json  runs/f125_muon_v1 (plain structure, step 5600, tokenizer32k.json) exported with `forge export-hf` and scored with --tokenizer-file tokenizer32k.json, n 300 per task:
                                           the numbers the Rust evaluators must reproduce on the GPU (see the note, section 7)
pythia_published_values_from_repo.json     the published harness values of Pythia-160M at steps 1000, 3000, 13000, 143000 (EleutherAI/pythia, evals/pythia-v1/pythia-160m/zero-shot), extracted, with the sha256 of the source files
opt-125m_published_pythia_repo.json        evals/opt/opt-125m.json of the same repository, unchanged (sha256 0a8128197147...)

Commands (homelab, venv ~/forgeinfer/venv with CPU torch):
  OMP_NUM_THREADS=3 nice -n 19 ionice -c2 -n7 python tools/f1_eval_hf_ref.py MODEL_DIR --threads 3 --tasks winogrande,arc_easy,piqa,arc_challenge --limit 3000 --out pythia160_a.json
  python tools/f1_eval_hf_ref.py MODEL_DIR --threads 3 --ppl --ppl-windows 64 --out pythia160_ppl.json
  python tools/f1_eval_hf_ref.py /tmp/f1_test/f125_hf --tokenizer-file ~/forgeinfer/data/tokenizer32k.json --tasks winogrande,openbookqa,arc_easy,arc_challenge,piqa,sciq,mmlu_cloze,lambada_openai --limit 300 --out f125_expected_a.json

jax_f1_v1_step2000_tracker_subsets.txt     tools/f1_eval_jax.py (JAX model of tpu/forge_jax.py, fp32, CPU) scoring the real f1_v1 checkpoint of step 2000 (scratch/v1_ckpt.bin) on the first 1000 / 500 lines of the OLD unshuffled files
                                           that tools/f1_track.sh used (the Rust tracker's step-2200 block: HellaSwag 31.9 / 36.6 / 34.5, ARC-E 39.0 / 36.8, PIQA 57.2 / 57.8, WinoGrande 49.6)
jax_f1_v1_step2000_ladder_subsets.txt      the same checkpoint on the first N lines of the shuffled ladder files (winogrande 1000, openbookqa 500, arc_easy 1000, arc_challenge 600, piqa 1000, sciq 300, mmlu_cloze 500; the HellaSwag part was stopped unfinished at the time limit)
  steps: python tools/f1_eval_jax.py prep --tokenizer tokenizer_f1.json --seq 1024 --tasks NAME:LIMIT:FILE,... --out seqs.json   (system Python with `tokenizers`)
         tpu/venv/Scripts/python.exe tools/f1_eval_jax.py score --ckpt scratch/v1_ckpt.bin --seqs seqs.json                          (JAX CPU, 3 cores, BelowNormal priority)
