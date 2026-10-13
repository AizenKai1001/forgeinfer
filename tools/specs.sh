#!/bin/bash
# Run every executable specification on the homelab and print one line per spec (the regression suite for the oracles and goldens themselves).
#   tools/specs.sh
cd "$(dirname "$0")/.." || exit 1
bash tools/push.sh "${COMMIT_MESSAGE:-specs}" >/dev/null 2>&1
exec bash tools/hl.sh 'PY=~/forgeinfer/venv/bin/python
echo "== forward (Llama/Qwen3)";  $PY tools/ref_forward_numpy.py smollm2 | tail -n 2; $PY tools/ref_forward_numpy.py qwen3 | tail -n 2
echo "== Qwen3.5";                $PY tools/ref_qwen35_numpy.py | tail -n 1
echo "== tokenizers";             for m in smollm2 qwen3 qwen35; do python3 tools/ref_tokenizer.py $m | tail -n 1; done
echo "== GGUF + quant";           $PY tools/ref_gguf.py | tail -n 5
echo "== backward";               $PY tools/ref_backward_numpy.py | tail -n 2
echo "== flash attention";        $PY tools/ref_flash_numpy.py | tail -n 1
echo "== sampling";               $PY tools/ref_sampling_numpy.py | tail -n 1'
