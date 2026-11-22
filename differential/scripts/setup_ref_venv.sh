#!/bin/bash
# New reference venv: reads torch/transformers from venv-vllm's site-packages (read-only .pth), plus its own small extras. Engine venvs are NOT modified.
source /root/kvc/scripts/common.sh
uv venv --python 3.12 $KVC/venv-ref
echo "$KVC/venv-vllm/lib/python3.12/site-packages" > $KVC/venv-ref/lib/python3.12/site-packages/vllm_site.pth
source $KVC/venv-ref/bin/activate
uv pip install jsonschema scipy 2>&1 | tail -2
python -c "import torch,transformers,jsonschema;print('torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available())"
