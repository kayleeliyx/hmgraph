#!/usr/bin/env bash
set -euo pipefail
cd /data/yaxuanli/ActGround
export PYTHONDONTWRITEBYTECODE=1
TASK_ENV=/data/yaxuanli/ActGround/.venvs/hdepic-qwen35-35b
TASK_RUN=/data/yaxuanli/ActGround/runs/hdepic_qwen35_35b_baseline_v1
mkdir -p "$TASK_RUN/environment"
if [ ! -x "$TASK_ENV/bin/python" ]; then
  /usr/bin/python3.12 -m venv --copies "$TASK_ENV"
fi
# Copy the working CPython3.12 CUDA runtime into the isolated environment.
# No shared package search path and no modification of the original SFT env.
"$TASK_ENV/bin/python" - <<'PY'
from pathlib import Path
import shutil
src=Path('/data/yaxuanli/ActGround/.venvs/hdepic-swift-8b/lib/python3.12/site-packages')
dst=Path('/data/yaxuanli/ActGround/.venvs/hdepic-qwen35-35b/lib/python3.12/site-packages')
names=('torch','torchgen','torchvision','torchvision.libs','nvidia','cusparselt','triton','sympy','mpmath','networkx','jinja2','markupsafe','filelock','fsspec','typing_extensions.py')
prefixes=('torch-','torchvision-','nvidia_','triton-','sympy-','mpmath-','networkx-','jinja2-','markupsafe-','filelock-','fsspec-','typing_extensions-')
for p in src.iterdir():
 if p.name in names or (p.name.endswith('.dist-info') and p.name.startswith(prefixes)):
  q=dst/p.name
  if q.exists():continue
  if p.is_dir():shutil.copytree(p,q)
  else:shutil.copy2(p,q)
PY
export PIP_CACHE_DIR=/data/yaxuanli/ActGround/.cache/pip-qwen35
"$TASK_ENV/bin/python" -m pip install \
  'transformers==5.3.0' 'accelerate==1.12.0' \
  'huggingface-hub==1.5.0' 'tokenizers==0.22.2' \
  'safetensors==0.7.0' 'numpy==1.26.4' 'Pillow==11.1.0'
"$TASK_ENV/bin/python" -m pip check
"$TASK_ENV/bin/python" -m pip freeze > "$TASK_RUN/environment/packages.txt"
"$TASK_ENV/bin/python" - <<'PY'
import sys,json,importlib.metadata
from pathlib import Path
import torch,transformers
from transformers import Qwen3_5MoeForConditionalGeneration,AutoProcessor
p=Path('/data/yaxuanli/ActGround/runs/hdepic_qwen35_35b_baseline_v1/environment')
d={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
   'transformers':transformers.__version__,'qwen35_moe_native_import':True,
   'packages':{x:importlib.metadata.version(x) for x in ('torch','torchvision','transformers','accelerate','huggingface-hub','tokenizers','safetensors','numpy','Pillow')},
   'optional_fla_installed':False,'system_site_packages':False,
   'runtime_source':'Independent copy of existing CPython3.12 Torch2.6 CUDA12.4 distributions; new pinned HF packages',
   'gpu_model_load_performed':False}
(p/'compatibility.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
PY
