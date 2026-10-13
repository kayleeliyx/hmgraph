#!/usr/bin/env bash
set -euo pipefail
cd /data/yaxuanli/ActGround
source /data/yaxuanli/ActGround/scripts/hdepic_data_storage.sh
/usr/bin/python3 -B -c 'from code.hdepic_sft_qwen38.storage import validate_storage; validate_storage(require_space=True)'
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK_ENV=/tmp/hdepic-qwen38-27b-env
TASK_RUN=/data/yaxuanli/ActGround/runs/hdepic_qwen38_27b_baseline_v1
mkdir -p "$TASK_RUN/environment"
if [ ! -x "$TASK_ENV/bin/python" ]; then
  /usr/bin/python3.12 -m venv --copies "$TASK_ENV"
fi
"$TASK_ENV/bin/python" - <<'PY'
from pathlib import Path
import shutil
src=Path('/data/yaxuanli/ActGround/.venvs/hdepic-qwen35-35b/lib/python3.12/site-packages')
dst=Path('/tmp/hdepic-qwen38-27b-env/lib/python3.12/site-packages')
# Independent files, no shared search paths, no package changes in source env.
for p in src.iterdir():
    if p.name == '__pycache__' or p.name == 'pip' or p.name.startswith('pip-'):
        continue
    q=dst/p.name
    if q.exists():
        continue
    if p.is_dir():
        shutil.copytree(p,q,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:
        shutil.copy2(p,q)
PY
"$TASK_ENV/bin/python" -m pip check
"$TASK_ENV/bin/python" -m pip freeze > "$TASK_RUN/environment/packages.txt"
"$TASK_ENV/bin/python" - <<'PY'
import hashlib, importlib.metadata, inspect, json, sys
from pathlib import Path
import torch, transformers
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5RMSNormGated
run=Path('/data/yaxuanli/ActGround/runs/hdepic_qwen38_27b_baseline_v1')
metadata=run/'model_metadata'
config=AutoConfig.from_pretrained(metadata,local_files_only=True,trust_remote_code=False)
assert config.model_type == 'qwen3_5'
assert config.architectures == ['Qwen3_5ForConditionalGeneration']
assert transformers.__version__ == '5.3.0'
assert torch.__version__ == '2.6.0+cu124'
processor=AutoProcessor.from_pretrained(metadata,local_files_only=True,trust_remote_code=False,use_fast=True,min_pixels=65536,max_pixels=147456)
assert type(processor).__name__=='Qwen3VLProcessor'
assert type(processor.image_processor).__name__=='Qwen2VLImageProcessorFast'
assert processor.image_processor.size=={'shortest_edge':65536,'longest_edge':147456}
messages=[{'role':'system','content':[{'type':'text','text':'Frozen test system.'}]},{'role':'user','content':[{'type':'text','text':'Frozen test question.'}]}]
rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False,add_vision_id=False)
expected='<|im_start|>system\nFrozen test system.<|im_end|>\n<|im_start|>user\nFrozen test question.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
assert rendered==expected
with torch.device('meta'):
    model=Qwen3_5ForConditionalGeneration(config)
state=model.state_dict()
index=json.loads((metadata/'model.safetensors.index.json').read_text())
missing=sorted(set(state)-set(index['weight_map']))
extras=sorted(set(index['weight_map'])-set(state))
assert not missing,missing
assert all(x.startswith('mtp.') for x in extras),extras
attn_source=inspect.getsource(Qwen3_5Attention.forward)
norm_source=inspect.getsource(Qwen3_5RMSNormGated)
assert 'torch.sigmoid(gate)' in attn_source
assert 'F.silu' in norm_source
source_root=Path('/data/yaxuanli/ActGround/.venvs/hdepic-qwen35-35b/lib/python3.12/site-packages')
envroot=Path('/tmp/hdepic-qwen38-27b-env/lib/python3.12/site-packages')
assert not any(str(source_root) in p for p in sys.path)
files={}
for rel in ['transformers/models/qwen3_5/modeling_qwen3_5.py','transformers/models/qwen3_5/configuration_qwen3_5.py','torch/__init__.py']:
    old=source_root/rel;new=envroot/rel
    old_sha=hashlib.sha256(old.read_bytes()).hexdigest();new_sha=hashlib.sha256(new.read_bytes()).hexdigest()
    assert old_sha==new_sha
    assert (old.stat().st_dev,old.stat().st_ino)!=(new.stat().st_dev,new.stat().st_ino)
    files[rel]={'sha256':new_sha,'independent_inode':True}
proof={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,
 'qwen38_dense_native_import':True,'packages':{x:importlib.metadata.version(x) for x in ('torch','torchvision','transformers','accelerate','huggingface-hub','tokenizers','safetensors','numpy','Pillow')},
 'optional_fla_installed':False,'system_site_packages':False,'source_environment_unchanged':True,
 'runtime_source':'Independent copy of the verified Qwen3.5 environment; no dependency upgrades or modeling patches.',
 'environment_root':'/tmp/hdepic-qwen38-27b-env','environment_is_ephemeral':True,
 'source_runtime_files':files,'native_class':'Qwen3_5ForConditionalGeneration','model_type':config.model_type,
 'processor_class':type(processor).__name__,'image_processor_class':type(processor.image_processor).__name__,
 'nonthinking_template_exact':True,'nonthinking_template_sample':rendered,
 'meta_architecture_audit':{'device':'meta','real_weight_bytes_allocated':0,'parameter_count':sum(p.numel() for p in model.parameters()),'native_state_keys':len(state),'checkpoint_index_keys':len(index['weight_map']),'missing_keys':missing,'extra_mtp_keys':extras,'all_parameters_meta':all(p.device.type=='meta' for p in model.parameters())},
 'gate_handling':{'full_attention':'Native official sigmoid gate','linear_attention_rmsnorm':'Native official SiLU/Swish gate','manual_override':False,'config_output_gate_type':config.text_config.output_gate_type,'caution':'The output_gate_type export attribute does not establish that the full-attention sigmoid should be replaced. Official Transformers keeps sigmoid there; no hand patch is applied.'},
 'gpu_model_load_performed':False,'model_inference_performed':False,'gpu_accessed':False,
 'remaining_gate':'Actual two-GPU loading and image-conditioned inference smoke must pass before all 228 answers.'}
path=run/'environment/compatibility.json'
payload=json.dumps(proof,indent=2)+'\n'
if path.exists():
    assert path.read_text()==payload,'Environment compatibility proof changed'
else:path.write_text(payload)
print(json.dumps(proof,indent=2))
PY
