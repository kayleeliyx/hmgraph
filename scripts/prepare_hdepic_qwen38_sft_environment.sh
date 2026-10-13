#!/usr/bin/env bash
# Independent, pinned native Transformers/PEFT environment; CPU/meta checks only.
set -euo pipefail
cd /data/yaxuanli/ActGround
source /data/yaxuanli/ActGround/scripts/hdepic_data_storage.sh
# Frozen setup provenance is retained. Validate the relocated legacy paths
# before any historical command can create a directory or cache.
/usr/bin/python3 -B -c 'from code.hdepic_sft_qwen38.storage import validate_storage; validate_storage(require_space=True)'
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK_ENV=/tmp/hdepic-qwen38-sft-env
TASK_SOURCE_ENV=/tmp/hdepic-qwen38-27b-env
TASK_RUN=/data/yaxuanli/ActGround/runs/hdepic_sft_qwen38_27b_v1
mkdir -p "$TASK_RUN/environment"
if [ ! -x "$TASK_ENV/bin/python" ]; then
  /usr/bin/python3.12 -m venv --copies "$TASK_ENV"
fi
"$TASK_ENV/bin/python" - <<'PY'
from pathlib import Path
import hashlib, shutil, urllib.request
src=Path('/tmp/hdepic-qwen38-27b-env/lib/python3.12/site-packages')
dst=Path('/tmp/hdepic-qwen38-sft-env/lib/python3.12/site-packages')
assert src.is_dir()
for p in src.iterdir():
    if p.name in ('__pycache__','pip') or p.name.startswith('pip-'):
        continue
    q=dst/p.name
    if q.exists():
        continue
    if p.is_dir():
        shutil.copytree(p,q,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:
        shutil.copy2(p,q)
wheel=Path('/data/yaxuanli/ActGround/runs/hdepic_sft_qwen38_27b_v1/environment/peft-0.18.1-py3-none-any.whl')
url='https://files.pythonhosted.org/packages/b3/14/b4e3f574acf349ae6f61f9c000a77f97a3b315b4bb6ad03791e79ae4a568/peft-0.18.1-py3-none-any.whl'
expected='0bf06847a3551e3019fc58c440cffc9a6b73e6e2962c95b52e224f77bbdb50f1'
if not wheel.exists():
    data=urllib.request.urlopen(url,timeout=60).read()
    assert hashlib.sha256(data).hexdigest()==expected
    wheel.write_bytes(data)
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==expected
PY
"$TASK_ENV/bin/python" -m pip install --no-deps --no-index "$TASK_RUN/environment/peft-0.18.1-py3-none-any.whl"
"$TASK_ENV/bin/python" -m pip check
"$TASK_ENV/bin/python" -m pip freeze > "$TASK_RUN/environment/packages.txt"
"$TASK_ENV/bin/python" - <<'PY'
import collections, hashlib, importlib.metadata as md, inspect, json, re, sys
from pathlib import Path
import accelerate, peft, torch, transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
run=Path('/data/yaxuanli/ActGround/runs/hdepic_sft_qwen38_27b_v1')
src=Path('/tmp/hdepic-qwen38-27b-env/lib/python3.12/site-packages')
dst=Path('/tmp/hdepic-qwen38-sft-env/lib/python3.12/site-packages')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def immutable_json(path,data):
    payload=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if path.exists():assert path.read_text()==payload,f'Changed proof: {path}'
    else:path.write_text(payload)
assert torch.__version__=='2.6.0+cu124'
assert transformers.__version__=='5.3.0'
assert accelerate.__version__=='1.12.0'
assert peft.__version__=='0.18.1'
assert not any(str(src) in p for p in sys.path)
source_versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions(path=[str(src)])}
target_versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions(path=[str(dst)])}
for name,version in source_versions.items():
    if name!='pip':assert target_versions.get(name)==version,(name,version,target_versions.get(name))
assert set(target_versions)-set(source_versions)=={'peft'}
copy_audit={}
for rel in ['torch/__init__.py','transformers/models/qwen3_5/modeling_qwen3_5.py','accelerate/accelerator.py']:
    old,new=src/rel,dst/rel
    assert sha(old)==sha(new)
    assert (old.stat().st_dev,old.stat().st_ino)!=(new.stat().st_dev,new.stat().st_ino)
    copy_audit[rel]={'sha256':sha(new),'independent_file':True}
metadata=Path('/data/yaxuanli/ActGround/runs/hdepic_qwen38_27b_baseline_v1/model_metadata')
config=AutoConfig.from_pretrained(metadata,local_files_only=True,trust_remote_code=False)
assert config.model_type=='qwen3_5'
config.use_cache=False
config.text_config.use_cache=False
config._attn_implementation='sdpa'
target_regex=r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))'
with torch.device('meta'):
    model=Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    base_parameter_count=sum(p.numel() for p in model.parameters())
    candidates=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear) and re.fullmatch(target_regex,name)]
    all_text_linears=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear) and name.startswith('model.language_model.layers.')]
    assert sorted(candidates)==sorted(all_text_linears)
    assert len(candidates)==496
    lora_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias='none',target_modules=target_regex,task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
trainable=[{'name':n,'shape':list(p.shape),'numel':p.numel(),'dtype':str(p.dtype)} for n,p in model.named_parameters() if p.requires_grad]
assert len(trainable)==992
assert all('.lora_' in p['name'] and '.model.language_model.layers.' in p['name'] for p in trainable)
assert not any('visual' in p['name'] or 'merger' in p['name'] or 'lm_head' in p['name'] or 'embed_tokens' in p['name'] for p in trainable)
assert all(p.device.type=='meta' for p in model.parameters())
assert model.is_gradient_checkpointing
assert all(not p.requires_grad for n,p in model.named_parameters() if '.lora_' not in n)
assert not torch.cuda.is_initialized()
immutable_json(run/'environment/trainable_parameters_meta.json',trainable)
immutable_json(run/'environment/text_lora_target_modules.json',candidates)
native_forward_signature=str(inspect.signature(Qwen3_5ForConditionalGeneration.forward))
proof={
 'python':sys.version,'executable':sys.executable,'environment_root':str(dst.parents[2]),'environment_ephemeral':True,
 'source_environment':'/tmp/hdepic-qwen38-27b-env','source_packages_preserved':True,'independent_runtime_files':copy_audit,
 'packages':target_versions,'new_packages_only':{'peft':'0.18.1'},
 'qwen38_dense_native_import':True,'peft_get_peft_model_import':True,'fsdp2_fully_shard_import':True,
 'fsdp2_fully_shard_signature':str(inspect.signature(fully_shard)),
 'fsdp2_mixed_precision_policy_signature':str(inspect.signature(MixedPrecisionPolicy)),
 'native_class':type(model.base_model.model).__name__,'native_model_type':config.model_type,
 'model_identity':{'repo':'Qwen/Qwen3.8-27B','revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','config_sha256':sha(metadata/'config.json')},
 'meta_lora_injection':{'base_parameter_count':base_parameter_count,'base_bf16_bytes':base_parameter_count*2,'target_regex':target_regex,'target_linear_count':len(candidates),'trainable_tensor_count':len(trainable),'trainable_parameter_count':sum(x['numel'] for x in trainable),'r':16,'alpha':32,'dropout':0,'bias':'none','task_type':'CAUSAL_LM','vision_and_aligner_frozen':True,'embeddings_and_lm_head_frozen':True,'all_non_lora_frozen':True,'all_parameters_meta':True,'real_model_weight_bytes_allocated':0,'nonreentrant_gradient_checkpointing_enabled':True,'use_cache':False},
 'native_forward_signature':native_forward_signature,
 'training_warning':'Inference logits_to_keep=1 must not be reused for complete target loss; use full target logits or validated target-only logit selection.',
 'code_identity':{'setup_script_sha256':sha(Path('scripts/prepare_hdepic_qwen38_sft_environment.sh')),'peft_model_sha256':sha(dst/'peft/peft_model.py'),'peft_lora_model_sha256':sha(dst/'peft/tuners/lora/model.py'),'fsdp2_fully_shard_source_sha256':sha(Path(inspect.getsourcefile(fully_shard)))},
 'gpu_accessed':False,'gpu_model_load_performed':False,'model_inference_performed':False,'training_performed':False,
 'limitations':['Meta injection and import checks do not establish GPU training correctness.','Native Torch GatedDeltaNet fallback is used; actual backward/memory must be tested.','FSDP2 wrapping, checkpoint resumption, adapter saving and reload remain runtime checks.']}
immutable_json(run/'environment/compatibility.json',proof)
print(json.dumps(proof,indent=2,sort_keys=True))
PY
