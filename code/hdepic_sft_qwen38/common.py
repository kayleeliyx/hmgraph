from __future__ import annotations
import hashlib,json,os,subprocess,time
from pathlib import Path
PROJECT=Path('/data/yaxuanli/ActGround')
RUN=PROJECT/'runs/hdepic_sft_qwen38_27b_v1'
DATASET=PROJECT/'datasets/hdepic_sft_v2_repaired'
ENV=Path('/tmp/hdepic-qwen38-sft-env')
BASELINE=PROJECT/'runs/hdepic_qwen38_27b_baseline_v1'
MODEL_REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MODEL_REPO='Qwen/Qwen3.8-27B'
MODEL=Path('/tmp/hdepic-qwen38-27b-hf')/'models--Qwen--Qwen3.8-27B'/'snapshots'/MODEL_REVISION
GPU_UUIDS=['GPU-f63ffaff-cdd7-e954-cea4-b051c47fbb20','GPU-1ac69f34-45b7-8b31-05a6-98cce9530168']

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def canonical(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def write_json(path,value,immutable=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 if immutable and path.exists():assert path.read_text()==data,('Frozen artifact changed',path)
 else:
  tmp=path.with_name(path.name+'.tmp');tmp.write_text(data);tmp.replace(path)
def append_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def read_json(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def configure(cpu=False):
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_DISABLED='true')
 if cpu:os.environ['CUDA_VISIBLE_DEVICES']=''

def gpu_probe(require_idle=False):
 fields='index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu'
 command=['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits']
 p=subprocess.run(command,capture_output=True,text=True,check=True)
 all_rows=[v.strip().split(', ') for v in p.stdout.splitlines()]; selected=[v for v in all_rows if v[0] in ('0','1')]
 assert len(selected)==2 and [v[1] for v in selected]==GPU_UUIDS
 proc=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if require_idle:
  assert all(int(v[4])<1024 and int(v[6])<5 for v in selected),('Authorized GPU occupied',selected)
  assert not any(v in proc.stdout for v in GPU_UUIDS),'Authorized GPU has compute process'
 scheduler={k:v for k,v in os.environ.items() if k.startswith(('SLURM_','PBS_'))}
 visible=os.environ.get('CUDA_VISIBLE_DEVICES')
 if scheduler and visible is not None:
  allowed=set(visible.split(','));assert allowed=={'0','1'} or allowed==set(GPU_UUIDS),'Scheduler allocation does not match authorized GPUs0,1'
 return {'at_unix':time.time(),'physical_gpus':selected,'processes':proc.stdout,'scheduler':scheduler,'caller_cuda_visible_devices':visible,'idle_required':require_idle}

def base_config():
 old=read_json(PROJECT/'runs/hdepic_sft_8b_v1/frozen_config.json')
 return {'protocol':'hdepic_sft_qwen38_27b_v1','model':{'repo_id':MODEL_REPO,'revision':MODEL_REVISION,'path':str(MODEL),'original_base_only':True},
  'dataset':str(DATASET),'data':old['data'],'split_sha256':old['split_sha256'],
  'processor':{'individual_images':16,'width':384,'height':384,'image_tokens_per_frame':144,'min_pixels':65536,'max_pixels':147456,'max_length':8192,'thinking':False,'truncation':False,'assistant_target_and_im_end_only_loss':True},
  'training':{'method':'text_lora','rank':16,'alpha':32,'dropout':0.0,'target_modules':'all torch.nn.Linear under model.language_model.layers, excluding visual and lm_head','precision':'bfloat16','lora_master_dtype':'float32','vision_frozen':True,'aligner_frozen':True,'attention_implementation':'sdpa','gradient_checkpointing':{'enabled':True,'use_reentrant':False},'packing':False,'world_size':2,'per_device_batch_size':1,'gradient_accumulation_steps':8,'effective_batch_size':16,'epochs':2,'learning_rate':1e-4,'scheduler':'cosine','warmup_ratio':0.03,'optimizer':'torch.optim.AdamW','weight_decay':0.1,'betas':[0.9,0.95],'epsilon':1e-8,'max_grad_norm':1.0,'seed':20261003,'sampler':'DistributedSampler(shuffle=True,drop_last=False)','parallelism':'native torch2.6 FSDP2 FULL_SHARD','save_every_optimizer_steps':250,'save_epoch_boundaries':True,'save_optimizer_scheduler_rng':True,'max_steps_override':None},
  'gpus':{'authorized_physical_indices':[0,1],'uuids':GPU_UUIDS},
  'validation':{'partition':'validation','records':2209,'explain':228,'track':1981,'loss_aggregation':'arithmetic mean of per-example mean assistant-plus-termination token NLL by task','checkpoint_selection':'lowest Explain validation mean per-example NLL, tie epoch1','generation':'same228 Explain prompt IDs and parser as saved original27B baseline, greedy max256; no test access'},
  'deviations_from_8b':{'framework':'ms-swift3.12.6 cannot load native qwen3_5; use isolated native Transformers5.3 + PEFT + Torch2.6 FSDP2','parallelism':'27,356,728,560 native parameters require50.96GiB BF16 before activations; full-replica DDP exceeds48GB, so shard weights/gradients/optimizer across authorized pair','template':'native Qwen3.8 nonthinking assistant prefix is masked; original target and im_end are supervised','lora_targets':'native hybrid text projections recomputed from nn.Linear, all-linear policy preserved'},
  'limitations':{'reason_labels':'provisional; independent human review pending','second_stage_excluded':True,'test_access':False,'grpo':False,'external_inference_api':False,'old_artifacts_preserved':True}}

def freeze_config():
 config=base_config()
 for part,v in config['data'].items():
  for kind in ('export','scoring','prompts','text'):assert sha(v[kind])==v[kind+'_sha256'],(part,kind)
 assert sha(DATASET/'split_manifest.json')==config['split_sha256']
 write_json(RUN/'frozen_config.json',config,immutable=True)
 return config
