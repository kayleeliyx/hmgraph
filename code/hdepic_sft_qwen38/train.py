"""Two-rank FSDP2 text-LoRA training; compact resumable state and task NLL."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
from .common import RUN,MODEL,MODEL_REVISION,GPU_UUIDS,configure,read_json,write_json,append_json,sha,rows


def identity(run):
 c=read_json(run/'frozen_config.json')
 paths=[Path(__file__).with_name(n) for n in ('common.py','data.py','model.py','checkpoints.py','train.py')]
 return {'config_sha256':sha(run/'frozen_config.json'),'model_revision':MODEL_REVISION,
         'data_sha256':{p:v['export_sha256'] for p,v in c['data'].items()},
         'code_sha256':{p.name:sha(p) for p in paths},'world_size':2}


def local_tensor(t):
 return t.to_local() if hasattr(t,'to_local') else t


def freeze_samples(model):
 """Deterministic sampled parameter-byte proof; optimizer membership is exhaustive."""
 import torch
 h={}
 for n,p in model.named_parameters():
  if p.requires_grad:continue
  t=local_tensor(p.detach()).reshape(-1)
  count=min(t.numel(),64)
  index=torch.arange(count,device=t.device,dtype=torch.int64)*max(t.numel()-1,0)//max(count-1,1)
  sample=t[index].contiguous().view(torch.uint8).cpu().numpy().tobytes()
  h[n]=hashlib.sha256(sample).hexdigest()
 return h


def grad_norm_and_clip(model,max_norm):
 import torch,torch.distributed as dist
 grads=[]
 for n,p in model.named_parameters():
  if not p.requires_grad:
   assert p.grad is None,('Frozen parameter received gradient',n)
   continue
  if p.grad is None:raise RuntimeError('Trainable parameter has no gradient: '+n)
  grads.append(local_tensor(p.grad))
 # One device-side norm over every gradient detects NaN/Inf without992 host
 # synchronizations. Each local shard contributes exactly once globally.
 norms=torch._foreach_norm(grads,2)
 total=torch.stack(norms).double().square().sum()
 dist.all_reduce(total,op=dist.ReduceOp.SUM)
 norm=total.sqrt();assert bool(torch.isfinite(norm))
 factor=min(1.0,max_norm/(norm.item()+1e-6))
 if factor<1:
  torch._foreach_mul_(grads,factor)
 return norm.item(),len(grads)


def memory():
 import torch
 return {'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def make_loader(dataset,indices,seed,workers=2):
 import torch
 from torch.utils.data import DataLoader,Subset
 from .data import collate_single
 kw={'dataset':Subset(dataset,indices),'batch_size':1,'shuffle':False,'num_workers':workers,
     'collate_fn':collate_single,'pin_memory':True,'generator':torch.Generator().manual_seed(seed)}
 if workers:kw.update(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=False)
 return DataLoader(**kw)


def validation_loss(model,processor,rank,run,stage):
 import torch,torch.distributed as dist
 from .data import TrainingDataset
 from .model import training_loss
 folder=run/'validation'/stage; folder.mkdir(parents=True,exist_ok=True)
 complete=folder/'completed.json'
 if complete.exists():
  cc=read_json(complete);assert cc['identity']==identity(run);return cc
 dataset=TrainingDataset('validation',processor,run=run);n=len(dataset);assert n==2209
 indices=list(range(rank,n,2)); valid=len(indices)
 if len(indices)<math.ceil(n/2):indices.append(0)
 # Every rank must call each FSDP collective in the same forward order/count.
 loader=make_loader(dataset,indices,20261003+1000+rank)
 model.eval();results=[];start=time.perf_counter()
 with torch.no_grad():
  for pos,batch in enumerate(loader):
   batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
   loss=training_loss(model,batch);assert bool(torch.isfinite(loss))
   if pos<valid:
    idx=indices[pos];s=dataset.scoring[idx]
    results.append({'id':s['id'],'task':s['task'],'row_index':idx,'nll':loss.item(),'supervised_tokens':int((batch['labels']!=-100).sum()),'stage':stage})
   if (pos+1)%100==0:print(json.dumps({'stage':stage,'rank':rank,'loss_records':pos+1,'total_per_rank':len(indices)}),flush=True)
 gathered=[None,None];dist.all_gather_object(gathered,results)
 merged=sorted(gathered[0]+gathered[1],key=lambda r:r['row_index'])
 assert len(merged)==len({r['id'] for r in merged})==2209
 assert [r['id'] for r in merged]==[s['id'] for s in dataset.scoring]
 result={'identity':identity(run),'stage':stage,'records':2209,'explicit_distributed_padding_not_scored':1,
         'aggregation':'arithmetic mean of per-example mean assistant+im_end token NLL, grouped by task',
         'tasks':{task:{'count':sum(r['task']==task for r in merged),'mean_per_example_nll':sum(r['nll'] for r in merged if r['task']==task)/sum(r['task']==task for r in merged)} for task in ('explain','track')},
         'seconds':time.perf_counter()-start}
 if rank==0:
  output=folder/'loss.jsonl';output.write_text(''.join(json.dumps(r)+'\n' for r in merged))
  result['loss_sha256']=sha(output);write_json(complete,result)
 dist.barrier();model.train()
 return result


def checkpoint_destination(folder, step, smoke=False):
 """Return real storage path; symlinks are published only after atomic save."""
 import shutil
 if smoke or step not in (909,1818):
  path=Path('/tmp/hdepic-qwen38-sft-checkpoints')/('smoke_'+folder.name if smoke else 'training')/f'checkpoint-{step}'
 else:path=folder/'checkpoints'/f'checkpoint-{step}'
 path.parent.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(path.parent).free>3*2**30, 'Checkpoint filesystem has less than3GiB free'
 return path


def publish_checkpoint(folder,path):
 alias=folder/'checkpoints'/path.name
 alias.parent.mkdir(parents=True,exist_ok=True)
 if alias==path:return
 if not alias.exists() and not alias.is_symlink():
  try:alias.symlink_to(path,target_is_directory=True)
  except FileExistsError:pass
 assert alias.is_symlink() and alias.resolve()==path.resolve()


def find_resume(folder,ident,smoke):
 from .checkpoints import choose_resume,validate_checkpoint
 external=Path('/tmp/hdepic-qwen38-sft-checkpoints')/('smoke_'+folder.name if smoke else 'training')
 candidates=[x for x in (choose_resume(folder/'checkpoints',ident),choose_resume(external,ident)) if x]
 return max(candidates,key=lambda p:validate_checkpoint(p,ident)['state']['step']) if candidates else None


def run_training(args):
 configure();run=Path(args.run);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
 assert rank==local and rank in (0,1) and os.environ['WORLD_SIZE']=='2'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(GPU_UUIDS)
 import numpy as np,torch,torch.distributed as dist
 from torch.utils.data.distributed import DistributedSampler
 from transformers import get_cosine_schedule_with_warmup
 from .data import TrainingDataset
 from .model import load_model,training_loss
 from .checkpoints import save_checkpoint,load_checkpoint,choose_resume
 from code.hdepic_qwen38_baseline.processor import load_processor
 torch.set_num_threads(2);torch.cuda.set_device(local)
 dist.init_process_group('nccl',device_id=torch.device('cuda',local))
 random.seed(20261003);np.random.seed(20261003);torch.manual_seed(20261003);torch.cuda.manual_seed_all(20261003)
 c=read_json(run/'frozen_config.json');tag='smoke' if args.smoke else 'training'
 folder=run/tag/(args.attempt if args.smoke else 'main');folder.mkdir(parents=True,exist_ok=True)
 ident=identity(run)
 if (folder/'completed.json').exists():
  assert read_json(folder/'completed.json')['identity']==ident
  if rank==0:print('Already complete: '+str(folder),flush=True)
  dist.destroy_process_group();return
 if not args.smoke:
  pf=read_json(run/'preflight/summary.json');assert pf['passed'] and pf['processed']==pf['retained']==16752
  sm=read_json(run/'smoke/passed.json');assert sm['identity']==ident and sm['passed']
 for part,d in c['data'].items():assert sha(d['export'])==d['export_sha256']
 processor=load_processor(MODEL)
 dataset=TrainingDataset('initial_sft',processor,run=run);assert len(dataset)==14543
 smoke_indices=[r['row_index'] for r in read_json(run/'inputs/smoke_selection.json')['records']]
 resume=find_resume(folder,ident,args.smoke)
 if not resume and (folder/f'optimizer_steps_rank{rank}.jsonl').exists():
  logs=rows(folder/f'optimizer_steps_rank{rank}.jsonl')
  if logs:raise RuntimeError('Logged optimizer updates without a valid checkpoint; refusing silent restart')
 started=time.perf_counter();model,modelaudit=load_model(rank,run=run,shard=True)
 write_json(folder/f'model_audit_rank{rank}.json',modelaudit)
 parameters=[p for p in model.parameters() if p.requires_grad]
 assert sum(p.numel() for p in parameters)==116727808 and len(parameters)==992
 assert all('lora_' in n and 'visual' not in n for n,p in model.named_parameters() if p.requires_grad)
 optimizer=torch.optim.AdamW(parameters,lr=1e-4,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,foreach=False)
 total_steps=2 if args.smoke else 1818;warmup=0 if args.smoke else math.ceil(total_steps*0.03)
 scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,total_steps)
 state={'step':0,'epoch':0,'microbatch_consumed':0,'sampler_state':{'seed':20261003}}
 # Exhaustive membership plus sampled frozen bytes and full local LoRA deltas.
 # Capture deterministic ORIGINAL adapter initialization before restoring a
 # trained checkpoint, including a completed-final-step resume.
 frozen_before=freeze_samples(model)
 lora_before={n:local_tensor(p.detach()).cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
 if resume:
  state=load_checkpoint(model,optimizer,scheduler,resume,ident)
  logpath=folder/f'optimizer_steps_rank{rank}.jsonl'
  if logpath.exists():
   oldlogs=rows(logpath);tail=[x for x in oldlogs if x['step']>state['step']]
   if tail:
    archive=folder/'resume_history'/f'rank{rank}_discarded_unsaved_tail_{time.time_ns()}.jsonl'
    archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(logpath.read_bytes())
    logpath.write_text(''.join(json.dumps(x)+'\n' for x in oldlogs if x['step']<=state['step']))
    write_json(archive.with_suffix('.json'),{'restored_checkpoint':str(resume),'saved_step':state['step'],'replayed_unsaved_steps':len(tail),'original_log_preserved':True})
 torch.cuda.reset_peak_memory_stats();model.train()
 write_json(folder/f'resolved_rank{rank}.json',{'identity':ident,'world_size':2,'physical_uuid':GPU_UUIDS[rank],
   'trainable_numel':116727808,'trainable_tensors':992,'optimizer_frozen_params':0,'gradient_accumulation_steps':8,
   'effective_batch':16,'total_optimizer_steps':total_steps,'warmup_steps':warmup,'resume':str(resume) if resume else None,
  'base_start':str(MODEL),'smoke_adapter_loaded_in_full_run':False,'smoke':args.smoke,'optimizer_class':type(optimizer).__name__})
 if not args.smoke and not resume:
  validation_loss(model,processor,rank,run,'base')
 micro_count=state['microbatch_consumed'];step=state['step']
 for epoch in range(state['epoch'],1 if args.smoke else 2):
  if args.smoke:
   indices=[smoke_indices[(i*2+rank)%4] for i in range(16)]
  else:
   sampler=DistributedSampler(dataset,num_replicas=2,rank=rank,shuffle=True,seed=20261003,drop_last=False);sampler.set_epoch(epoch)
   indices=list(sampler);assert len(indices)==7272
   g=torch.Generator().manual_seed(20261003+epoch);padding_index=torch.randperm(len(dataset),generator=g).tolist()[0]
   write_json(folder/f'sampler_epoch{epoch+1}_rank{rank}.json',{'epoch':epoch+1,'source_records':14543,'rank_records':len(indices),'rank_indices':indices,'explicit_global_padding_id':dataset.scoring[padding_index]['id'],'optimizer_steps':909})
  consumed=micro_count if epoch==state['epoch'] else 0
  assert consumed%8==0
  loader=make_loader(dataset,indices[consumed:],20261003+epoch*2+rank)
  optimizer.zero_grad(set_to_none=True); losses=[];t_step=time.perf_counter()
  for local_i,batch in enumerate(loader):
   i=consumed+local_i;boundary=(i+1)%8==0
   model.set_requires_gradient_sync(boundary);model.set_is_last_backward(boundary)
   batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
   assert tuple(batch['image_grid_thw'].shape)==(16,3) and int((batch['input_ids']==248056).sum())==2304
   assert batch['input_ids'].shape==batch['labels'].shape and batch['input_ids'].shape[1]<=8192
   loss=training_loss(model,batch);assert bool(torch.isfinite(loss)),('Nonfinite loss',step,i)
   losses.append(loss.detach());(loss/8).backward()
   if not boundary:continue
   norm,tensors=grad_norm_and_clip(model,1.0);used_lr=optimizer.param_groups[0]['lr']
   optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
   torch.cuda.synchronize();elapsed=time.perf_counter()-t_step
   mean=torch.stack(losses).mean();dist.all_reduce(mean);mean/=2
   log={'step':step,'epoch':epoch+(i+1)/len(indices),'microbatch_consumed':i+1,'loss':mean.item(),'lr_used':used_lr,'lr_next':optimizer.param_groups[0]['lr'],
        'grad_norm_before_clip':norm,'finite_grad_tensors':tensors,'seconds':elapsed,**memory()}
   append_json(folder/f'optimizer_steps_rank{rank}.jsonl',log)
   if rank==0:print(json.dumps({'training':tag,**log}),flush=True)
   losses=[]
   if step%250==0 or i+1==len(indices) or args.smoke:
    state={'step':step,'epoch':epoch,'microbatch_consumed':i+1,'sampler_state':{'seed':20261003,'epoch':epoch}}
    checkpoint_path=checkpoint_destination(folder,step,args.smoke)
    save_checkpoint(model,optimizer,scheduler,checkpoint_path,state,ident)
    if rank==0:publish_checkpoint(folder,checkpoint_path)
    dist.barrier()
   t_step=time.perf_counter()
  micro_count=0
  if not args.smoke:
   assert step==(epoch+1)*909
   validation_loss(model,processor,rank,run,f'epoch{epoch+1}')
 assert step==total_steps
 frozen_after=freeze_samples(model);assert frozen_before==frozen_after,'Frozen parameter sample changed'
 delta=[]
 for n,p in model.named_parameters():
  if not p.requires_grad:continue
  now=local_tensor(p.detach()).cpu();diff=(now-lora_before[n]).abs()
  assert bool(torch.isfinite(now).all())
  delta.append({'name':n,'changed_elements':int((diff>0).sum()),'max_abs_change':float(diff.max())})
 assert all(r['changed_elements']>0 for r in delta), 'A trainable LoRA tensor failed to update'
 write_json(folder/f'parameter_updates_rank{rank}.json',{'all_992_local_lora_tensors_changed':True,'tensors':delta,'frozen_parameter_sample_count':len(frozen_before),'frozen_sample_hashes_unchanged':True,'frozen_optimizer_membership':0})
 result={'identity':ident,'passed':True,'optimizer_steps':step,'rank':rank,'training_and_validation_seconds':time.perf_counter()-started,**memory(),
         'checkpoint':str(folder/'checkpoints'/f'checkpoint-{step}'),'smoke':args.smoke,'frozen_sample_hashes_unchanged':True}
 write_json(folder/f'completed_rank{rank}.json',result);dist.barrier()
 if rank==0:
  result['ranks']=[read_json(folder/f'completed_rank{r}.json') for r in (0,1)]
  write_json(folder/'completed.json',result)
 dist.barrier();dist.destroy_process_group()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(RUN));parser.add_argument('--smoke',action='store_true');parser.add_argument('--attempt',default='attempt-001')
 args=parser.parse_args();run_training(args)
if __name__=='__main__':main()
