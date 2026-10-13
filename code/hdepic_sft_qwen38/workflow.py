"""Exclusive local stage launcher; logs all GPU commands and never uses test data."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .common import RUN,ENV,PROJECT,GPU_UUIDS,gpu_probe,write_json,read_json,append_json,sha,freeze_config,rows


def validate_loss_caches():
 from .train import identity
 from .data import load_partition
 markers=list((RUN/'validation').glob('*/completed.json'))
 if not markers:return
 _,scoring,_=load_partition('validation',RUN)
 expected=[r['id'] for r in scoring]
 for marker in markers:
  saved=read_json(marker);path=marker.parent/'loss.jsonl'
  assert saved['identity']==identity(RUN) and path.is_file()
  assert sha(path)==saved['loss_sha256'], 'Saved validation loss changed: '+str(path)
  records=rows(path)
  assert [r['id'] for r in records]==expected and len(set(expected))==2209
  assert all(r['task']==s['task'] for r,s in zip(records,scoring))
  assert {task:sum(r['task']==task for r in records) for task in ('explain','track')}=={'explain':228,'track':1981}


def launch(stage,extra=()):
 if stage=='train':
  validate_loss_caches()
  smoke_gate(read_json(RUN/'smoke/passed.json')['attempt'])
 from .storage import validate_storage
 storage=validate_storage(require_space=True,run=RUN)
 python=str((ENV/'bin/python').resolve())
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=','.join(GPU_UUIDS),PYTHONPATH=str(PROJECT)+":"+str(PROJECT/"code"),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NCCL_DEBUG='WARN')
 probe=gpu_probe(require_idle=True);write_json(RUN/'resources'/f'{time.time_ns()}_{stage}.json',probe)
 if stage in ('smoke','train'):
  command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','code.hdepic_sft_qwen38.train']
  if stage=='smoke':command+=['--smoke',*extra]
 elif stage=='verify-smoke':
  command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','code.hdepic_sft_qwen38.verify',*extra]
 elif stage=='diagnose-loss':
  command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','code.hdepic_sft_qwen38.parity_diagnostic',*extra]
 elif stage=='diagnose-finalization':
  command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','code.hdepic_sft_qwen38.finalize_diagnostic',*extra]
 elif stage in ('smoke-reload','epoch1','epoch2'):
  command=[python,'-m','code.hdepic_sft_qwen38.evaluate',stage,*extra]
 else:raise ValueError(stage)
 stamp=time.time_ns();log=RUN/'logs'/f'{stamp}_{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
 record={'stage':stage,'command':command,'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],'log':str(log),'started_unix':time.time(),'storage':storage}
 write_json(log.with_suffix('.command.json'),record);print(json.dumps(record),flush=True)
 with log.open('w') as stream:
  process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
  write_json(RUN/'active_process.json',{'stage':stage,'pid':process.pid,'parent_pid':os.getpid(),'log':str(log),'started_unix':record['started_unix']})
  status=process.wait()
 record.update(returncode=status,elapsed_seconds=time.time()-record['started_unix']);write_json(log.with_suffix('.result.json'),record)
 write_json(RUN/'active_process.json',{'stage':stage,'pid':process.pid,'parent_pid':os.getpid(),
  'log':str(log),'started_unix':record['started_unix'],'finished_unix':time.time(),'returncode':status,
  'status':'completed' if status==0 else 'failed'})
 if status:raise RuntimeError(f'{stage} failed ({status}); preserved log {log}')
 print(json.dumps(record),flush=True)


def smoke_gate(attempt):
 from .train import identity
 assert all(completed_stage(s,attempt) for s in ('smoke','diagnose-loss','verify-smoke','smoke-reload'))
 ident=identity(RUN);root=RUN/'smoke'/attempt
 complete=read_json(root/'completed.json');reload=read_json(root/'fresh_fsdp_reload.json')
 inference=read_json(root/'reload/completed.json')
 assert complete['identity']==reload['identity']==ident
 assert complete['passed'] and complete['optimizer_steps']==2 and reload['passed']
 assert inference['passed'] and inference['count']==4
 for rank in (0,1):
  updates=read_json(root/f'parameter_updates_rank{rank}.json')
  proof=read_json(root/f'fresh_fsdp_reload_rank{rank}.json')
  assert updates['all_992_local_lora_tensors_changed'] and updates['frozen_sample_hashes_unchanged']
  assert updates['frozen_optimizer_membership']==0
  assert proof['exact_local_adapter_tensors']==proof['restored_optimizer_states']==992
  assert proof['no_new_optimizer_steps'] and len(proof['records'])==4
 pf=read_json(RUN/'preflight/summary.json')
 assert pf['passed'] and pf['processed']==pf['retained']==16752
 proof_paths=[root/'completed.json',root/'fresh_fsdp_reload.json',root/'reload/completed.json',RUN/'preflight/summary.json']
 result={'identity':ident,'passed':True,'attempt':attempt,'optimizer_steps':2,
  'source_sha256':{str(p.relative_to(RUN)):sha(p) for p in proof_paths},
  'fresh_full_run_starts_original_base':True,'smoke_adapter_used_for_full_training':False,
  'smoke_tag_compliance_diagnostic':inference['strict_format_valid'],
  'semantic_accuracy_established':False}
 write_json(RUN/'smoke/passed.json',result,immutable=True)
 print(json.dumps(result),flush=True)


def completed_stage(stage,attempt):
 """Validate completion identities before skipping an expensive GPU stage."""
 if stage in ('smoke','train'):
  from .train import identity
  folder=RUN/'smoke'/attempt if stage=='smoke' else RUN/'training/main'
  marker=folder/'completed.json'
  if not marker.exists():return False
  result=read_json(marker)
  assert result['passed'] and result['identity']==identity(RUN)
  assert result['optimizer_steps']==(2 if stage=='smoke' else 1818)
  from .checkpoints import validate_checkpoint
  checkpoint=validate_checkpoint(Path(result['checkpoint']).resolve(),result['identity'])
  assert checkpoint['state']['step']==result['optimizer_steps']
  assert checkpoint['state']['epoch']==(0 if stage=='smoke' else 1)
  assert checkpoint['state']['microbatch_consumed']==(16 if stage=='smoke' else 7272)
  if stage=='train':validate_loss_caches()
  return True
 if stage=='verify-smoke':
  from .train import identity
  marker=RUN/'smoke'/attempt/'fresh_fsdp_reload.json'
  if not marker.exists():return False
  result=read_json(marker);assert result['passed'] and result['identity']==identity(RUN)
  return True
 if stage=='diagnose-loss':
  from .train import identity
  marker=RUN/'smoke'/attempt/'parity_diagnostic/record0.json'
  if not marker.exists():return False
  result=read_json(marker);assert result['identity']==identity(RUN)
  assert result['diagnostic_code_sha256']==sha(Path(__file__).with_name('parity_diagnostic.py'))
  assert len(result['ranks'])==2
  assert all(r['classification']=='bf16_head_projection_shape_dependence_supported' for r in result['ranks'])
  return True
 if stage in ('smoke-reload','epoch1','epoch2'):
  from .evaluate import stage_paths,run_evaluation
  _,output=stage_paths(stage,RUN,attempt)
  if not (output/'completed.json').exists():return False
  # Its own immutable configuration, prediction and checkpoint checks execute;
  # a completed stage returns before model loading or GPU probing.
  assert run_evaluation(stage,RUN,attempt)['passed']
  return True
 raise ValueError(stage)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','preflight','smoke','verify-smoke','smoke-reload','diagnose-loss','diagnose-finalization','gate','train','evaluate','report','status','all'));parser.add_argument('--attempt',default='attempt-002')
 args=parser.parse_args()
 from .storage import validate_storage
 validate_storage(run=RUN)
 RUN.mkdir(parents=True,exist_ok=True)
 if args.stage=='status':
  print(json.dumps({'active':read_json(RUN/'active_process.json') if (RUN/'active_process.json').exists() else None,
    'preflight_complete':(RUN/'preflight/summary.json').exists(),'smoke_passed':(RUN/'smoke/passed.json').exists(),
    'training_complete':(RUN/'training/main/completed.json').exists()},indent=2));return
 with (RUN/'workflow.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if args.stage=='diagnose-finalization':launch('diagnose-finalization')
  if args.stage in ('prepare','all'):
   from .data import prepare
   freeze_config();prepare()
  if args.stage in ('preflight','all'):
   from .data import preflight
   preflight(workers=4)
  for stage in ('smoke','diagnose-loss','verify-smoke','smoke-reload'):
   if args.stage in (stage,'all'):
    if completed_stage(stage,args.attempt):print(stage+' already complete and verified.',flush=True)
    else:launch(stage,['--attempt',args.attempt])
  if args.stage in ('gate','all'):smoke_gate(args.attempt)
  if args.stage in ('train','all'):
   if completed_stage('train',args.attempt):print('Training already complete and verified; no new updates.',flush=True)
   else:launch('train')
  if args.stage in ('evaluate','all'):
   assert completed_stage('train',args.attempt),'Finish training before sequential checkpoint generation'
   for stage in ('epoch1','epoch2'):
    if completed_stage(stage,args.attempt):print(stage+' evaluation already complete and verified.',flush=True)
    else:launch(stage)
  if args.stage in ('report','all'):
   from .report import generate_report
   result=generate_report(RUN)
   print(json.dumps({'status':result['status'],'execution_complete':result['execution_complete'],
    'selection':result['selection'],'report':str(RUN/'execution_report.md')},indent=2),flush=True)
if __name__=='__main__':main()
