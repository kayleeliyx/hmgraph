"""Fresh two-rank CUDA reload of the disposable smoke state, without more updates."""
from pathlib import Path
import argparse,os,json
from .common import RUN,MODEL,GPU_UUIDS,configure,read_json,write_json,sha
from .train import identity,local_tensor,memory


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(RUN));parser.add_argument('--attempt',default='attempt-002');args=parser.parse_args()
 run=Path(args.run);folder=run/'smoke'/args.attempt;ident=identity(run)
 completed=read_json(folder/'completed.json');assert completed['identity']==ident and completed['optimizer_steps']==2
 configure();assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(GPU_UUIDS)
 import torch,torch.distributed as dist
 from transformers import get_cosine_schedule_with_warmup
 from .model import load_model,training_loss
 from .checkpoints import load_checkpoint,validate_checkpoint
 from .data import load_partition,build_training_inputs
 from code.hdepic_qwen38_baseline.processor import load_processor
 rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
 dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
 model,audit=load_model(rank,run,shard=True)
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
 scheduler=get_cosine_schedule_with_warmup(optimizer,0,2)
 checkpoint=Path(completed['checkpoint']).resolve();state=load_checkpoint(model,optimizer,scheduler,checkpoint,ident)
 assert state['step']==2 and state['microbatch_consumed']==16
 payload=torch.load(checkpoint/f'rank-{rank:05}.pt',map_location='cpu',weights_only=False)
 exact=[]
 for n,p in model.named_parameters():
  if p.requires_grad:
   assert torch.equal(local_tensor(p.detach()).cpu(),payload['trainable'][n]['value']),n
   exact.append(n)
 assert len(exact)==992 and len(optimizer.state)==992
 # The separate diagnostic established identical hidden states, exact causal
 # labels and native loss reduction. BF16 lm_head GEMMs with different row
 # counts round differently; requiring FP32-level equality between those two
 # projections is invalid. Preserve the original failed assertion and verify
 # the concrete evidence instead of loosening its tolerance.
 diagnostic_path=folder/'parity_diagnostic'/f'record0_rank{rank}.json'
 diagnostic=read_json(diagnostic_path)
 assert diagnostic['identity']==ident
 assert diagnostic['checkpoint_completed_sha256']==sha(checkpoint/'completed.json')
 assert diagnostic['diagnostic_code_sha256']==sha(Path(__file__).with_name('parity_diagnostic.py'))
 assert diagnostic['classification']=='bf16_head_projection_shape_dependence_supported'
 assert diagnostic['label_alignment']['passed'] and diagnostic['same_weight_samples']
 assert diagnostic['selected_vs_full_hidden']['exact_equal']
 assert diagnostic['selected_vs_repeat_logits']['exact_equal']
 assert diagnostic['selected_vs_short_bf16_on_full_hidden']['exact_equal']
 assert diagnostic['native_full_vs_same_logits_masked_ce_agrees_2e_minus5']
 assert diagnostic['all_992_local_adapter_hashes_unchanged'] and diagnostic['no_optimizer_updates']
 processor=load_processor(MODEL);exports,scoring,_=load_partition('initial_sft',run)
 records=read_json(run/'inputs/smoke_selection.json')['records'];results=[];model.eval()
 with torch.no_grad():
  for i,s in enumerate(records):
   batch,check=build_training_inputs(processor,exports[s['row_index']],s['id'],s['row_index'])
   batch={k:v.cuda() for k,v in batch.items()}
   loss=training_loss(model,batch);assert bool(torch.isfinite(loss))
   if i==0:
    assert s['id']==diagnostic['id']
    assert abs(loss.item()-diagnostic['selected_training_loss'])<=2e-5
   results.append({'id':s['id'],'category':s['category'],'loss':loss.item(),'tokens':check.get('sequence_tokens',batch['input_ids'].numel()),'16_images':True,'complete_target':True})
 result={'identity':ident,'passed':True,'rank':rank,'checkpoint':str(checkpoint),'exact_local_adapter_tensors':len(exact),'restored_optimizer_states':len(optimizer.state),'restored_progress':state,'scheduler_last_epoch':scheduler.last_epoch,'records':results,'no_new_optimizer_steps':True,
  'loss_objective_verification':{'causal_indices_and_native_loss_reduction_verified':True,'bf16_full_vs_selected_logits_bitwise_equal':False,'diagnostic_path':str(diagnostic_path),'diagnostic_sha256':sha(diagnostic_path),'cause':diagnostic['classification'],'original_failed_parity_assertion_preserved':True},**memory()}
 write_json(folder/f'fresh_fsdp_reload_rank{rank}.json',result)
 dist.barrier()
 if rank==0:
  result['ranks']=[read_json(folder/f'fresh_fsdp_reload_rank{r}.json') for r in (0,1)]
  write_json(folder/'fresh_fsdp_reload.json',result)
 dist.barrier();dist.destroy_process_group();print(json.dumps({'rank':rank,'fresh_fsdp_reload_passed':True}),flush=True)
if __name__=='__main__':main()
