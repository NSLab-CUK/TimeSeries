"""One isolated GPU process, one lead shard; no shared summary-file writes."""
from pathlib import Path
import argparse,json,os,sys,time,platform,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,torch
import pipeline
from run_server import Tee

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--job-dir',required=True,type=Path); parser.add_argument('--gpu-id',required=True); parser.add_argument('--lead',required=True,type=int,choices=pipeline.LEADS)
 parser.add_argument('--phase',required=True,choices=['preflight','smoke','full']); parser.add_argument('--seeds',nargs='+',type=int,required=True)
 parser.add_argument('--resume',action='store_true'); parser.add_argument('--allow-cpu-smoke',action='store_true'); parser.add_argument('--log-every-seconds',type=float,default=10)
 args=parser.parse_args(); worker=args.job_dir/'workers'/f'gpu_{args.gpu_id}'; worker.mkdir(parents=True,exist_ok=True)
 with (worker/'console.log').open('a',encoding='utf-8',buffering=1) as log:
  sys.stdout=Tee(sys.stdout,log); sys.stderr=Tee(sys.stderr,log)
  try:
   use_cuda=torch.cuda.is_available()
   if not use_cuda and not (args.allow_cpu_smoke and args.phase!='full'): raise RuntimeError(f'GPU {args.gpu_id}: CUDA PyTorch/device unavailable; no full CPU fallback')
   if use_cuda:
    if torch.cuda.device_count()!=1: raise RuntimeError('Worker must see exactly one GPU; launcher isolation failed')
    torch.cuda.set_device(0)
    probe=torch.arange(16,device='cuda').sum().item(); assert probe==120
   environment={'phase':args.phase,'UTC_start':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),'PID':os.getpid(),
    'assigned_GPU':args.gpu_id,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'lead_seconds':args.lead,'python':sys.version,
    'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),
    'GPU':torch.cuda.get_device_name(0) if use_cuda else None,'visible_device_count':torch.cuda.device_count(),
    'GPU_memory_bytes':torch.cuda.get_device_properties(0).total_memory if use_cuda else 0,'CPU_smoke_only':not use_cuda}
   pipeline.dump(environment,worker/f'{args.phase}_environment.json')
   print('DEVICE READY',json.dumps(environment,ensure_ascii=False),flush=True)
   if args.phase=='preflight': return
   plan=json.loads((args.job_dir/'job_plan.json').read_text(encoding='utf-8'))
   if pipeline.sha(Path(pipeline.__file__))!=plan['pipeline_source_sha256']: raise ValueError('Pipeline changed after job planning')
   if pipeline.sha(args.job_dir/'data_contract/contract_and_locked_config.json')!=plan['contract_sha256']: raise ValueError('Locked data contract changed')
   prepared=args.job_dir/'data_contract'; data,index,scaler,contract=pipeline.load_prepared_cache(prepared,plan['cache'])
   seeds=[args.seeds[0]] if args.phase=='smoke' else args.seeds
   print(f'ASSIGNMENT phase={args.phase} lead={args.lead}s fits={3*len(seeds)} seeds={seeds}',flush=True)
   pipeline.execute(data,index,scaler,contract,args.phase,leads=[args.lead],seeds=seeds,resume=args.resume,
    output_root=args.job_dir,prepared_root=prepared,summary_dir=worker/args.phase,
    log_every_seconds=args.log_every_seconds,progress_file=worker/'progress.json')
   print(f'WORKER DONE phase={args.phase} lead={args.lead}s',flush=True)
  except BaseException as exc:
   pipeline.dump({'state':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed','phase':args.phase,'lead_seconds':args.lead,
    'GPU':args.gpu_id,'exception':repr(exc),'unix_time':time.time()},worker/'failure.json')
   traceback.print_exc(); raise
  finally:
   sys.stdout=sys.__stdout__; sys.stderr=sys.__stderr__

if __name__=='__main__': main()
