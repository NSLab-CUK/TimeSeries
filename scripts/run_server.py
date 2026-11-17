"""서버 GPU 실행 진입점: 선행 점검, 12회 smoke, 고정 120회 학습, 검증된 재개."""
from pathlib import Path
import argparse,sys,json,time,os,traceback,hashlib,platform,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,torch
import pipeline

class Tee:
 def __init__(self,*streams): self.streams=streams
 def write(self,s):
  for stream in self.streams: stream.write(s); stream.flush()
 def flush(self):
  for stream in self.streams: stream.flush()

def summarize(job,seeds):
 rows=[]; metrics=[]; target_hashes=set()
 for model in ['rnn','lstm','gru']:
  for lead in [1,3,5,10]:
   for seed in seeds:
    run=job/f'full/{model}/lead_{lead}/seed_{seed}'; complete=run/'COMPLETED.json'
    if not complete.exists():
     rows.append({'model':model,'lead_seconds':lead,'seed':seed,'completed':False,'path':str(run)}); continue
    verify=json.loads((run/'verification.json').read_text(encoding='utf-8'))
    rows.append({'model':model,'lead_seconds':lead,'seed':seed,'completed':True,'path':str(run),**verify})
    target_hashes.add(pipeline.sha(run/'test_targets.csv')); metrics.append(pd.read_csv(run/'metrics_per_record_gas.csv'))
 pd.DataFrame(rows).to_csv(job/'run_status.csv',index=False,encoding='utf-8-sig')
 if metrics:
  combined=pd.concat(metrics); combined.to_csv(job/'metrics_all_runs.csv',index=False,encoding='utf-8-sig')
  aggregate=combined.groupby(['model','lead_seconds','gas']).agg(n_record_run_scores=('RMSE','size'),RMSE_mean=('RMSE','mean'),RMSE_SD=('RMSE','std'),
   MAE_mean=('MAE','mean'),MAE_SD=('MAE','std'),R2_mean=('R2','mean'),R2_SD=('R2','std')).reset_index()
  aggregate['aggregation']='equal_record/run_score_mean; SD over record/run scores; not an independent-experiment CI'
  aggregate.to_csv(job/'metrics_summary_descriptive.csv',index=False,encoding='utf-8-sig')
 assert len(target_hashes)<=1,'Models/leads/seeds do not share identical full test targets'
 pipeline.dump({'expected_fits':len(rows),'completed_fits':sum(r['completed'] for r in rows),'all_complete':all(r['completed'] for r in rows),
  'common_full_test_target_hash_count':len(target_hashes),'scope':'NEW forecasting; not historical reproduction; previously inspected test records',
  'next_step':'Use this completed package to revise manuscript/SI/figures/responses together. Smoke results are not paper metrics.'},job/'RESULT_STATUS.json')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--job',default='new_forecasting_v1'); parser.add_argument('--resume',action='store_true'); parser.add_argument('--smoke-only',action='store_true')
 parser.add_argument('--pilot',action='store_true',help='고정 seed 1개, 모델×lead 12회만 전체 데이터로 학습')
 parser.add_argument('--allow-cpu-smoke',action='store_true',help='CPU에서 smoke-only 검증할 때만 허용')
 args=parser.parse_args()
 if not args.job or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.job): raise ValueError('Use a simple job name')
 if not torch.cuda.is_available() and not (args.smoke_only and args.allow_cpu_smoke): raise RuntimeError('GPU PyTorch/CUDA unavailable. Activate the server CUDA PyTorch environment; full CPU fallback is disabled.')
 job=Path(__file__).resolve().parents[1]/'results'/args.job; job.mkdir(parents=True,exist_ok=True)
 log=(job/'console.log').open('a',encoding='utf-8',buffering=1); sys.stdout=Tee(sys.stdout,log); sys.stderr=Tee(sys.stderr,log)
 seeds=[20261113] if args.pilot else list(range(20261113,20261123)); start=time.perf_counter()
 print('NEW FORECASTING A — NOT historical reproduction',flush=True)
 hardware={'UTC_start':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
  'pandas':pd.__version__,'platform':platform.platform(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
  'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'GPU_memory_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}
 pipeline.dump(hardware,job/'server_environment.json'); print(json.dumps(hardware,ensure_ascii=False),flush=True)
 print('Planned full fits:',12*len(seeds),'(3 models x 4 leads x',len(seeds),'seeds); max200 epochs, patience20; no test tuning.',flush=True)
 prepared=job/'data_contract'; data,index,scaler,contract=pipeline.prepare(prepared)
 print('Locked files:',contract['counts_by_role'],'common targets:',contract['common_targets_by_role'],flush=True)
 print('CO2 gap is split; all history ends h actual seconds before its target. Unit/AC/analyzer-latency definitions remain laboratory items.',flush=True)
 print('Running/replaying 12 small two-epoch flow checks before full fits.',flush=True)
 pipeline.execute(data,index,scaler,contract,'smoke',seeds=[seeds[0]],resume=args.resume,output_root=job,prepared_root=prepared)
 measured=job/'hardware_and_execution_scope_additional.json'
 if measured.exists():
  estimate=json.loads(measured.read_text(encoding='utf-8'))['extrapolated_training_seconds_12_fits_max200']*len(seeds)/3600
  print(f'This server smoke extrapolation: {estimate:.2f} training-step hours for the declared full scope; validation, I/O and stopping effects excluded.',flush=True)
 if args.smoke_only:
  print('SMOKE ONLY completed. Full training was not launched. Output:',job,flush=True); return
 print('Starting/resuming full fits. Test is evaluated after validation-based selection; stored test records were historically inspected.',flush=True)
 try:
  pipeline.execute(data,index,scaler,contract,'full',seeds=seeds,resume=args.resume,output_root=job,prepared_root=prepared)
 finally: summarize(job,seeds)
 print('Finished. Output:',job,'wall hours:',(time.perf_counter()-start)/3600,flush=True)

if __name__=='__main__': main()
