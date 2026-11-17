"""4개 GPU에 독립 실행을 배분하고 터미널/파일 로그를 실시간 통합한다."""
from pathlib import Path
import argparse,contextlib,datetime,itertools,json,os,platform,queue,signal,subprocess,sys,threading,time,shutil,hashlib,math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'src'
sys.path.insert(0,str(SOURCE))
KST=datetime.timezone(datetime.timedelta(hours=9))

def stamp(): return datetime.datetime.now(KST).strftime('%m-%d %H:%M:%S KST')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(value,path): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def schedule(gpus,seeds):
 if len(gpus)!=4 or len(set(gpus))!=4 or any(not str(g).isdigit() for g in gpus): raise ValueError('Specify four different GPU numbers, e.g. 0,1,2,3')
 return [{'gpu_id':gpu,'lead_seconds':lead,'tasks':[{'model':model,'lead_seconds':lead,'seed':seed} for model,seed in itertools.product(['rnn','lstm','gru'],seeds)]}
         for gpu,lead in zip(gpus,[1,3,5,10])]

@contextlib.contextmanager
def exclusive_job(job):
 """OS lock is released even after a crash; refuse a second launcher."""
 path=job/'.launcher.lock'; handle=path.open('a+b'); handle.seek(0,2)
 if handle.tell()==0: handle.write(b'0'); handle.flush()
 handle.seek(0); locked=False
 try:
  try:
   if os.name=='nt':
    import msvcrt; msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
   else:
    import fcntl; fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   locked=True
  except OSError as exc: raise RuntimeError(f'Another launcher owns this job: {job}') from exc
  yield
 finally:
  if locked:
   handle.seek(0)
   if os.name=='nt':
    import msvcrt; msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
   else:
    import fcntl; fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
  handle.close()

def command_for(job,assignment,phase,seeds,args):
 command=[sys.executable,'-u',str(HERE/'gpu_worker.py'),'--job-dir',str(job),'--gpu-id',assignment['gpu_id'],
  '--lead',str(assignment['lead_seconds']),'--phase',phase,'--seeds',*[str(s) for s in seeds],'--log-every-seconds',str(args.log_every_seconds)]
 if args.resume: command.append('--resume')
 if args.allow_cpu_smoke: command.append('--allow-cpu-smoke')
 return command

def stop_workers(processes):
 for process in processes.values():
  if process.poll() is None:
   if os.name=='nt': process.terminate()
   else: process.send_signal(signal.SIGINT)
 deadline=time.monotonic()+12
 for process in processes.values():
  if process.poll() is None:
   try: process.wait(timeout=max(.1,deadline-time.monotonic()))
   except subprocess.TimeoutExpired: process.kill(); process.wait()

def run_stage(job,assignments,phase,seeds,args,emit):
 messages=queue.Queue(); processes={}; threads=[]; readers_done=set(); start=time.monotonic(); heartbeat=start
 def read_lines(gpu,stream):
  try:
   for line in stream: messages.put((gpu,line.rstrip('\r\n')))
  finally: stream.close(); messages.put((gpu,None))
 try:
  for assignment in assignments:
   gpu=assignment['gpu_id']; env=os.environ.copy()
   env.update({'CUDA_VISIBLE_DEVICES':'-1' if args.allow_cpu_smoke else gpu,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8',
     'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':str(args.cpu_threads),'MKL_NUM_THREADS':str(args.cpu_threads),'JCOU_CPU_THREADS':str(args.cpu_threads)})
   command=command_for(job,assignment,phase,seeds,args)
   options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
   process=subprocess.Popen(command,cwd=str(HERE),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
    text=True,encoding='utf-8',errors='replace',bufsize=1,**options)
   processes[gpu]=process; thread=threading.Thread(target=read_lines,args=(gpu,process.stdout),daemon=True); thread.start(); threads.append(thread)
   emit(f'[LAUNCH {phase}] GPU {gpu} lead={assignment["lead_seconds"]}s PID={process.pid}')
  while len(readers_done)<len(processes) or any(p.poll() is None for p in processes.values()):
   try:
    gpu,line=messages.get(timeout=.5)
    if line is None: readers_done.add(gpu)
    else: emit(f'[GPU {gpu}][{phase}] {line}')
   except queue.Empty: pass
   if time.monotonic()-heartbeat>=args.status_seconds:
    heartbeat=time.monotonic(); completed=sum(1 for _ in (job/phase).glob('*/lead_*/seed_*/COMPLETED.json')) if phase!='preflight' else 0
    states=[]
    for gpu,process in processes.items():
     progress=job/'workers'/f'gpu_{gpu}'/'progress.json'; detail=''
     try:
      obj=json.loads(progress.read_text(encoding='utf-8')); age=time.time()-obj.get('unix_time',time.time())
      detail=f" {obj.get('model','')} h={obj.get('lead_seconds','')} epoch={obj.get('epoch','-')} {obj.get('state','')} last_update={age:.0f}s"
     except (OSError,ValueError): pass
     states.append(f'GPU{gpu} '+('running' if process.poll() is None else f'exit={process.returncode}')+detail)
    expected=12 if phase=='smoke' else 12*len(seeds)
    emit(f'[STATUS {phase}] completed={completed}/{expected} elapsed={(heartbeat-start)/60:.1f}min | '+' | '.join(states))
  for thread in threads: thread.join(timeout=2)
  # Drain final buffered messages before reporting exit codes.
  while not messages.empty():
   gpu,line=messages.get_nowait()
   if line is not None: emit(f'[GPU {gpu}][{phase}] {line}')
  codes={gpu:process.wait() for gpu,process in processes.items()}; write_json(codes,job/f'{phase}_exit_codes.json')
  for gpu,code in codes.items(): emit(f'[EXIT {phase}] GPU {gpu}: code={code}')
  if any(code!=0 for code in codes.values()): raise RuntimeError(f'{phase} failed: {codes}; inspect terminal and workers/gpu_*/console.log')
 except BaseException:
  stop_workers(processes); raise

def estimate_time(job,assignments,seeds):
 import pandas as pd
 rows=[]
 for assignment in assignments:
  total=0.
  for model in ['rnn','lstm','gru']:
   run=job/f'smoke/{model}/lead_{assignment["lead_seconds"]}/seed_{seeds[0]}'
   log=pd.read_csv(run/'training_log.csv'); config=json.loads((run/'config.json').read_text(encoding='utf-8')); last=log.iloc[-1]
   train=last.training_seconds/math.ceil(config['train_target_count']/64)*math.ceil(config['common_targets_by_role']['train']/64)
   validation=last.validation_seconds/math.ceil(config['validation_target_count']/64)*math.ceil(config['common_targets_by_role']['validation']/64)
   total+=(train+validation)*200*len(seeds)
  rows.append({'GPU':assignment['gpu_id'],'lead_seconds':assignment['lead_seconds'],'full_fits':len(assignment['tasks']),
    'extrapolated_train_validation_hours_max200':total/3600})
 actual=[json.loads((job/'workers'/f'gpu_{a["gpu_id"]}'/'smoke_environment.json').read_text(encoding='utf-8')) for a in assignments]
 result={'per_GPU':rows,'execution_scope':'CPU_process_flow_check_only' if all(r['CPU_smoke_only'] for r in actual) else 'actual_GPU_smoke_extrapolation',
  'actual_devices':actual,'parallel_wall_hours_max200_linear_estimate':max(r['extrapolated_train_validation_hours_max200'] for r in rows),
  'limitation':'smoke batch-time extrapolation; shared-resource/I/O/checkpoint effects and early stopping not predicted; not completed full training'}
 write_json(result,job/'server_parallel_time_estimate.json'); return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--gpus',default=os.environ.get('GPU_IDS','0,1,2,3')); parser.add_argument('--job',default='new_forecasting_4gpu_v2')
 parser.add_argument('--resume',action='store_true'); parser.add_argument('--smoke-only',action='store_true'); parser.add_argument('--pilot',action='store_true')
 parser.add_argument('--allow-cpu-smoke',action='store_true',help='Local 4-process plumbing check only; requires --smoke-only')
 parser.add_argument('--log-every-seconds',type=float,default=10); parser.add_argument('--status-seconds',type=float,default=30); parser.add_argument('--cpu-threads',type=int,default=2)
 parser.add_argument('--plan-only',action='store_true',help='Print deterministic assignment; starts no training')
 args=parser.parse_args()
 if args.allow_cpu_smoke and not args.smoke_only: parser.error('--allow-cpu-smoke requires --smoke-only; full CPU execution is disabled')
 if args.cpu_threads<1 or args.status_seconds<=0 or args.log_every_seconds<=0: parser.error('thread counts/intervals must be positive')
 if not args.job or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.job): parser.error('Use a simple job name')
 seeds=[20261113] if args.pilot else list(range(20261113,20261123)); assignments=schedule(args.gpus.split(','),seeds)
 if args.plan_only:
  print(json.dumps({'expected_full_fits':12*len(seeds),'assignments':assignments},ensure_ascii=False,indent=2)); return
 job=Path(__file__).resolve().parents[1]/'results'/args.job; job.mkdir(parents=True,exist_ok=True)
 with exclusive_job(job), (job/'console.log').open('a',encoding='utf-8',buffering=1) as log:
  def emit(message):
   line=f'[{stamp()}] {message}'; print(line,flush=True); log.write(line+'\n'); log.flush()
  emit(f'START NEW forecasting A; four independent GPU processes; planned full fits={12*len(seeds)}; output={job}')
  for a in assignments: emit(f'GPU {a["gpu_id"]} -> lead {a["lead_seconds"]}s -> {len(a["tasks"])} fits')
  write_json({'PID':os.getpid(),'platform':platform.platform(),'start_KST':stamp(),'arguments':vars(args)},job/'launcher_latest.json')
  # Verify all four devices before preparing data or launching a full fit.
  run_stage(job,assignments,'preflight',seeds,args,emit)
  import pipeline
  prepared=job/'data_contract'; plan_path=job/'job_plan.json'
  if plan_path.exists():
   if not args.resume: raise FileExistsError(f'Job already planned: {job}; use --resume or a new --job')
   plan=json.loads(plan_path.read_text(encoding='utf-8'))
   if plan['pipeline_source_sha256']!=digest(SOURCE/'pipeline.py'): raise ValueError('Pipeline changed; use a new --job to preserve this experiment')
   if plan['contract_sha256']!=digest(prepared/'contract_and_locked_config.json'): raise ValueError('Locked data contract changed')
   old_tasks=[t for a in plan['assignments'] for t in a['tasks']]; new_tasks=[t for a in assignments for t in a['tasks']]
   if old_tasks!=new_tasks: raise ValueError('Task/model/lead/seed scope changed; use a new --job')
   pipeline.load_prepared_cache(prepared,plan['cache'],verify_raw=True)
   emit('Verified existing file split/scaler/common targets/cache/raw hashes; workers will resume without refitting the data contract.')
  else:
   emit('PREPARE raw records once; freezing common split/scaler/target timestamps and observation cache.')
   data,index,scaler,contract=pipeline.prepare(prepared,verbose=True); cache=pipeline.save_prepared_cache(data,prepared)
   plan={'pipeline_source_sha256':digest(SOURCE/'pipeline.py'),'assignments':assignments,'seeds':seeds,'expected_fits':12*len(seeds),'cache':cache,
    'contract_sha256':digest(prepared/'contract_and_locked_config.json'),'test_scope':'previously inspected records; NEW fitting/tuning held out; not new independent physical experiments'}
   write_json(plan,plan_path)
   snapshot=job/'source_snapshot'; snapshot.mkdir(exist_ok=True)
   for name in ['gpu_worker.py','run_4gpu.py','run_4gpu.sh']:
    shutil.copy2(HERE/name,snapshot/name)
   shutil.copy2(SOURCE/'pipeline.py',snapshot/'pipeline.py')
   shutil.copy2(pipeline.RAW_PARSER,snapshot/'audit_raw.py')
   shutil.copy2(ROOT/'requirements/training.txt',snapshot/'requirements_server.txt')
   del data,index,scaler
   emit(f'LOCKED files={contract["counts_by_role"]}; targets={contract["common_targets_by_role"]}; all workers read the same train-only scaler.')
  try:
   run_stage(job,assignments,'smoke',seeds,args,emit)
   estimate=estimate_time(job,assignments,seeds)
   if args.allow_cpu_smoke: emit('CPU process flow check: timings are not server GPU runtime evidence.')
   else:
    for row in estimate['per_GPU']: emit(f'[TIME ESTIMATE] GPU{row["GPU"]} h={row["lead_seconds"]}: {row["extrapolated_train_validation_hours_max200"]:.2f} hours for its full shard')
    emit(f'[TIME ESTIMATE] Four-GPU wall estimate={estimate["parallel_wall_hours_max200_linear_estimate"]:.2f}h; linear smoke extrapolation, before I/O/checkpoint/stopping effects.')
   if args.smoke_only:
    emit('SMOKE ONLY PASS: 4 processes / 12 two-epoch fits; no full training launched.'); return
   emit('All four GPU smoke shards passed. Starting/resuming full training; validation selects checkpoints; test is evaluated afterward.')
   try: run_stage(job,assignments,'full',seeds,args,emit)
   finally:
    from run_server import summarize
    summarize(job,seeds)
   status=json.loads((job/'RESULT_STATUS.json').read_text(encoding='utf-8'))
   if not status['all_complete']: raise RuntimeError(f'Incomplete full run: {status}')
   emit(f'ALL COMPLETE: {status["completed_fits"]}/{status["expected_fits"]}; metrics_all_runs.csv and run_status.csv ready.')
  except KeyboardInterrupt:
   emit('INTERRUPTED: workers stopped; rerun the same command with --resume. Completed checkpoints/results are preserved.'); raise
  except Exception as exc:
   emit(f'FAILED: {exc}. Completed runs are preserved; see per-GPU logs.'); raise

if __name__=='__main__': main()
