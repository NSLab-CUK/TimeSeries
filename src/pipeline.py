"""NEW strict forecasting pipeline; never a reconstruction of the June training run."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,hashlib,random,argparse,math,platform,shutil
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results'
DATA_ROOT=ROOT/'data/raw'
METADATA=ROOT/'metadata'
RAW_PARSER=Path(__file__).resolve().with_name('audit_raw.py')
from audit_raw import load_log,timestamps
FEATURES=['FNC-1','FNC-2','FNC-3','TI-2','TI-3','TI-4','TI-5','PI-2','PI-3','PI-4','PI-5','CO','CO2','CH4','H2']
GASES=['CO','CO2','CH4','H2']; LEADS=[1,3,5,10]; WINDOW=30
VAL_PREFIXES=['20210614','22.05.25','20210528','220608']
TEST_PREFIXES=['22.04.07','22.06.02']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(obj,p): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def atomic_save(obj,path):
 temporary=path.with_name(path.stem+'.tmp.pth'); torch.save(obj,temporary)
 try: os.replace(temporary,path)
 except PermissionError:
  if os.name!='nt': raise
  # Some Windows SFTP mounts reject replacement. Preserve the preceding state
  # before a checked copy; Linux server storage uses atomic replacement above.
  backup=path.with_name(path.stem+'.previous.pth')
  if path.exists(): shutil.copyfile(path,backup)
  shutil.copyfile(temporary,path)
  if sha(path)!=sha(temporary): raise IOError(f'Checkpoint copy mismatch: {path}')
  temporary.unlink()
def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)

class ForecastDataset(Dataset):
 def __init__(self,data,index,lead,scaler): self.data=data; self.index=index.reset_index(drop=True); self.lead=lead; self.sc=scaler
 def __len__(self): return len(self.index)
 def __getitem__(self,i):
  r=self.index.iloc[i]; values=self.data[r.experiment_id]['values']; target=int(r.raw_target_row); origin=target-self.lead
  x=values[origin-WINDOW+1:origin+1]
  y=values[target,-4:]
  return torch.tensor((x-self.sc['x_mean'])/self.sc['x_std'],dtype=torch.float32),torch.tensor((y-self.sc['y_mean'])/self.sc['y_std'],dtype=torch.float32)

class Forecaster(nn.Module):
 def __init__(self,kind,hidden=64):
  super().__init__(); self.recurrent={'rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU}[kind](len(FEATURES),hidden,batch_first=True,num_layers=1)
  self.head=nn.Linear(hidden,4)
 def forward(self,x): return self.head(self.recurrent(x)[0][:,-1])

def prepare(destination=None,verbose=False):
 dest=Path(destination) if destination else BASE/'prepared'; dest.mkdir(parents=True,exist_ok=True)
 manifest=pd.read_csv(METADATA/'data_manifest.csv'); records=[]; segments=[]; indices=[]; data={}; excluded=[]
 for r in manifest.itertuples():
  if not r.candidate_17_gasification_inventory: continue
  if verbose: print('PREPARE',r.experiment_id,flush=True)
  path=DATA_ROOT/r.source_file; d,enc,meta,first_line=load_log(path); ts,basis=timestamps(d,meta)
  if not ts.notna().any():
   excluded.append({'experiment_id':r.experiment_id,'source_file':r.source_file,'reason':'no_absolute_clock; cannot_verify_elapsed_lead; do_not_synthesize_timestamps'}); continue
  missing=[c for c in FEATURES if c not in d]
  if missing: raise ValueError((r.experiment_id,missing))
  role='test' if any(r.experiment_id.startswith(p) for p in TEST_PREFIXES) else 'validation' if any(r.experiment_id.startswith(p) for p in VAL_PREFIXES) else 'train'
  values=d[FEATURES].to_numpy(float); valid=np.isfinite(values).all(1)&ts.notna().to_numpy(); delta=ts.diff().dt.total_seconds().to_numpy()
  start=None; valid_segments=[]
  for i in range(len(d)+1):
   continuation=i<len(d) and valid[i] and (start is None or (i>0 and valid[i-1] and delta[i]==1))
   if start is not None and not continuation:
    valid_segments.append((start,i)); start=None
   if i<len(d) and valid[i] and start is None: start=i
  data[r.experiment_id]={'values':values,'timestamps':ts,'role':role,'source':r.source_file}
  n_targets=0
  for segment_id,(start,end) in enumerate(valid_segments):
   # Shared across all models AND all leads. Longest required span is 40 rows.
   targets=range(start+WINDOW-1+max(LEADS),end)
   segments.append({'experiment_id':r.experiment_id,'role':role,'segment_id':segment_id,'raw_start_row':start,'raw_end_row_exclusive':end,
     'start_timestamp':str(ts.iloc[start]),'end_timestamp':str(ts.iloc[end-1]),'n_rows':end-start,'n_common_targets':len(targets)})
   for target in targets:
    indices.append({'experiment_id':r.experiment_id,'role':role,'segment_id':segment_id,'raw_target_row':target,'target_timestamp':str(ts.iloc[target])})
    for h in LEADS:
     assert (ts.iloc[target]-ts.iloc[target-h]).total_seconds()==h
     assert target-h-WINDOW+1>=start
   n_targets+=len(targets)
  records.append({'experiment_id':r.experiment_id,'role':role,'source_file':r.source_file,'sha256':sha(path),
    'n_source_rows':len(d),'n_finite_clocked_rows':int(valid.sum()),'n_segments':len(valid_segments),'n_common_targets':n_targets,
    'negative_input_cells_preserved':int((values[valid]<0).sum()),'AC_status':'unverified_lab_confirmation_pending'})
 index=pd.DataFrame(indices); split=pd.DataFrame(records)
 assert split.experiment_id.nunique()==len(split) and set(split.role)=={'train','validation','test'}
 assert split.groupby('sha256').role.nunique().max()==1
 for a in ['train','validation','test']:
  for b in ['train','validation','test']:
   if a!=b: assert set(split[split.role==a].experiment_id).isdisjoint(set(split[split.role==b].experiment_id))
 # Fit once to unique finite observed rows of training files only, not repeated
 # windows and never validation/test. No clipping or negative-value exclusion.
 trainrows=[]
 for exp,obj in data.items():
  if obj['role']=='train':
   v=obj['values']; valid=np.isfinite(v).all(1)&obj['timestamps'].notna().to_numpy(); trainrows.append(v[valid])
 alltrain=np.concatenate(trainrows); xm=alltrain.mean(0); xs=alltrain.std(0); xs=np.where(xs<1e-12,1.,xs)
 ym=alltrain[:,-4:].mean(0); ys=alltrain[:,-4:].std(0); ys=np.where(ys<1e-12,1.,ys)
 scaler={'x_mean':xm,'x_std':xs,'y_mean':ym,'y_std':ys}
 split.to_csv(dest/'split_manifest.csv',index=False,encoding='utf-8-sig',lineterminator='\n'); pd.DataFrame(segments).to_csv(dest/'continuous_segments.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
 pd.DataFrame(excluded).to_csv(dest/'excluded_records.csv',index=False,encoding='utf-8-sig',lineterminator='\n'); index.to_csv(dest/'common_target_index.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
 np.savez(dest/'train_only_scaler.npz',**scaler)
 contract={'pipeline':'NEW_STRICT_FORECASTING_A; not_historical_reproduction','features':FEATURES,'targets':GASES,'window':WINDOW,
  'lead_seconds':LEADS,'process_and_gas_latest_time':'target_timestamp - h seconds','history_rows':'[target_row-h-window+1, target_row-h] inclusive',
  'target_gas_exposure':False,'target_time_process_exposure':False,'intermediate_future_gas_exposure':False,
  'mask':'none; no future rows are inserted','availability_assumption':'logged sensor/gas values acquired no later than origin; analyzer latency still requires laboratory confirmation',
  'flow_channels':'excluded_in_new_pipeline; uncertain physical definitions and missing-channel assumptions avoided',
  'record_rules':'exclude_filename_pyrolysis; absolute_clock_required; finite selected features; split at any non-1s step; retain negatives/spikes',
  'scaler_fit_roles':['train'],'scaler_fit_observation_rows':len(alltrain),'scaler_fit_file_ids':split[split.role=='train'].experiment_id.tolist(),
  'split_sha256':sha(dest/'split_manifest.csv'),'scaler_sha256':sha(dest/'train_only_scaler.npz'),
  'target_index_sha256':sha(dest/'common_target_index.csv'),'raw_parser_sha256':sha(RAW_PARSER),
  'hyperparameters':'predeclared single layer, hidden64, Adam lr0.001 weight_decay0.0001, L1 standardized target, batch64, max200, patience20',
  'tuning':'none; validation-only checkpoint selection; never select using test',
  'test_scope':'two historically inspected records held out from NEW fitting/tuning; not new physical or prospectively blind experiments',
  'counts_by_role':split.groupby('role').size().to_dict(),'common_targets_by_role':index.groupby('role').size().to_dict()}
 dump(contract,dest/'contract_and_locked_config.json')
 trace=[]
 for exp,obj in data.items():
  candidates=index[index.experiment_id==exp]
  if candidates.empty: continue
  # First target and first post-gap common target, when available.
  selected=[candidates.iloc[0]]
  post=candidates[candidates.raw_target_row>=1983]
  if len(post): selected.append(post.iloc[0])
  for r in selected:
   target=int(r.raw_target_row)
   for h in LEADS:
    origin=target-h
    for j in range(origin-WINDOW+1,origin+1):
     for c,name in enumerate(FEATURES):
      trace.append({'sample_id':f'{exp}_target{target}_h{h}','experiment_id':exp,'role':obj['role'],'lead_label':h,
       'prediction_origin_timestamp':str(obj['timestamps'].iloc[origin]),'target_timestamp':str(obj['timestamps'].iloc[target]),
       'input_timestamp':str(obj['timestamps'].iloc[j]),'raw_input_row':j,'raw_target_row':target,'feature_name':name,
       'feature_kind':'process' if c<11 else 'gas','is_masked':False,'input_value':float(obj['values'][j,c])})
 pd.DataFrame(trace).to_csv(dest/'actual_input_timestamp_trace.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
 return data,index,scaler,contract

def save_prepared_cache(data,destination):
 """Freeze raw observations once; workers only read this common cache."""
 destination=Path(destination); arrays={}; records=[]
 for i,(experiment,obj) in enumerate(data.items()):
  arrays[f'values_{i}']=obj['values']; arrays[f'timestamps_{i}']=obj['timestamps'].astype('int64').to_numpy()
  records.append({'experiment_id':experiment,'role':obj['role'],'source':obj['source']})
 np.savez_compressed(destination/'observations_cache.npz',**arrays)
 dump(records,destination/'observations_cache_records.json')
 return {'cache_sha256':sha(destination/'observations_cache.npz'),'cache_records_sha256':sha(destination/'observations_cache_records.json')}

def load_prepared_cache(destination,expected_cache=None,verify_raw=False):
 destination=Path(destination); contract=json.loads((destination/'contract_and_locked_config.json').read_text(encoding='utf-8'))
 for key,name in [('split_sha256','split_manifest.csv'),('scaler_sha256','train_only_scaler.npz'),('target_index_sha256','common_target_index.csv')]:
  if sha(destination/name)!=contract[key]: raise ValueError(f'Prepared artifact mismatch: {name}')
 if sha(RAW_PARSER)!=contract['raw_parser_sha256']: raise ValueError('Raw parser changed after preparation')
 if expected_cache:
  for key,name in [('cache_sha256','observations_cache.npz'),('cache_records_sha256','observations_cache_records.json')]:
   if sha(destination/name)!=expected_cache[key]: raise ValueError(f'Observation cache mismatch: {name}')
 split=pd.read_csv(destination/'split_manifest.csv')
 if verify_raw:
  for r in split.itertuples():
   if sha(DATA_ROOT/r.source_file)!=r.sha256: raise ValueError(f'Raw input changed: {r.source_file}')
 records=json.loads((destination/'observations_cache_records.json').read_text(encoding='utf-8')); data={}
 with np.load(destination/'observations_cache.npz',allow_pickle=False) as cached:
  for i,r in enumerate(records):
   data[r['experiment_id']]={'values':cached[f'values_{i}'],'timestamps':pd.Series(pd.to_datetime(cached[f'timestamps_{i}'])),
    'role':r['role'],'source':r['source']}
 index=pd.read_csv(destination/'common_target_index.csv')
 with np.load(destination/'train_only_scaler.npz',allow_pickle=False) as stored: scaler={k:stored[k] for k in stored.files}
 assert set(data)==set(split.experiment_id)
 for role,count in contract['common_targets_by_role'].items(): assert len(index[index.role==role])==count
 return data,index,scaler,contract

def cap_stratified(index,limit,seed):
 if len(index)<=limit: return index.copy()
 rng=np.random.default_rng(seed); groups=list(index.groupby('experiment_id')); selected=[]
 per=max(1,limit//len(groups))
 for exp,g in groups: selected.append(g.iloc[np.sort(rng.choice(len(g),min(per,len(g)),replace=False))])
 result=pd.concat(selected).sort_values(['experiment_id','raw_target_row']).head(limit)
 return result

def predict(model,loader,device,scaler):
 model.eval(); preds=[]; truth=[]
 with torch.inference_mode():
  for x,y in loader:
   preds.append(model(x.to(device)).cpu().numpy()); truth.append(y.numpy())
 p=np.concatenate(preds)*scaler['y_std']+scaler['y_mean']; t=np.concatenate(truth)*scaler['y_std']+scaler['y_mean']
 return t,p

def execute(data,index,scaler,contract,mode='smoke',models=None,leads=None,seeds=None,epochs=None,resume=False,output_root=None,prepared_root=None,
            summary_dir=None,log_every_seconds=10.,progress_file=None):
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(int(os.environ.get('JCOU_CPU_THREADS','4')))
 results=[]; timings=[]; wall_start=time.perf_counter()
 tasks=list(__import__('itertools').product(models or ['rnn','lstm','gru'],leads or LEADS,seeds or [20261113]))
 def progress(state):
  if progress_file: dump({**state,'unix_time':time.time(),'visible_GPU':os.environ.get('CUDA_VISIBLE_DEVICES')},Path(progress_file))
 for fit_number,(model_type,h,seed) in enumerate(tasks,1):
  label=f'{mode.upper()} {model_type.upper()} lead={h}s seed={seed} fit={fit_number}/{len(tasks)}'
  seed_all(seed); run=(Path(output_root) if output_root else BASE/'prepared')/f'{mode}/{model_type}/lead_{h}/seed_{seed}'
  if run.exists() and not resume: raise FileExistsError(f'Preserve existing run: {run}; choose a different output directory or --resume')
  run.mkdir(parents=True,exist_ok=True)
  train=index[index.role=='train'].copy(); val=index[index.role=='validation'].copy(); test=index[index.role=='test'].copy()
  if mode=='smoke':
   train=cap_stratified(train,512,seed); val=cap_stratified(val,256,seed); test=cap_stratified(test,128,seed)
  td=ForecastDataset(data,train,h,scaler); vd=ForecastDataset(data,val,h,scaler); sd=ForecastDataset(data,test,h,scaler)
  generator=torch.Generator().manual_seed(seed)
  tl=DataLoader(td,batch_size=64,shuffle=True,generator=generator); vl=DataLoader(vd,batch_size=64); sl=DataLoader(sd,batch_size=64)
  model=Forecaster(model_type).to(device); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4); loss_fn=nn.L1Loss()
  max_epochs=epochs or (2 if mode=='smoke' else 200)
  config={**contract,'model_type':model_type,'hidden_dim':64,'num_layers':1,'seed':seed,'lead_seconds':h,
   'mode':'flow_validation_smoke_not_accuracy_evidence' if mode=='smoke' else 'NEW_forecasting_not_historical_reproduction',
   'epochs':max_epochs,'patience':20,'train_target_count':len(td),'validation_target_count':len(vd),'test_target_count':len(sd),'device':str(device),
   'pipeline_source_sha256':sha(Path(__file__))}
  if (run/'config.json').exists():
   existing=json.loads((run/'config.json').read_text(encoding='utf-8'))
   for key in ['model_type','seed','lead_seconds','epochs','split_sha256','scaler_sha256','pipeline_source_sha256','target_index_sha256','raw_parser_sha256','mode']:
    if existing.get(key)!=config.get(key): raise ValueError(f'Resume provenance mismatch: {run}: {key}')
   if (run/'COMPLETED.json').exists():
    completed=json.loads((run/'COMPLETED.json').read_text(encoding='utf-8'))
    for name,digest in completed['artifact_sha256'].items():
     if not (run/name).exists() or sha(run/name)!=digest: raise ValueError(f'Completed artifact mismatch: {run/name}')
    print('SKIP verified completed',label,flush=True)
    progress({'state':'completed_skipped','fit_number':fit_number,'fit_total':len(tasks),'model':model_type,'lead_seconds':h,'seed':seed}); continue
  else: dump(config,run/'config.json')
  for artifact in ['split_manifest.csv','train_only_scaler.npz','contract_and_locked_config.json']:
   shutil.copy2((Path(prepared_root) if prepared_root else BASE/'prepared')/artifact,run/artifact)
  for role,idx in [('train',train),('validation',val),('test',test)]: idx.to_csv(run/f'{role}_targets.csv',index=False,encoding='utf-8-sig')
  shutil.copy2(Path(__file__),run/'pipeline_source.py')
  print('START',label,f'targets(train/val/test)={len(td)}/{len(vd)}/{len(sd)} device={device}',flush=True)
  progress({'state':'starting','fit_number':fit_number,'fit_total':len(tasks),'model':model_type,'lead_seconds':h,'seed':seed})
  dump({'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
   'CUDA_available':torch.cuda.is_available(),'GPU':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
   'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'scope':'actual_execution_machine'},run/'environment.json')
  logs=[]; best=float('inf'); train_seconds=0.; stale=0; start_epoch=0
  if resume and (run/'last.pth').exists():
   try: last=torch.load(run/'last.pth',map_location=device,weights_only=True)
   except Exception:
    if not (run/'last.previous.pth').exists(): raise
    print('Recovering preceding Windows-mount checkpoint backup',run,flush=True)
    last=torch.load(run/'last.previous.pth',map_location=device,weights_only=True)
   model.load_state_dict(last['model_state_dict']); opt.load_state_dict(last['optimizer_state_dict'])
   start_epoch=last['epoch']+1; logs=last['logs']; best=last['best_validation_loss']; stale=last['stale']; generator.set_state(last['loader_generator_state'].cpu())
   torch.set_rng_state(last['torch_rng_state'].cpu()); random.setstate(last['python_rng_state'])
   nr=last['numpy_rng_state']; np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
   if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in last['cuda_rng_state']])
   print('RESUME',model_type,h,seed,'next_epoch',start_epoch,flush=True)
  for epoch in range(start_epoch,max_epochs):
   if mode=='full' and stale>=20: break
   model.train(); running=0.; n=0; torch.cuda.synchronize() if device.type=='cuda' else None; start=time.perf_counter(); heartbeat=start
   for batch_number,(x,y) in enumerate(tl,1):
    x=x.to(device); y=y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); running+=loss.item()*len(x); n+=len(x)
    if time.perf_counter()-heartbeat>=log_every_seconds:
     heartbeat=time.perf_counter()
     print(label,f'epoch={epoch+1:03d}/{max_epochs} batch={batch_number}/{len(tl)} train_L1={running/n:.6f} elapsed={heartbeat-start:.1f}s',flush=True)
     progress({'state':'training','model':model_type,'lead_seconds':h,'seed':seed,'epoch':epoch+1,'batch':batch_number,'batches':len(tl)})
   torch.cuda.synchronize() if device.type=='cuda' else None; elapsed=time.perf_counter()-start; train_seconds+=elapsed
   model.eval(); losses=[]; validation_start=time.perf_counter()
   with torch.inference_mode():
    for x,y in vl: losses.append((loss_fn(model(x.to(device)),y.to(device)).item(),len(x)))
   val_loss=sum(v*n for v,n in losses)/sum(n for v,n in losses)
   validation_elapsed=time.perf_counter()-validation_start
   logs.append({'epoch':epoch,'train_MAE_scaled':running/n,'validation_MAE_scaled':val_loss,'training_seconds':elapsed,'validation_seconds':validation_elapsed})
   if val_loss<best:
    best=val_loss; stale=0; atomic_save({'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':config,
     'epoch':epoch,'validation_MAE_scaled':val_loss,'scaler':{k:v.tolist() for k,v in scaler.items()},'split_sha256':contract['split_sha256']},run/'best.pth')
   else: stale+=1
   pd.DataFrame(logs).to_csv(run/'training_log.csv',index=False)
   nr=np.random.get_state()
   atomic_save({'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'logs':logs,'stale':stale,
    'best_validation_loss':best,'loader_generator_state':generator.get_state(),'torch_rng_state':torch.get_rng_state(),'python_rng_state':random.getstate(),
    'numpy_rng_state':(nr[0],nr[1].tolist(),nr[2],nr[3],nr[4]),'cuda_rng_state':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},run/'last.pth')
   memory=f' GPU_allocated={torch.cuda.memory_allocated()/1024**2:.0f}MiB' if device.type=='cuda' else ''
   print(label,f'epoch={epoch+1:03d}/{max_epochs} train_L1={running/n:.6f} val_L1={val_loss:.6f} best={best:.6f} patience={stale}/20 train={elapsed:.2f}s val={validation_elapsed:.2f}s'+memory,flush=True)
   progress({'state':'epoch_complete','fit_number':fit_number,'fit_total':len(tasks),'model':model_type,'lead_seconds':h,'seed':seed,
    'epoch':epoch+1,'train_L1':running/n,'validation_L1':val_loss,'best_validation_L1':best,'patience_count':stale})
   if mode=='full' and stale>=20: break
  pd.DataFrame(logs).to_csv(run/'training_log.csv',index=False)
  checkpoint=torch.load(run/'best.pth',map_location=device,weights_only=True); model.load_state_dict(checkpoint['model_state_dict'])
  print('EVALUATE and checkpoint replay',label,'selected_epoch',checkpoint['epoch']+1,flush=True)
  inverse_truth,pred=predict(model,sl,device,scaler)
  truth=np.stack([data[r.experiment_id]['values'][r.raw_target_row,-4:] for r in test.itertuples()])
  inverse_roundtrip_difference=float(abs(truth-inverse_truth).max())
  np.save(run/'true.npy',truth); np.save(run/'pred.npy',pred)
  output=[]
  for i,r in enumerate(test.itertuples()):
   for c,gas in enumerate(GASES): output.append({'experiment_id':r.experiment_id,'model':model_type,'lead_seconds':h,'seed':seed,
    'target_timestamp':r.target_timestamp,'origin_timestamp':str(data[r.experiment_id]['timestamps'].iloc[r.raw_target_row-h]),'raw_target_row':r.raw_target_row,
    'gas':gas,'true':truth[i,c],'prediction':pred[i,c],'mode':config['mode']})
  pd.DataFrame(output).to_csv(run/'predictions_with_timestamps.csv',index=False,encoding='utf-8-sig')
  scores=[]
  for exp,group in test.reset_index(drop=True).groupby('experiment_id'):
   rows=group.index.to_numpy()
   for c,gas in enumerate(GASES):
    t=truth[rows,c]; p=pred[rows,c]; mse=float(np.mean((t-p)**2)); ss=float(np.sum((t-t.mean())**2))
    scores.append({'experiment_id':exp,'model':model_type,'lead_seconds':h,'seed':seed,'gas':gas,'n_samples':len(rows),
      'MAE':float(np.mean(abs(t-p))),'MSE':mse,'RMSE':math.sqrt(mse),'R2':1-float(np.sum((t-p)**2))/ss if ss else float('nan'),
      'mode':config['mode'],'unit_basis':'raw_source_concentration_units; laboratory_confirmation_pending'})
  pd.DataFrame(scores).to_csv(run/'metrics_per_record_gas.csv',index=False,encoding='utf-8-sig')
  # Reload checkpoint and regenerate to test artifact completeness.
  replay=Forecaster(model_type).to(device); replay.load_state_dict(checkpoint['model_state_dict']); tr2,pr2=predict(replay,sl,device,scaler)
  difference=float(abs(pr2-pred).max()); assert difference<1e-5
  full_batches=math.ceil(contract['common_targets_by_role']['train']/64); measured_batches=len(tl)
  # Extrapolation only, training step wall-time after first-epoch CUDA warmup.
  perbatch=logs[-1]['training_seconds']/measured_batches; epoch_est=perbatch*full_batches
  timings.append({'model':model_type,'lead_seconds':h,'smoke_epochs':2,'smoke_train_samples':len(td),'smoke_batches_per_epoch':len(tl),
   'last_smoke_epoch_training_seconds':logs[-1]['training_seconds'],'full_train_batches_per_epoch':full_batches,
   'extrapolated_full_epoch_training_seconds':epoch_est,'extrapolated_200_epoch_training_seconds_per_fit':200*epoch_est,
   'limitation':'linear_batch_extrapolation; excludes_validation_io_and_early_stopping; not_completed_full_training'})
  results.append({'model':model_type,'lead_seconds':h,'seed':seed,'checkpoint_reload_max_pred_difference':difference,
    'train_test_file_overlap':False,'timestamp_contract_passed':True,'same_target_rows_across_models_and_leads':True,
    'full_training_executed':mode=='full','mode':config['mode']})
  dump({'checkpoint_reload_max_pred_difference':difference,'target_inverse_roundtrip_max_difference':inverse_roundtrip_difference,
   'true_values_saved_directly_from_raw_rows':True,'full_training_executed':mode=='full','epochs_executed':len(logs)},run/'verification.json')
  required=['best.pth','last.pth','config.json','train_targets.csv','validation_targets.csv','test_targets.csv','train_only_scaler.npz','split_manifest.csv',
   'pipeline_source.py','training_log.csv','true.npy','pred.npy','predictions_with_timestamps.csv','metrics_per_record_gas.csv','verification.json','environment.json']
  dump({'completed':True,'scope':config['mode'],'artifact_sha256':{name:sha(run/name) for name in required}},run/'COMPLETED.json')
  print(mode.upper(),'PASS',model_type,h,'epochs',len(logs),'checkpoint replay diff',difference,flush=True)
  progress({'state':'completed','fit_number':fit_number,'fit_total':len(tasks),'model':model_type,'lead_seconds':h,'seed':seed,
   'epochs_executed':len(logs),'checkpoint_replay_difference':difference})
 summary_root=Path(summary_dir) if summary_dir else Path(output_root) if output_root else BASE/'prepared'
 summary_root.mkdir(parents=True,exist_ok=True)
 if mode!='smoke':
  if results: pd.DataFrame(results).to_csv(summary_root/f'{mode}_execution_validation.csv',index=False)
  return
 # Preserve the first completed 12-fit benchmark when an additional check uses another seed.
 suffix='' if seeds is None else '_additional'
 if not results: return
 pd.DataFrame(results).to_csv(summary_root/f'smoke_validation{suffix}.csv',index=False)
 pd.DataFrame(timings).to_csv(summary_root/f'timing_estimates{suffix}.csv',index=False)
 hardware={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
  'CUDA_available':torch.cuda.is_available(),'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'GPU_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
  'scope':'current_smoke_machine_only; not_historical_benchmark','smoke_wall_seconds':time.perf_counter()-wall_start,
  'new_full_fit_scope':{'minimum_all_model_lead_points':12,'ten_fixed_split_seed_repetitions':120},
  'extrapolated_training_seconds_12_fits_max200':sum(t['extrapolated_200_epoch_training_seconds_per_fit'] for t in timings),
  'extrapolated_training_seconds_120_fits_max200':10*sum(t['extrapolated_200_epoch_training_seconds_per_fit'] for t in timings)}
 dump(hardware,summary_root/f'hardware_and_execution_scope{suffix}.json')

if __name__=='__main__':
 parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(); mode.add_argument('--smoke',action='store_true'); mode.add_argument('--full',action='store_true')
 parser.add_argument('--models',nargs='+',choices=['rnn','lstm','gru']); parser.add_argument('--leads',nargs='+',type=int,choices=LEADS)
 parser.add_argument('--seeds',nargs='+',type=int); parser.add_argument('--epochs',type=int); parser.add_argument('--resume',action='store_true'); parser.add_argument('--output-root',type=Path); args=parser.parse_args()
 data,index,scaler,contract=prepare(); print('Prepared NEW forecasting A',contract['counts_by_role'],contract['common_targets_by_role'],flush=True)
 if args.smoke or args.full: execute(data,index,scaler,contract,'full' if args.full else 'smoke',args.models,args.leads,args.seeds,args.epochs,args.resume,args.output_root)
