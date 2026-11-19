"""Permission-aware NEW experiment reproduction entry points. Never train implicitly."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parent
TRAINER=ROOT/'src'
DATA_ROOT=ROOT/'data/raw'
GASES=['CO','CO2','CH4','H2']; METRICS=['MAE','MSE','RMSE','R2']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pipeline_module():
    sys.path.insert(0,str(TRAINER))
    import pipeline
    return pipeline
def new_output(p):
    p=Path(p)
    if p.exists() and any(p.iterdir()): raise FileExistsError(f'Output must be empty; preserve existing results: {p}')
    p.mkdir(parents=True,exist_ok=True)
    return p
def cache(job):
    import pandas as pd
    pipe=pipeline_module(); prepared=job/'data_contract'
    data,index,scaler,contract=pipe.load_prepared_cache(prepared,verify_raw=False)
    return pipe,data,index[index.role=='test'].reset_index(drop=True),scaler,contract
def scores(y,p):
    import numpy as np
    y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
    mse=float(np.mean((y-p)**2)); denom=float(np.sum((y-y.mean())**2))
    return dict(MAE=float(np.mean(abs(y-p))),MSE=mse,RMSE=mse**.5,
        R2=1-float(np.sum((y-p)**2))/denom if denom else float('nan'))
def inspect(args):
    import pandas as pd
    split=pd.read_csv(ROOT/'metadata/split_manifest.csv')
    manifest=pd.read_csv(ROOT/'metadata/data_manifest.csv')
    missing=[r.source_file for r in manifest.itertuples() if r.candidate_17_gasification_inventory and not (DATA_ROOT/r.source_file).is_file()]
    print(json.dumps(dict(scope='NEW forecasting; not historical execution reproduction',
        file_roles=split.groupby('role').size().to_dict(),full_fits=120,
        raw_records_included=False,missing_permissioned_record_paths=missing,
        checkpoint_files_included=False,repository_status='code and experiment metadata'),ensure_ascii=False,indent=2))
def prepare(args):
    import numpy as np
    import pandas as pd
    pipe=pipeline_module(); out=new_output(args.output)
    # Fail on source drift before constructing samples. Do not silently accept a different population.
    wanted=pd.read_csv(ROOT/'metadata/split_manifest.csv')
    for r in wanted.itertuples():
        if not (DATA_ROOT/r.source_file).exists(): raise FileNotFoundError(f'Author permission and source record required: {r.source_file}')
        if sha(DATA_ROOT/r.source_file)!=r.sha256: raise ValueError(f'Raw-source hash differs: {r.source_file}')
    data,index,scaler,contract=pipe.prepare(out,verbose=True)
    actual=pd.read_csv(out/'split_manifest.csv')
    pd.testing.assert_frame_equal(actual,wanted,check_dtype=False)
    target=pd.read_csv(out/'common_target_index.csv')
    expected=json.loads((ROOT/'metadata/data_contract_digests.json').read_text(encoding='utf-8'))
    assert sha(out/'common_target_index.csv')==expected['common_target_index_sha256']
    assert all(hashlib.sha256(np.ascontiguousarray(scaler[k],dtype='<f8').tobytes()).hexdigest()==expected['scaler_array_sha256'][k] for k in scaler)
    pipe.save_prepared_cache(data,out)
    (out/'reproduction_check.json').write_text(json.dumps(dict(passed=True,
        split_matches=True,all_target_timestamps_match=True,scaler_max_difference=0,
        training_started=False),indent=2),encoding='utf-8')
    print('PASS raw hashes, 10/4/2 roles, all common targets and exact training-only scaler. No fitting.')
def infer(args):
    import numpy as np
    import pandas as pd
    import torch
    pipe,data,index,scaler,contract=cache(args.job_dir)
    run=args.job_dir/f'full/{args.model}/lead_{args.lead}/seed_{args.seed}'
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    if (config['model_type'],config['lead_seconds'],config['seed'])!=(args.model,args.lead,args.seed): raise ValueError('Run/config mismatch')
    completed=json.loads((run/'COMPLETED.json').read_text(encoding='utf-8'))
    if sha(run/'best.pth')!=completed['artifact_sha256']['best.pth']: raise ValueError('Checkpoint hash mismatch')
    pd.testing.assert_frame_equal(pd.read_csv(run/'test_targets.csv'),index,check_dtype=False)
    cp=torch.load(run/'best.pth',map_location='cpu',weights_only=True)
    for k in scaler:
        assert np.array_equal(np.array(cp['scaler'][k]),scaler[k])
    pipe.seed_all(args.seed); torch.set_num_threads(2)
    device=torch.device(args.device)
    model=pipe.Forecaster(args.model,config['hidden_dim']).to(device)
    model.load_state_dict(cp['model_state_dict'])
    loader=torch.utils.data.DataLoader(pipe.ForecastDataset(data,index,args.lead,scaler),batch_size=64,shuffle=False)
    _,p=pipe.predict(model,loader,device,scaler)
    y=np.stack([data[r.experiment_id]['values'][r.raw_target_row,-4:] for r in index.itertuples()])
    stored=np.load(run/'pred.npy',allow_pickle=False)
    diff=float(np.max(abs(stored-p)))
    if diff>args.tolerance: raise ValueError(f'Replay differs by {diff}; tolerance={args.tolerance}; hardware/software rounding may differ')
    out=new_output(args.output); np.save(out/'true.npy',y); np.save(out/'pred.npy',p)
    frame=index.copy(); frame['origin_timestamp']=[str(data[r.experiment_id]['timestamps'].iloc[r.raw_target_row-args.lead]) for r in index.itertuples()]
    for c,g in enumerate(GASES): frame[g+'_truth']=y[:,c]; frame[g+'_prediction']=p[:,c]
    frame.to_csv(out/'predictions_with_timestamps.csv',index=False)
    report=dict(model=args.model,lead_seconds=args.lead,seed=args.seed,n_targets=len(y),
        maximum_prediction_difference=diff,tolerance=args.tolerance,checkpoint_sha256=sha(run/'best.pth'),device=str(device),
        note='No fitting; CPU/other device replay may differ within declared numerical tolerance. Archived same-GPU audit was exact.')
    (out/'inference_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report))
def evaluate(args):
    import numpy as np
    import pandas as pd
    pipe,data,index,scaler,contract=cache(args.job_dir)
    y=np.stack([data[r.experiment_id]['values'][r.raw_target_row,-4:] for r in index.itertuples()])
    rows=[]; linked=[]
    for model,h,seed in itertools.product(['rnn','lstm','gru'],[1,3,5,10],range(20261113,20261123)):
        run=args.job_dir/f'full/{model}/lead_{h}/seed_{seed}'
        if not (run/'COMPLETED.json').exists(): raise FileNotFoundError(run/'COMPLETED.json')
        record=json.loads((run/'COMPLETED.json').read_text(encoding='utf-8'))
        for name in ['pred.npy','true.npy','test_targets.csv','config.json','best.pth','train_only_scaler.npz','training_log.csv']:
            assert sha(run/name)==record['artifact_sha256'][name],(run,name)
        cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
        assert (cfg['model_type'],cfg['lead_seconds'],cfg['seed'])==(model,h,seed)
        pd.testing.assert_frame_equal(index,pd.read_csv(run/'test_targets.csv'),check_dtype=False)
        p=np.load(run/'pred.npy',allow_pickle=False)
        assert p.shape==y.shape and np.isfinite(p).all() and np.array_equal(y,np.load(run/'true.npy',allow_pickle=False))
        linked.append(dict(model=model,lead_seconds=h,seed=seed,run=run.relative_to(args.job_dir).as_posix(),
            checkpoint_sha256=sha(run/'best.pth'),predictions_sha256=sha(run/'pred.npy'),test_targets_sha256=sha(run/'test_targets.csv')))
        for exp,g in index.groupby('experiment_id',sort=False):
            ix=g.index.to_numpy()
            for c,gas in enumerate(GASES): rows.append(dict(experiment_id=exp,model=model,lead_seconds=h,seed=seed,gas=gas,n_samples=len(ix),**scores(y[ix,c],p[ix,c])))
    baseline=[]
    for h in [1,3,5,10]:
        p=np.stack([data[r.experiment_id]['values'][r.raw_target_row-h,-4:] for r in index.itertuples()])
        for exp,g in index.groupby('experiment_id',sort=False):
            ix=g.index.to_numpy()
            for c,gas in enumerate(GASES): baseline.append(dict(experiment_id=exp,lead_seconds=h,gas=gas,n_samples=len(ix),**scores(y[ix,c],p[ix,c])))
    out=new_output(args.output); df=pd.DataFrame(rows); bf=pd.DataFrame(baseline)
    df.to_csv(out/'metrics_per_record_seed_gas.csv',index=False); bf.to_csv(out/'persistence_baseline.csv',index=False)
    perseed=df.groupby(['model','lead_seconds','seed','gas'])[METRICS].mean().reset_index()
    perseed.to_csv(out/'metrics_equal_record_per_seed.csv',index=False)
    summary=perseed.groupby(['model','lead_seconds','gas'])[METRICS].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') for c in summary.columns]
    summary.to_csv(out/'metrics_summary.csv',index=False)
    pd.DataFrame(linked).to_csv(out/'fit_artifact_links.csv',index=False)
    check=dict(fitted_runs_read=len(linked),per_record_seed_gas_metrics=len(df),test_targets_per_run=len(index),
        aggregation='mean of 2 record metrics within seed; mean and sample SD(ddof=1) across 10 seeds',
        unit='source concentration units, pending laboratory basis',training_started=False)
    (out/'evaluation_check.json').write_text(json.dumps(check,indent=2),encoding='utf-8'); print(json.dumps(check))
def plot(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.read_csv(args.evaluation_dir/'metrics_summary.csv')
    baseline=pd.read_csv(args.evaluation_dir/'persistence_baseline.csv').groupby(['lead_seconds','gas']).RMSE.mean()
    out=new_output(args.output); fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,g in zip(axs.ravel(),GASES):
        for m in ['rnn','lstm','gru']:
            s=df[(df.model==m)&(df.gas==g)].sort_values('lead_seconds')
            ax.errorbar(s.lead_seconds,s.RMSE_mean,yerr=s.RMSE_std,marker='o',capsize=3,label=m.upper())
        ax.plot([1,3,5,10],[baseline.loc[h,g] for h in [1,3,5,10]],'k--',label='Persistence')
        ax.set(title=g,xlabel='Actual lead (seconds)',ylabel='Equal-record RMSE (source units)'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Fixed 2-record test; mean and sample SD over 10 optimization seeds; not confidence intervals')
    fig.tight_layout()
    for ext in ['png','pdf','svg']: fig.savefig(out/f'all_gas_persistence.{ext}',dpi=180)
    print('Saved the four-gas RMSE comparison plots.')
def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('inspect'); s.set_defaults(func=inspect)
    s=sub.add_parser('prepare'); s.add_argument('--output',type=Path,required=True); s.set_defaults(func=prepare)
    s=sub.add_parser('infer'); s.add_argument('--job-dir',type=Path,required=True); s.add_argument('--model',choices=['rnn','lstm','gru'],required=True)
    s.add_argument('--lead',type=int,choices=[1,3,5,10],required=True); s.add_argument('--seed',type=int,default=20261113)
    s.add_argument('--device',default='cpu'); s.add_argument('--tolerance',type=float,default=1e-5); s.add_argument('--output',type=Path,required=True); s.set_defaults(func=infer)
    s=sub.add_parser('evaluate'); s.add_argument('--job-dir',type=Path,required=True); s.add_argument('--output',type=Path,required=True); s.set_defaults(func=evaluate)
    s=sub.add_parser('plot'); s.add_argument('--evaluation-dir',type=Path,required=True); s.add_argument('--output',type=Path,required=True); s.set_defaults(func=plot)
    args=p.parse_args(); args.func(args)
if __name__=='__main__': main()
