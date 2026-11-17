"""NEW completed experiments: explicit seed/case reductions and traceable paper figures."""
from pathlib import Path
import json,hashlib,math,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]; JOB=BASE/'training'; FIG=BASE/'figures'; TABLE=BASE/'tables'
for p in [FIG,TABLE]: p.mkdir(exist_ok=True)
GASES=['CO','CO2','CH4','H2']; MODELS=['rnn','lstm','gru']; LEADS=[1,3,5,10]; SEEDS=list(range(20261113,20261123)); COLORS={'rnn':'#2463a6','lstm':'#d77418','gru':'#21835c'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,stem):
 for ext in ['png','svg','pdf']: fig.savefig(FIG/f'{stem}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
df=pd.read_csv(BASE/'metrics_per_record_seed_gas.csv'); audit=pd.read_csv(BASE/'training_run_audit.csv')
METRICS=['MAE','MSE','RMSE','R2']
perseed=df.groupby(['model','lead_seconds','seed','gas'])[METRICS].mean().reset_index()
perseed['aggregation']='arithmetic_mean_of_two_record_metrics_within_same_fitted_seed'
perseed.to_csv(TABLE/'metrics_equal_record_per_seed.csv',index=False,encoding='utf-8-sig')
summary=perseed.groupby(['model','lead_seconds','gas'])[METRICS].agg(['mean','std']).reset_index(); summary.columns=['_'.join(x).strip('_') for x in summary.columns]
summary['n_fitted_seeds']=10; summary['n_test_records']=2; summary['SD_definition']='sample_SD_ddof1_across_10_seedwise_two_record_mean_scores; not_CI'
summary.to_csv(TABLE/'metrics_equal_record_seed_summary.csv',index=False,encoding='utf-8-sig')
case=df.groupby(['experiment_id','model','lead_seconds','gas'])[METRICS].agg(['mean','std']).reset_index(); case.columns=['_'.join(x).strip('_') for x in case.columns]
case.to_csv(TABLE/'metrics_per_record_seed_summary.csv',index=False,encoding='utf-8-sig')
test=pd.read_csv(JOB/'data_contract/common_target_index.csv'); test=test[test.role=='test'].reset_index(drop=True); experiments=test.experiment_id.unique().tolist()
def date(exp): return '2022-04-07' if exp.startswith('22.04.07') else '2022-06-02'
data={}; records=json.loads((JOB/'data_contract/observations_cache_records.json').read_text(encoding='utf-8'))
with np.load(JOB/'data_contract/observations_cache.npz',allow_pickle=False) as z:
 for i,r in enumerate(records): data[r['experiment_id']]={**r,'values':z[f'values_{i}'],'timestamps':pd.Series(pd.to_datetime(z[f'timestamps_{i}']))}
predictions={}; true=np.load(JOB/'full/rnn/lead_1/seed_20261113/true.npy'); plotframes=[]; sources=[]; pooled=[]
for model in MODELS:
 for h in LEADS:
  path=JOB/f'full/{model}/lead_{h}/seed_20261113/pred.npy'; pred=np.load(path); predictions[model,h]=pred
  f=test[['experiment_id','raw_target_row','target_timestamp']].copy(); f['model']=model; f['lead_seconds']=h; f['seed']=20261113
  f['origin_timestamp']=[str(data[r.experiment_id]['timestamps'].iloc[r.raw_target_row-h]) for r in test.itertuples()]
  for c,g in enumerate(GASES): f[g+'_true']=true[:,c]; f[g+'_prediction']=pred[:,c]
  plotframes.append(f); sources.append({'model':model,'lead_seconds':h,'seed':20261113,'prediction_file':str(path.relative_to(BASE)),'sha256':sha(path),'selection':'first_predeclared_seed; not_performance_selected'})
  for seed in ([] if (BASE/'metrics_pooled_targets_per_seed.csv').exists() else SEEDS):
   p=np.load(JOB/f'full/{model}/lead_{h}/seed_{seed}/pred.npy')
   for c,g in enumerate(GASES):
    t=true[:,c]; yh=p[:,c]; mse=np.mean((t-yh)**2)
    pooled.append({'model':model,'lead_seconds':h,'seed':seed,'gas':g,'n_targets':len(t),'MAE':np.mean(abs(t-yh)),'MSE':mse,'RMSE':np.sqrt(mse),'R2':1-np.sum((t-yh)**2)/np.sum((t-t.mean())**2),'aggregation':'pooled_targets; distinct_from_equal_record_means'})
pd.concat(plotframes).to_csv(FIG/'prediction_plot_values_seed20261113.csv.gz',index=False,compression='gzip')
pd.DataFrame(sources).to_csv(FIG/'prediction_figure_sources.csv',index=False,encoding='utf-8-sig')
if pooled: pd.DataFrame(pooled).to_csv(TABLE/'metrics_pooled_targets_per_seed.csv',index=False,encoding='utf-8-sig')
else: shutil.copy2(BASE/'metrics_pooled_targets_per_seed.csv',TABLE/'metrics_pooled_targets_per_seed.csv')
baseline=pd.read_csv(BASE/'persistence_baseline.csv'); baseavg=baseline.groupby(['lead_seconds','gas'])[METRICS].mean().reset_index()
comparison=summary.merge(baseavg[['lead_seconds','gas','RMSE']].rename(columns={'RMSE':'persistence_equal_record_RMSE'}),on=['lead_seconds','gas'])
comparison['relative_RMSE_reduction_percent']=100*(comparison.persistence_equal_record_RMSE-comparison.RMSE_mean)/comparison.persistence_equal_record_RMSE
comparison.to_csv(TABLE/'persistence_comparison.csv',index=False,encoding='utf-8-sig')

def lines(ax,t,y,**kw):
 t=pd.to_datetime(t).reset_index(drop=True); breaks=np.flatnonzero(t.diff().dt.total_seconds().fillna(1).to_numpy()!=1); bounds=[0,*breaks,len(t)]
 for n,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
  opts=dict(kw)
  if n: opts.pop('label',None)
  ax.plot(t.iloc[a:b],np.asarray(y)[a:b],**opts)
def trajectories(stem,gas,leads):
 c=GASES.index(gas); fig,axes=plt.subplots(len(leads),2,figsize=(12,3.2*len(leads)),squeeze=False)
 for i,h in enumerate(leads):
  for j,exp in enumerate(experiments):
   ax=axes[i,j]; mask=(test.experiment_id==exp).to_numpy(); x=test.loc[mask,'target_timestamp']
   lines(ax,x,true[mask,c],color='.1',lw=1.2,label='Actual')
   for model in MODELS: lines(ax,x,predictions[model,h][mask,c],color=COLORS[model],lw=.8,alpha=.85,label=model.upper())
   ax.set_title(f'({chr(97+i*2+j)}) {date(exp)}, h={h}s',loc='left',fontsize=12)
   ax.set_xlabel('Recorded target clock'); ax.set_ylabel(f'{gas} (source units)'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); ax.grid(alpha=.2); ax.margins(x=.01,y=.1)
   if i==0: ax.legend(fontsize=9,ncol=4,loc='upper right')
 fig.suptitle(stem.split('_')[0]+' — revised forecasts, fixed seed 20261113',fontsize=15)
 fig.text(.04,.005,'Same 7,266 test targets across all models and leads. Lines break across omitted/gap rows; no clipping.\n'
   'Numeric concentrations retain source scale; laboratory unit basis and analyzer timing require confirmation.',fontsize=9)
 fig.tight_layout(rect=(0,.065 if len(leads)<3 else .035,1,.94 if len(leads)<3 else .97)); save(fig,stem)
trajectories('Fig4_H2_predictions','H2',[1,10]); trajectories('S1_H2_lead3','H2',[3]); trajectories('S2_H2_lead5','H2',[5])
trajectories('S3_CO_predictions','CO',LEADS); trajectories('S4_CO2_predictions','CO2',LEADS); trajectories('S5_CH4_predictions','CH4',LEADS)

fig,axs=plt.subplots(1,3,figsize=(13,4.8)); fig5source=[]; bounds=[]
for panel,metric in [(0,'RMSE'),(1,'R2')]:
 ax=axs[panel]
 for j,model in enumerate(MODELS):
  s=summary[(summary.model==model)&(summary.gas=='H2')].set_index('lead_seconds').loc[LEADS]
  x=np.arange(4)+(j-1)*.08; means=s[metric+'_mean'].to_numpy(); sd=s[metric+'_std'].to_numpy()
  ax.errorbar(x,means,yerr=sd,color=COLORS[model],marker='o',capsize=3,lw=1.7,label=model.upper())
  for k,h in enumerate(LEADS):
   values=perseed[(perseed.model==model)&(perseed.lead_seconds==h)&(perseed.gas=='H2')][metric].to_numpy()
   ax.scatter(x[k]+np.linspace(-.025,.025,10),values,s=9,color=COLORS[model],alpha=.35)
   fig5source.append({'panel':'ab'[panel],'model':model,'lead_seconds':h,'metric':metric,'mean':means[k],'SD_ddof1_seedwise_case_mean':sd[k],'lower':means[k]-sd[k],'upper':means[k]+sd[k],'point_min':values.min(),'point_max':values.max()})
 b=baseavg[baseavg.gas=='H2'].set_index('lead_seconds').loc[LEADS]
 ax.plot(range(4),b[metric],color='.35',ls='--',marker='x',lw=1.4,label='Persistence')
 for k,h in enumerate(LEADS):
  value=float(b.loc[h,metric]); fig5source.append({'panel':'ab'[panel],'model':'persistence','lead_seconds':h,'metric':metric,'mean':value,'SD_ddof1_seedwise_case_mean':0.,'lower':value,'upper':value,'point_min':value,'point_max':value})
 ax.set_xticks(range(4),LEADS); ax.set_xlabel('Lead h (actual seconds)'); ax.set_title(f'({"ab"[panel]}) H₂ {metric}',loc='left',fontsize=13)
 ax.set_ylabel('RMSE (source units)' if metric=='RMSE' else 'R²'); ax.legend(frameon=False,fontsize=10); ax.grid(alpha=.2); ax.margins(y=.15)
 if metric=='R2': ax.axhline(0,color='.6',ls='--',lw=.7)
ax=axs[2]
for j,model in enumerate(MODELS):
 vals=[]; sd=[]
 for exp in experiments:
  s=case[(case.experiment_id==exp)&(case.model==model)&(case.lead_seconds==1)&(case.gas=='H2')].iloc[0]; vals.append(s.MAE_mean); sd.append(s.MAE_std)
  fig5source.append({'panel':'c','model':model,'experiment_id':exp,'lead_seconds':1,'metric':'MAE','mean':s.MAE_mean,'SD_ddof1_seedwise_case_mean':s.MAE_std,'lower':s.MAE_mean-s.MAE_std,'upper':s.MAE_mean+s.MAE_std})
 ax.errorbar(np.arange(2)+(j-1)*.17,vals,yerr=sd,fmt='o',color=COLORS[model],capsize=4,label=model.upper())
ax.set_xticks(range(2),[date(e) for e in experiments]); ax.set_title('(c) H₂ MAE at h=1s',loc='left',fontsize=13); ax.set_ylabel('MAE (source units)'); ax.legend(frameon=False,fontsize=10); ax.grid(alpha=.2); ax.margins(x=.25,y=.15)
fig.suptitle('Fig. 5 — completed revised experiments',fontsize=15)
fig.text(.025,.01,'(a,b) Each fit: equal mean of two record scores; points and sample SD across ten fixed-split seeds.\n'
 '(c) Record-specific mean ± sample SD over ten seeds. Dashed: last-observation persistence. Bars are not confidence intervals.',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94)); fig.canvas.draw()
for k,ax in enumerate(axs):
 panel='abc'[k]; rows=[r for r in fig5source if r['panel']==panel]; lo=min(min(r['lower'],r.get('point_min',r['lower'])) for r in rows); hi=max(max(r['upper'],r.get('point_max',r['upper'])) for r in rows)
 a,b=ax.get_ylim(); assert a<=lo and hi<=b
 bounds.append({'panel':panel,'axis_min':a,'axis_max':b,'data_and_errorbars_min':lo,'data_and_errorbars_max':hi,'all_visible':True})
pd.DataFrame(fig5source).to_csv(FIG/'Fig5_source_values.csv',index=False,encoding='utf-8-sig'); (FIG/'Fig5_axis_validation.json').write_text(json.dumps(bounds,indent=2),encoding='utf-8'); save(fig,'Fig5_metrics')

fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for ax,exp in zip(axs,experiments):
 for j,model in enumerate(MODELS):
  s=case[(case.experiment_id==exp)&(case.model==model)&(case.gas=='H2')].set_index('lead_seconds').loc[LEADS]
  ax.errorbar(np.arange(4)+(j-1)*.06,s.RMSE_mean,yerr=s.RMSE_std,color=COLORS[model],marker='o',capsize=3,label=model.upper())
 b=baseline[(baseline.experiment_id==exp)&(baseline.gas=='H2')].set_index('lead_seconds').loc[LEADS]
 ax.plot(range(4),b.RMSE,color='.35',ls='--',marker='x',label='Persistence'); ax.set_xticks(range(4),LEADS); ax.set_title(date(exp)); ax.set_ylabel('H₂ RMSE (source units)'); ax.set_xlabel('Lead (seconds)'); ax.grid(alpha=.2); ax.legend(fontsize=9)
fig.suptitle('S6 — record-specific H₂ errors and persistence reference'); fig.tight_layout(rect=(0,0,1,.94)); save(fig,'S6_record_errors')
fig,axs=plt.subplots(2,2,figsize=(12,9))
for ax,g in zip(axs.ravel(),GASES):
 for j,model in enumerate(MODELS):
  s=summary[(summary.model==model)&(summary.gas==g)].set_index('lead_seconds').loc[LEADS]
  ax.errorbar(np.arange(4)+(j-1)*.06,s.RMSE_mean,yerr=s.RMSE_std,color=COLORS[model],marker='o',capsize=3,label=model.upper())
 b=baseavg[baseavg.gas==g].set_index('lead_seconds').loc[LEADS]; ax.plot(range(4),b.RMSE,color='.35',ls='--',marker='x',label='Persistence')
 ax.set_xticks(range(4),LEADS); ax.set_title(g); ax.set_xlabel('Lead (seconds)'); ax.set_ylabel('RMSE (source units)'); ax.grid(alpha=.2); ax.legend(fontsize=9)
fig.suptitle('S7 — all-gas equal-record errors, mean ± ten-seed sample SD'); fig.tight_layout(rect=(0,0,1,.95)); save(fig,'S7_all_gas_metrics')
fig,axs=plt.subplots(3,4,figsize=(14,10))
for i,model in enumerate(MODELS):
 for j,h in enumerate(LEADS):
  ax=axs[i,j]
  for seed in SEEDS:
   log=pd.read_csv(JOB/f'full/{model}/lead_{h}/seed_{seed}/training_log.csv'); x=log.epoch+1
   ax.plot(x,log.train_MAE_scaled,color='.45',ls='--',alpha=.3,lw=.65); ax.plot(x,log.validation_MAE_scaled,color=COLORS[model],alpha=.3,lw=.8)
  ax.set_title(f'{model.upper()}, h={h}s',fontsize=12); ax.set_xlabel('Actual epoch'); ax.set_ylabel('Scaled-target L1'); ax.set_yscale('log'); ax.grid(alpha=.2)
fig.suptitle('S8 — actual histories for all 120 fits; dashed training / colored validation',fontsize=14); fig.tight_layout(rect=(0,0,1,.95)); save(fig,'S8_training_histories')

FEATURES=['FNC-1','FNC-2','FNC-3','TI-2','TI-3','TI-4','TI-5','PI-2','PI-3','PI-4','PI-5',*GASES]
observed=[]; within=[]; population=[]
split=pd.read_csv(JOB/'data_contract/split_manifest.csv')
for r in split[split.role=='train'].itertuples():
 o=data[r.experiment_id]; v=o['values']; finite=np.isfinite(v).all(1)&o['timestamps'].notna().to_numpy(); f=pd.DataFrame(v[finite],columns=FEATURES); observed.append(f)
 c=f.corr(method='spearman').loc[FEATURES[:11],GASES]
 for sensor in c.index:
  for gas in GASES: within.append({'experiment_id':r.experiment_id,'sensor':sensor,'gas':gas,'Spearman_rho':c.loc[sensor,gas],'n_rows':len(f)})
 population.append({'experiment_id':r.experiment_id,'source_file':r.source_file,'source_sha256':r.sha256,'role':'train','n_rows':len(f)})
pooled=pd.concat(observed).corr(method='spearman').loc[FEATURES[:11],GASES]
average=pd.DataFrame(within).groupby(['sensor','gas']).Spearman_rho.mean().unstack().reindex(index=FEATURES[:11],columns=GASES)
pd.DataFrame(within).to_csv(FIG/'Fig6_per_record_spearman.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(population).to_csv(FIG/'Fig6_population.csv',index=False,encoding='utf-8-sig')
pooled.to_csv(FIG/'Fig6_pooled_spearman.csv',encoding='utf-8-sig'); average.to_csv(FIG/'Fig6_equal_record_spearman.csv',encoding='utf-8-sig')
fig,axs=plt.subplots(1,2,figsize=(10,7))
for ax,values,title in zip(axs,[pooled,average],['(a) Pooled observations','(b) Equal-record mean of within-record rho']):
 im=ax.imshow(values.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto'); ax.set_xticks(range(4),GASES); ax.set_yticks(range(11),FEATURES[:11]); ax.set_title(title,fontsize=11)
 for i in range(11):
  for j in range(4):
   v=values.iloc[i,j]; ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=10,color='white' if abs(v)>.65 else 'black')
fig.colorbar(im,ax=axs.tolist(),label='Spearman rho',fraction=.035,pad=.035); fig.suptitle('Fig. 6 — ten training records only; no causal interpretation',fontsize=13)
fig.subplots_adjust(left=.1,right=.83,bottom=.1,top=.9,wspace=.7); save(fig,'Fig6_training_correlations')

fig,ax=plt.subplots(figsize=(11,5)); ax.axis('off')
ax.text(.02,.95,'Fig. 2 — past-only gas-composition forecasting',fontsize=18,weight='bold',va='top')
for y,text in [(.73,'(a) Input: 30 consecutive observed rows ending at prediction origin t'),(.52,'11 logged sensor channels + past CO, CO2, CH4, H2 → one recurrent layer'),(.32,'(b) Final hidden state (64) → linear head → four gases at t+h'),(.12,'h = 1, 3, 5, 10 actual seconds; no measurements after t are supplied')]:
 ax.text(.03,y,text,fontsize=14,bbox={'boxstyle':'round','facecolor':'#e8f1fb','edgecolor':'#2463a6'})
save(fig,'Fig2_forecasting_definition')
fig,ax=plt.subplots(figsize=(11,5.2)); ax.axis('off'); ax.text(.02,.96,'Algorithm 1 — executed revised protocol',fontsize=18,weight='bold',va='top')
algorithm=['1  Lock 10 training / 4 validation / 2 test files and seed list.',
 '2  Split at invalid rows or any non-1-s timestamp interval; retain negatives.',
 '3  Fit input/target scalers only on unique training observations.',
 '4  Use past 30 rows through t to predict Y(t+h); share targets across leads.',
 '5  Train RNN/LSTM/GRU; select checkpoint by validation L1; patience 20.',
 '6  Evaluate fixed test targets; inverse-transform predictions; no clipping.',
 '7  Save model, optimizer/RNG states, scalers, timestamps, logs and metrics.']
for i,line in enumerate(algorithm): ax.text(.03,.8-i*.11,line,fontsize=13,va='top')
save(fig,'Algorithm1_executed')
oldfig=ROOT/'revision/JCOU_D_26_00869_resolution/figures/Table1_logged_examples.png'
if oldfig.exists(): shutil.copy2(oldfig,FIG/oldfig.name)
else: assert (FIG/oldfig.name).exists(),'Provide the frozen source-header Table1 figure with the analysis bundle'
training_summary=audit.groupby(['model','lead_seconds']).agg(n_seeds=('seed','size'),epochs_min=('epochs_executed','min'),epochs_median=('epochs_executed','median'),epochs_max=('epochs_executed','max'),best_epoch_min=('best_epoch_one_based','min'),best_epoch_max=('best_epoch_one_based','max')).reset_index()
training_summary.to_csv(TABLE/'training_epoch_summary.csv',index=False,encoding='utf-8-sig')
manifest=[{'figure':p.stem,'file':str(p.relative_to(BASE)),'sha256':sha(p),'source':'completed_NEW_runs_or_explicit_training_population','legacy_logo_bitmap':False} for p in sorted(FIG.glob('*.png'))]
pd.DataFrame(manifest).to_csv(FIG/'figure_manifest.csv',index=False,encoding='utf-8-sig')
print('H2 equal-record mean + seed SD\n',summary[summary.gas=='H2'][['model','lead_seconds','RMSE_mean','RMSE_std','R2_mean','R2_std']].to_string(index=False),flush=True)
print('H2 persistence comparison\n',comparison[comparison.gas=='H2'][['model','lead_seconds','persistence_equal_record_RMSE','relative_RMSE_reduction_percent']].to_string(index=False),flush=True)
print('Completed tables and figures.',flush=True)
