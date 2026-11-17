"""Inventory supplied raw logs and match archived true arrays by value sequence."""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import re
import numpy as np
import pandas as pd

GASES=('CO','CO2','CH4','H2')


def load_log(path):
    if path.suffix.lower()=='.xlsx':
        raw=pd.read_excel(path,header=None,dtype=object)
        enc='xlsx_openpyxl'
    else:
        for enc in ('utf-8-sig','cp949','utf-16'):
            try:
                raw=pd.read_csv(path,header=None,encoding=enc,dtype=object)
                break
            except UnicodeError:
                continue
        else: raise ValueError(f'No supported encoding: {path}')
    header_i=next(i for i in range(min(12,len(raw)))
                  if 'H2' in [str(v).strip() for v in raw.iloc[i]]
                  and any('feeding' in str(v).lower() for v in raw.iloc[i]))
    headers=[str(x).strip() for x in raw.iloc[header_i]]
    data=raw.iloc[header_i+1:].copy(); data.columns=headers
    data=data.apply(lambda c:pd.to_numeric(c,errors='coerce'))
    return data.reset_index(drop=True),enc,str(raw.iloc[0,0]),header_i+2


def timestamps(data,metadata):
    date_numbers=[int(x) for x in re.findall(r'\d+',metadata)[:3]]
    if len(date_numbers)!=3 or date_numbers[0]<1900:
        return pd.Series(pd.NaT,index=data.index), 'date_metadata_unparsed'
    base=pd.Timestamp(year=date_numbers[0],month=date_numbers[1],day=date_numbers[2])
    hc=next(c for c in data if c.strip().lower()=='hour')
    mc=next(c for c in data if c.strip().lower().startswith('min'))
    sc=next(c for c in data if c.strip().lower().startswith('sec'))
    secs=data[hc]*3600+data[mc]*60+data[sc]
    # Increment calendar day only for an observed clock rollover, not arbitrary gaps.
    roll=((secs.diff() < -43200).fillna(False)).cumsum()
    return base+pd.to_timedelta(secs+roll*86400,unit='s'),'raw_date_metadata_plus_Hour_Min_Sec'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3])
    ap.add_argument('--revision-dir',type=Path,default=Path(__file__).resolve().parents[1])
    args=ap.parse_args(); root=args.project_root.resolve(); out=args.revision_dir.resolve(); ev=out/'evidence'
    raw_dirs=[p for p in root.iterdir() if p.is_dir() and p.name not in ('revision','ReslutTables','Figures') and any(p.glob('*.csv'))]
    if len(raw_dirs)!=1: raise SystemExit('Specify exactly one unambiguous raw-log directory.')
    rawdir=raw_dirs[0]; manifest=[]; flow=[]; cadence=[]; raw={}; units=[]; raw_anomalies=[]
    for path in sorted(list(rawdir.glob('*.csv'))+list(rawdir.glob('*.xlsx'))):
        d,enc,metadata,first_line=load_log(path); ts,basis=timestamps(d,metadata)
        rel=path.relative_to(root).as_posix(); raw[path.stem]=(d,ts,path,first_line)
        lower=path.name.lower(); pyro='pyro' in lower
        atmosphere='pyrolysis' if pyro else 'O2/steam' if 'o2' in lower and 'steam' in lower else 'CO2' if 'co2' in lower else 'steam' if 'steam' in lower else 'air' if 'air' in lower else 'unverified'
        ac='without_AC_filename' if any(x in lower for x in ('without','non ac','non-ac','non_a','non a','non a.c','non-a.c')) else 'with_AC_filename' if any(x in lower for x in ('with ac','withac','with_ac')) else 'unverified'
        gaps=ts.diff().dt.total_seconds(); valid=gaps.dropna()
        complete=d.notna().all(axis=1)
        valid_times=ts.dropna()
        gases=d.loc[:,list(GASES)].to_numpy(float)
        role='archived_evaluation_case' if (root/'ReslutTables'/'lead_time_1'/'rnn'/path.stem).exists() else 'training_validation_or_unused_unverified'
        manifest.append({'experiment_id':path.stem,'source_file':rel,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'encoding':enc,'atmosphere_from_filename':atmosphere,'AC_from_filename':ac,
            'condition_confirmation':'filename_and_supplied_presentation_only; laboratory_confirmation_pending',
            'raw_date_metadata':metadata,'raw_start_timestamp':str(valid_times.iloc[0]) if len(valid_times) else 'unverified',
            'raw_end_timestamp':str(valid_times.iloc[-1]) if len(valid_times) else 'unverified',
            'raw_duration_seconds':(valid_times.iloc[-1]-valid_times.iloc[0]).total_seconds() if len(valid_times) else '',
            'n_raw_rows':len(d),'n_rows_with_clock_timestamp':int(ts.notna().sum()),
            'all_numeric_fields_missing_rows':int(d.isna().all(axis=1).sum()),
            'complete_CO_CO2_CH4_H2_rows':int(d.loc[:,list(GASES)].notna().all(axis=1).sum()),
            'n_complete_all_supplied_columns':int(complete.sum()),'missing_cells':int(d.isna().sum().sum()),
            'timestamp_basis':basis,'one_second_steps':int((valid==1).sum()),'non_one_second_steps':int((valid!=1).sum()),
            'candidate_17_gasification_inventory':not pyro,'candidate_selection_rule':'exclude_4_filename_pyrolysis_records; actual_training_selection_unverified',
            'split_role':role,'used_start_timestamp':'see raw_array_alignment.csv for archived evaluation suffixes; otherwise unverified',
            'used_end_timestamp':'see raw_array_alignment.csv; otherwise unverified','linked_paper_figures':'see figure_provenance.csv; panel_source_mapping_unverified',
            'column_names':json.dumps(list(d.columns),ensure_ascii=False)})
        for i in np.flatnonzero(((gaps!=1)&gaps.notna()).to_numpy()):
            cadence.append({'experiment_id':path.stem,'row_index_before':int(i-1),'row_index_after':int(i),
                            'timestamp_before':str(ts.iloc[i-1]),'timestamp_after':str(ts.iloc[i]),'delta_seconds':float(gaps.iloc[i])})
        for col in d:
            if any(k in col.lower() for k in ('flow','feeding')):
                vals=d[col].dropna()
                flow.append({'experiment_id':path.stem,'column':col,'n':len(vals),'n_unique':int(vals.nunique()),
                    'min':float(vals.min()),'max':float(vals.max()),'mean':float(vals.mean()),'std_ddof1':float(vals.std()),
                    'constant_within_file':vals.nunique()==1})
        for gas in GASES:
            vals=d[gas].to_numpy(float); jumps=np.abs(np.diff(vals)); pos=int(np.nanargmax(jumps))+1
            raw_anomalies.append({'experiment_id':path.stem,'gas':gas,'min':float(np.nanmin(vals)),
                'max':float(np.nanmax(vals)),'negative_count':int((vals<0).sum()),'largest_absolute_step':float(jumps[pos-1]),
                'step_row_index':pos,'timestamp':str(ts.iloc[pos]),'before':float(vals[pos-1]),'after':float(vals[pos]),
                'cause':'unverified; preserve measurement'})
        print('Raw log:',path.name,'rows',len(d),'non-1s',int((valid!=1).sum()),flush=True)
    pd.DataFrame(manifest).to_csv(out/'data_manifest.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(flow).to_csv(ev/'raw_flow_and_feeding.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cadence).to_csv(ev/'raw_timestamp_irregularities.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(raw_anomalies).to_csv(ev/'raw_signal_diagnostics.csv',index=False,encoding='utf-8-sig')
    # Actual true-array values are linked to raw gas values, including column order.
    arrays=pd.read_csv(ev/'array_manifest.csv'); matches=[]; rowlinks=[]
    for r in arrays.itertuples():
        d,ts,path,first_line=raw[r.experiment_id]
        t=np.load(root/r.true_path,allow_pickle=False).astype(float); g=d.loc[:,list(GASES)].to_numpy(float)
        best=(float('inf'),None)
        # Full-length contiguous match; do not silently drop NaNs or rows.
        for offset in range(len(g)-len(t)+1):
            error=float(np.max(abs(g[offset:offset+len(t)]-t)))
            if np.isfinite(error) and error<best[0]: best=(error,offset)
        error,offset=best; good=offset is not None and error<=1e-5
        matches.append({'experiment_id':r.experiment_id,'model':r.model,'lead_label':r.lead_label,'run':r.run,
            'raw_file':path.relative_to(root).as_posix(),'raw_n_rows':len(g),'array_n_rows':len(t),
            'raw_offset_zero_based':offset,'raw_csv_first_data_line_one_based':first_line,
            'raw_csv_first_matched_line_one_based':first_line+offset if offset is not None else '',
            'max_absolute_gas_difference':error,'full_contiguous_value_match_atol_1e_5':good,
            'matched_start_timestamp':str(ts.iloc[offset]) if good else 'unverified',
            'matched_end_timestamp':str(ts.iloc[offset+len(t)-1]) if good else 'unverified',
            'matched_has_non_1s_steps':int((ts.iloc[offset:offset+len(t)].diff().dt.total_seconds().dropna()!=1).sum()) if good else '',
            'target_order_status':'CO_CO2_CH4_H2_matches_raw' if good else 'unverified',
            'inverse_scale_status':'saved_true_matches_raw_concentrations; prediction_inverse_transform_not_independently_verified',
            'prediction_timestamp_status':'association_by_paired_array_row; model_target_index_still_unverified',
            'window_candidate_if_N_minus_w_minus_h_plus_1':len(g)-len(t)-int(r.lead_label)+1,
            'window_status':'inferred_candidate_only_no_dataset_code'})
        if good and r.run==0:
            # Raw timestamps recovered from full-sequence value matching. They are
            # not fabricated from row_index * 1 second.
            for i in range(len(t)):
                rowlinks.append({'experiment_id':r.experiment_id,'model':r.model,'lead_label':r.lead_label,
                    'run':r.run,'array_row_index':i,'raw_row_index':offset+i,'raw_timestamp':str(ts.iloc[offset+i]),
                    'status':'true_array_value_match; prediction_target_generation_unverified'})
    pd.DataFrame(matches).to_csv(ev/'raw_array_alignment.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rowlinks).to_csv(ev/'raw_array_row_timestamps_run0.csv',index=False,encoding='utf-8-sig')
    # Enrich canonical metric provenance; numerical metric values are unchanged.
    alignment=pd.DataFrame(matches)
    join=alignment[['experiment_id','model','lead_label','run','full_contiguous_value_match_atol_1e_5',
                    'raw_offset_zero_based','matched_start_timestamp','matched_end_timestamp']]
    metric_path=out/'metrics_per_run.csv'
    metrics_df=pd.read_csv(metric_path)
    extra=[c for c in join if c not in ('experiment_id','model','lead_label','run')]
    metrics_df=metrics_df.drop(columns=[c for c in extra if c in metrics_df])
    metrics_df=metrics_df.merge(join,on=['experiment_id','model','lead_label','run'],validate='many_to_one')
    metrics_df['units_status']='raw_concentration_scale_supported_by_true_array_match; volpercent_document_label; prediction_scaler_unverified'
    metrics_df['inverse_scaling_status']='saved_true_matches_raw; prediction_inverse_pipeline_unverified'
    metrics_df['timestamp_alignment_status']='true_rows_matched_to_raw_timestamps; prediction_target_generation_unverified'
    metrics_df.to_csv(metric_path,index=False,encoding='utf-8-sig')
    summaries=pd.read_csv(out/'metrics_summary.csv')
    summaries['units_status']=metrics_df.units_status.iloc[0]
    summaries.to_csv(out/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    common_path=out/'metrics_common_value_suffix_sensitivity.csv'
    common=pd.read_csv(common_path)
    common=common.drop(columns=[c for c in extra if c in common]).merge(join,on=['experiment_id','model','lead_label','run'],validate='many_to_one')
    common['common_raw_start_row']=common.raw_offset_zero_based+common.native_n_samples-common.n_samples
    common['units_status']=metrics_df.units_status.iloc[0]
    common['timestamp_alignment_status']='common_raw_target_rows_matched; model_input_target_generation_unverified'
    common.to_csv(common_path,index=False,encoding='utf-8-sig')
    common_summary=pd.read_csv(out/'metrics_common_value_suffix_summary.csv')
    common_summary['status']='common_raw_reference_targets_matched; model_input_target_generation_unverified'
    common_summary.to_csv(out/'metrics_common_value_suffix_summary.csv',index=False,encoding='utf-8-sig')
    validations=[]
    for (experiment,lead,run),g in common.groupby(['experiment_id','lead_label','run']):
        assert g.common_raw_start_row.nunique()==1
        validations.append({'experiment_id':experiment,'lead_label':lead,'run':run,
            'common_raw_start_row':int(g.common_raw_start_row.iloc[0]),'n_common_rows':int(g.n_samples.iloc[0]),
            'model_count':g.model.nunique(),'same_raw_reference_target_rows':True,
            'model_input_and_target_generation':'unverified'})
    pd.DataFrame(validations).to_csv(ev/'common_raw_target_interval_validation.csv',index=False,encoding='utf-8-sig')
    sample_trace=alignment[alignment.run==0][['experiment_id','model','lead_label','run','raw_offset_zero_based','matched_start_timestamp']].copy()
    sample_trace['input_process_timestamps']='unverified_no_original_dataset_code_or_input_tensors'
    sample_trace['input_gas_timestamps']='unverified_no_original_dataset_code_or_input_tensors'
    sample_trace['mask_location_and_method']='unverified'
    sample_trace['A_or_B']='unverified; Fig2d_schematic_is_B_for_1step_only'
    sample_trace.to_csv(ev/'sample_target_timestamp_trace.csv',index=False,encoding='utf-8-sig')
    fd=pd.DataFrame(flow); md=pd.DataFrame(manifest); candidate_ids=set(md[md.candidate_17_gasification_inventory].experiment_id)
    feeds=fd[fd.experiment_id.isin(candidate_ids)&fd.column.str.contains('feeding',case=False)].groupby('experiment_id')['mean'].mean()
    summary={'n_raw_experiments':len(md),'n_filename_pyrolysis':int((~md.candidate_17_gasification_inventory).sum()),
        'n_candidate_gasification':len(candidate_ids),'candidate_atmosphere_counts':md[md.candidate_17_gasification_inventory].atmosphere_from_filename.value_counts().to_dict(),
        'actual_17_selected_for_training':'unverified_no_selection_or_split_files',
        'candidate17_equal_experiment_feeding_mean_g_min':float(feeds.mean()),
        'candidate17_feeding_sample_sd_g_min_ddof1':float(feeds.std(ddof=1)),
        'n_archived_arrays_matched_to_raw':sum(r['full_contiguous_value_match_atol_1e_5'] for r in matches),
        'n_archived_arrays':len(matches),'input_window_timing_and_masking':'unverified_no_original_code_or_inputs'}
    (ev/'raw_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
