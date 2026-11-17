"""Final baseline-only audit. Reuse the completed checkpoint audit; never fit/replay models."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

B = Path(__file__).resolve().parents[1]
OUT = B / 'final_submission' / 'evidence'
OUT.mkdir(parents=True, exist_ok=True)
GASES = ['CO', 'CO2', 'CH4', 'H2']
METRICS = ['MAE', 'MSE', 'RMSE', 'R2']

def scores(y, p):
    e = np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    mse = np.mean(e * e)
    ss = np.sum((y - np.mean(y)) ** 2)
    return dict(MAE=np.mean(abs(e)), MSE=mse, RMSE=np.sqrt(mse),
                R2=1-np.sum(e*e)/ss if ss else np.nan)

def main():
    index = pd.read_csv(B/'training/data_contract/common_target_index.csv')
    index = index[index.role == 'test'].reset_index(drop=True)
    segments = pd.read_csv(B/'training/data_contract/continuous_segments.csv')
    records = json.loads((B/'training/data_contract/observations_cache_records.json').read_text(encoding='utf-8'))
    data = {}
    with np.load(B/'training/data_contract/observations_cache.npz', allow_pickle=False) as z:
        for i, r in enumerate(records):
            data[r['experiment_id']] = (z[f'values_{i}'], pd.to_datetime(z[f'timestamps_{i}']))
    truth = np.load(B/'training/full/rnn/lead_1/seed_20261113/true.npy')
    actual = np.stack([data[r.experiment_id][0][r.raw_target_row, -4:] for r in index.itertuples()])
    assert np.array_equal(actual, truth)
    # One common target list was already verified against all 120 saved run lists.
    prior = json.loads((B/'evidence/posttraining_validation.json').read_text(encoding='utf-8'))
    assert prior['verified_fits'] == 120 and prior['target_hash_counts_by_role']['test'] == 1
    traces, metrics, checks = [], [], []
    for h in [1, 3, 5, 10]:
        p = np.empty_like(truth)
        for n, r in enumerate(index.itertuples()):
            v, ts = data[r.experiment_id]
            target = int(r.raw_target_row); origin = target-h; first = origin-29
            seg = segments[(segments.experiment_id == r.experiment_id) & (segments.segment_id == r.segment_id)].iloc[0]
            assert first >= seg.raw_start_row and target < seg.raw_end_row_exclusive
            assert np.isfinite(v[first:target+1]).all()
            assert np.all(np.diff(ts[first:target+1].asi8) == 1_000_000_000)
            assert (ts[target]-ts[origin]).total_seconds() == h
            assert origin < target
            # Exactly the last row of the model's observed history, in original scale.
            p[n] = v[origin, -4:]
            traces.append(dict(experiment_id=r.experiment_id, lead_seconds=h,
                segment_id=r.segment_id, raw_input_first_row=first, raw_origin_row=origin,
                raw_target_row=target, input_first_timestamp=str(ts[first]),
                origin_timestamp=str(ts[origin]), target_timestamp=str(ts[target]),
                actual_lead_seconds=h, **{g+'_truth':truth[n,c] for c,g in enumerate(GASES)},
                **{g+'_persistence':p[n,c] for c,g in enumerate(GASES)}))
        for exp, group in index.groupby('experiment_id', sort=False):
            ix = group.index.to_numpy()
            checks.append(dict(experiment_id=exp, lead_seconds=h, n_targets=len(ix),
                gap_crossings=0, future_or_target_gas_reads=0, masked_gas_reads=0))
            for c,g in enumerate(GASES):
                metrics.append(dict(experiment_id=exp, lead_seconds=h, gas=g,
                    n_samples=len(ix), model='persistence_last_observation', **scores(truth[ix,c],p[ix,c])))
    m = pd.DataFrame(metrics)
    old = pd.read_csv(B/'persistence_baseline.csv')
    keys = ['experiment_id','lead_seconds','gas']
    delta = abs(m.set_index(keys)[METRICS].sort_index() - old.set_index(keys)[METRICS].sort_index()).max().max()
    assert delta < 1e-12, delta
    pd.DataFrame(traces).to_csv(OUT/'persistence_target_origin_trace.csv.gz',index=False,compression='gzip')
    pd.DataFrame(checks).to_csv(OUT/'persistence_interval_checks.csv',index=False,encoding='utf-8-sig')
    m.to_csv(OUT/'persistence_recomputed.csv',index=False,encoding='utf-8-sig')
    nn = pd.read_csv(B/'metrics_per_record_seed_gas.csv')
    seed = nn.groupby(['model','lead_seconds','seed','gas'])[METRICS].mean().reset_index()
    means = seed.groupby(['model','lead_seconds','gas'])[METRICS].mean().reset_index()
    avg = m.groupby(['lead_seconds','gas'])[METRICS].mean().reset_index()
    comp = means.merge(avg[['lead_seconds','gas','RMSE']].rename(columns={'RMSE':'persistence_equal_record_RMSE'}),on=['lead_seconds','gas'])
    comp['neural_minus_persistence_RMSE'] = comp.RMSE-comp.persistence_equal_record_RMSE
    selected = comp[((comp.gas=='H2') & (comp.lead_seconds==10)) | (comp.gas=='CO2')]
    assert (selected.neural_minus_persistence_RMSE>0).all() and len(selected)==15
    comp.to_csv(OUT/'persistence_final_comparison.csv',index=False,encoding='utf-8-sig')
    case = nn.groupby(['experiment_id','model','lead_seconds','gas']).RMSE.mean().reset_index()
    case = case.merge(m[keys+['RMSE']].rename(columns={'RMSE':'persistence_RMSE'}),on=keys)
    case['neural_minus_persistence_RMSE'] = case.RMSE-case.persistence_RMSE
    case.to_csv(OUT/'persistence_case_comparison.csv',index=False,encoding='utf-8-sig')
    report = dict(passed=True,scope='baseline-only final audit; previous 120-checkpoint replay reused',
        test_records=2,targets_per_lead=len(index),leads=[1,3,5,10],all_origin_target_pairs=len(traces),
        model_test_target_hash_count=prior['target_hash_counts_by_role']['test'],
        definition='persistence(target,h)=raw_gas(target-h); last input row of the neural window',
        common_scale='source concentration units; neural predictions inverse-transformed; raw persistence needs no scaling',
        common_aggregation='mean of two record-specific scores within seed, then mean and sample SD(ddof=1) across ten seeds; deterministic persistence has no seed variation',
        gap_crossings=0,future_or_target_gas_reads=0,mask='none in executed past-only input',
        maximum_stored_baseline_metric_difference=float(delta),
        H2_10s_and_CO2_all_leads_persistence_lower_equal_record_RMSE=True,
        individual_record_rankings_may_differ=True,statistical_superiority_claim=False,
        availability_limit='Both methods assume the gas value logged at origin is available then; analyzer transport/acquisition latency remains unconfirmed',
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            B/'training/data_contract/common_target_index.csv',B/'training/data_contract/observations_cache.npz',
            B/'persistence_baseline.csv',B/'metrics_per_record_seed_gas.csv',B/'evidence/posttraining_validation.json']})
    (OUT/'persistence_final_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
