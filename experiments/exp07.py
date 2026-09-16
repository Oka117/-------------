"""EXP07: replay fixed baseline probabilities; never train a model."""
import argparse
import copy
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from baseline import sha256, write_json
from metrics import aggregate, choose_threshold, score, event_weights
from postprocessing import postprocess_probabilities, smooth_probabilities


def segments(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def details(b, pred):
    y = b['y']; result = score(y, pred)
    result.update(fp=int(((y == 0) & (pred == 1)).sum()), fn=int(((y == 1) & (pred == 0)).sum()))
    events = []
    for a, z in segments(y):
        hits = np.flatnonzero(pred[a:z]); w = event_weights(y[a:z])
        events.append(dict(start=str(b['timestamp'][a]), end=str(b['timestamp'][z-1]), points=int(z-a),
            missed_event=not bool(len(hits)), delay_minutes=int(hits[0]*20) if len(hits) else None,
            recall_3h=float(pred[a:min(z,a+9)].mean()), recall_6h=float(pred[a:min(z,a+18)].mean()),
            weighted_recall=float(w @ pred[a:z] / w.sum())))
    runs = [dict(start=str(b['timestamp'][a]), end=str(b['timestamp'][z-1]), points=int(z-a), duration_hours=float((z-a)/3))
            for a,z in segments((y == 0) & (pred == 1))]
    result.update(events=events, false_positive_runs=runs, false_positive_run_count=len(runs),
                  long_false_positive_points=sum(r['points'] for r in runs if r['points'] >= 18))
    return result


def calibrate(blocks, variant):
    """Original >=2 eligible-event device rule; else pooled eligible calibration."""
    eligible = {d:[b for b in bs if b['eligible']] for d,bs in blocks.items()}
    pooled = [b for bs in eligible.values() for b in bs]
    def fit(bs):
        if variant['mode'] == 'smoothing':
            bs = [dict(b, p=smooth_probabilities(b['p'],variant['window'])[0]) for b in bs]
        base = choose_threshold(bs)
        if variant['mode'] != 'hysteresis':
            return base, dict(variant)
        # Fixed compact grid anchored to the original exact-sweep threshold.
        # No alarms is an additional candidate; ties prefer higher high threshold.
        highs = sorted(set([float(np.nextafter(1.,2.)), *[min(float(np.nextafter(1.,2.)),base*m) for m in (.5,1.,2.)]]), reverse=True)
        best = None
        for high in highs:
            conf = dict(mode='hysteresis',low_threshold=high*variant['ratio'])
            value = aggregate([score(b['y'],postprocess_probabilities(b['p'],high,conf)[0]) for b in bs])['score']
            if value is not None and (best is None or value > best[0]):
                best = value, high, conf
        return (best[1],best[2]) if best else (base,dict(mode='hysteresis',low_threshold=base*variant['ratio']))
    shared = fit(pooled)
    result = {}
    for d,bs in eligible.items():
        events = sum(len(segments(b['y'])) for b in bs)
        enough = events >= 2 and any((b['y']==0).any() for b in bs)
        threshold, conf = fit(bs) if enough else shared
        result[d] = dict(threshold=threshold, postprocessing=conf, threshold_source='device_calibration' if enough else 'pooled_calibration',calibration_events=events)
    return result


def evaluate(all_blocks, fold, configs, dest):
    dest.mkdir(parents=True)
    per_device = {}
    for d,bs in all_blocks.items():
        b=bs[fold]; meta=configs[d]
        pred,_=postprocess_probabilities(b['p'],meta['threshold'],meta['postprocessing'])
        chunks=[]; state=None
        for start in range(0,len(b['p']),137):
            pp,state=postprocess_probabilities(b['p'][start:start+137],meta['threshold'],meta['postprocessing'],state)
            chunks.extend(pp)
        np.testing.assert_array_equal(pred,chunks)
        per_device[d]=dict(details(b,pred), **meta)
        pd.DataFrame(dict(timestamp=b['timestamp'],label=b['y'],probability=b['p'],prediction=pred,event_weight=event_weights(b['y']))).to_csv(dest/f'{d}_fold{fold}.csv',index=False)
    result=dict(aggregate=aggregate(list(per_device.values())),devices=per_device)
    write_json(dest/'metrics.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',choices=['exp07'],required=True)
    ap.add_argument('--data-dir','--data_dir',default='data')
    ap.add_argument('--baseline-dir',default='runs/baseline_v1')
    ap.add_argument('--output',default='runs/exp07')
    args=ap.parse_args(); out=Path(args.output); baseline=Path(args.baseline_dir)
    if out.exists() and any(out.iterdir()): ap.error('Output directory is not empty')
    out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    manifest=json.loads((baseline/'manifest.json').read_text()); metrics=json.loads((baseline/'metrics.json').read_text())
    hashes={str(p):sha256(p) for p in baseline.rglob('*') if p.is_file()}
    variants={'baseline':dict(mode='none'), **{f'a_ratio_{r:g}':dict(mode='hysteresis',ratio=r) for r in (.25,.5,.75)},
              'b_window_2':dict(mode='smoothing',window=2),'b_window_3':dict(mode='smoothing',window=3)}
    protocol=dict(config=vars(args),variants=variants,high_multipliers=[.5,1,2],
        calibration='fold0 for fold1; fold0+fold1 for fold2/final. Original device eligibility rule.',
        initialization='Each fold and test stream starts alarm off/history empty; never transfer state between models. Continuous 20min input required; reset after gap.',
        selection='Select best A ratio and B window using fold1 only. Retain baseline unless both fold scores improve, total long FP points do not increase, and no matched event loses 3h/6h recall or worsens delay/miss status in either fold.',
        evidence='fold1 is historical development selection; fold2 previously viewed development diagnostic. Neither is an unseen test.',
        p412b='Only one labeled fault event; independent fault generalization cannot be established.',
        test='Cold start at test boundary deliberately matches validation initialization; no training-label state warmup.',
        training_seconds=0, baseline_hashes=hashes,
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(out/'protocol.json',protocol)
    (out/'source').mkdir()
    sources=['train.py','predict.py','baseline.py','metrics.py','postprocessing.py','experiments/exp07.py','package_submission.py','tests/test_postprocessing.py','baselinev1_独立实验计划.md','requirements.txt']
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (out/'changes.patch').write_text(subprocess.check_output(['git','diff'],text=True))
    data_hashes={}; all_blocks={}
    for d in manifest['devices']:
        for suffix,expected in manifest['models'][d]['source_sha256'].items():
            p=Path(args.data_dir)/d/f'{d}_{suffix}.csv';actual=sha256(p)
            if actual != expected: raise ValueError(f'Changed source {p}')
            data_hashes[str(p)]=actual
        test_path=Path(args.data_dir)/d/f'{d}_test.csv'; data_hashes[str(test_path)]=sha256(test_path)
        bs=[]
        for f in range(3):
            df=pd.read_csv(baseline/'validation'/f'{d}_fold{f}.csv',float_precision='round_trip')
            row=next(x for x in metrics['folds'] if x['device']==d and x['fold']==f)
            bs.append(dict(y=df.label.to_numpy(),p=df.probability.to_numpy(),timestamp=df.timestamp.to_numpy(),eligible=row['model']=='lightgbm'))
        all_blocks[d]=bs
    write_json(out/'data_hashes.json',data_hashes)
    results={}; final_configs={}
    # Finish and freeze all fold1 selections before opening fold2 results.
    for name,variant in variants.items():
        t=time.perf_counter(); conf=calibrate({d:bs[:1] for d,bs in all_blocks.items()},variant)
        results[name]={'fold1':evaluate(all_blocks,1,conf,out/name/'fold1'),'calibration_and_fold1_seconds':time.perf_counter()-t}
        print(name,'fold1',results[name]['fold1']['aggregate'],flush=True)
    selected={group:max([n for n in variants if n.startswith(group+'_')],key=lambda n:results[n]['fold1']['aggregate']['score']) for group in ('a','b')}
    write_json(out/'fold1_selection.json',selected)
    for name,variant in variants.items():
        conf=calibrate({d:bs[:2] for d,bs in all_blocks.items()},variant);final_configs[name]=conf
        results[name]['fold2']=evaluate(all_blocks,2,conf,out/name/'fold2')
        print(name,'fold2',results[name]['fold2']['aggregate'],flush=True)
    assert abs(results['baseline']['fold2']['aggregate']['score']-metrics['holdout_aggregate']['score']) < 1e-10
    decisions={}
    for group,name in selected.items():
        reasons=[]
        for fold in ('fold1','fold2'):
            candidate=results[name][fold]; control=results['baseline'][fold]
            if candidate['aggregate']['score'] <= control['aggregate']['score']: reasons.append(f'{fold}: score did not improve')
            if sum(x['long_false_positive_points'] for x in candidate['devices'].values()) > sum(x['long_false_positive_points'] for x in control['devices'].values()): reasons.append(f'{fold}: more long false-positive points')
            for d,c in candidate['devices'].items():
                for e,b in zip(c['events'],control['devices'][d]['events']):
                    if e['recall_3h']<b['recall_3h'] or e['recall_6h']<b['recall_6h'] or (b['delay_minutes'] is not None and (e['delay_minutes'] is None or e['delay_minutes']>b['delay_minutes'])):
                        reasons.append(f'{fold}/{d}/{e["start"]}: early response regression')
        decisions[group]=dict(candidate=name,accepted=not reasons,reasons=reasons)
    write_json(out/'selection.json',dict(decisions=decisions,baseline_preserved=True,official_submission=False))
    # Independent manifests copy the exact baseline models; no fit or threshold tuning on test.
    verification=[]
    for name in variants:
        run=out/name; model_manifest=copy.deepcopy(manifest)
        (run/'models').mkdir()
        for d,meta in model_manifest['models'].items():
            meta.update(final_configs[name][d])
            if meta['path']:
                shutil.copy2(baseline/meta['path'],run/meta['path'])
                assert sha256(baseline/meta['path'])==sha256(run/meta['path'])
        model_manifest['experiment']=dict(name=name,initialization='cold',baseline=str(baseline.resolve()))
        write_json(run/'manifest.json',model_manifest)
        t=time.perf_counter()
        command=[sys.executable,'predict.py','--data-dir',args.data_dir,'--model-dir',str(run),'--output',str(run/'predictions'),'--zip']
        log=subprocess.check_output(command,text=True);(run/'inference.log').write_text(log)
        results[name]['inference_seconds']=time.perf_counter()-t
        for d,meta in model_manifest['models'].items():
            pp=pd.read_csv(run/'predictions'/f'{d}_probabilities.csv',float_precision='round_trip')
            original=pd.read_csv(baseline/'predictions'/f'{d}_probabilities.csv',float_precision='round_trip')
            np.testing.assert_allclose(pp.probability,original.probability,rtol=0,atol=1e-14)
            test=pd.read_csv(Path(args.data_dir)/d/f'{d}_test.csv',usecols=['timestamp'])
            pred=pd.read_csv(run/'predictions'/f'{d}_predict.csv')
            assert list(pred.columns)==['pump_id','timestamp','label'] and pred.timestamp.equals(test.timestamp) and pred.pump_id.eq(d).all()
            expected,_=postprocess_probabilities(pp.probability,meta['threshold'],meta['postprocessing'])
            parts=[];state=None
            for i in range(0,len(pp),137):
                chunk,state=postprocess_probabilities(pp.probability.iloc[i:i+137],meta['threshold'],meta['postprocessing'],state);parts.extend(chunk)
            np.testing.assert_array_equal(expected,parts);np.testing.assert_array_equal(expected,pred.label)
            verification.append(dict(candidate=name,device=d,rows=len(pred),probabilities_match=True,chunk_parity=True,timestamps_match=True))
    assert hashes=={str(p):sha256(p) for p in baseline.rglob('*') if p.is_file()}
    write_json(out/'verification.json',dict(baseline_unchanged=True,checks=verification,validation_chunk_checks=len(variants)*12))
    elapsed=time.perf_counter()-started
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    resources=dict(elapsed_seconds=elapsed,training_seconds=0,peak_rss_mib=rss/(1024**2 if platform.system()=='Darwin' else 1024),child_peak_rss_mib=child_rss/(1024**2 if platform.system()=='Darwin' else 1024),feature_counts={d:len(m['features']) for d,m in manifest['models'].items()})
    write_json(out/'resources.json',resources);write_json(out/'results.json',results)
    lines=['# EXP07 报警后处理实验结果','', '- 状态：已运行；各候选结论见下。原始 baselinev1 完整保留。','- 父实验：无；A 与 B 均独立从 baselinev1 出发，不叠加其他实验。','- 模型、原始概率、数据与划分固定；训练耗时为 0。原始数据 SHA256 已核对。',
        '- fold0 校准 → fold1 选择 A 的比例与 B 的窗口；fold0+fold1 校准 → fold2 开发诊断。fold2 已被查看，不能视为独立测试。',
        '- A 的高阈值候选为原算法阈值的 0.5/1/2 倍及禁止触发阈值，低阈值为高阈值的 0.25/0.5/0.75 倍；B 单独比较 2/3 点均值并重新精确扫描阈值。',
        '- 沿用设备至少 2 段有效校准异常的资格规则，否则使用共享校准阈值。每个块、模型和测试流均冷启动，不跨模型携带报警状态；平滑开头使用已有点。',
        '- 每个候选都生成独立六设备模型副本、manifest、验证与测试预测；未经官方提交。','', '| 配置 | fold1 分数 | 相对基线 | fold2 分数 | 相对基线 | fold2 Accuracy | fold2 加权召回 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        a=r['fold1']['aggregate'];b=r['fold2']['aggregate']
        lines.append(f"| {name} | {a['score']:.6f} | {a['score']-results['baseline']['fold1']['aggregate']['score']:+.6f} | {b['score']:.6f} | {b['score']-results['baseline']['fold2']['aggregate']['score']:+.6f} | {b['accuracy']:.4%} | {b['weighted_recall']:.4%} |")
    for group,decision in decisions.items():
        lines += ['',f"## EXP07{group.upper()}：{decision['candidate']}",f"结论：{'通过预设局部门槛，仍需独立数据验证' if decision['accepted'] else '未通过预设稳定性门槛，不替换基线'}。",*['- '+r for r in decision['reasons']], '', '| 折/设备 | FP 基线→候选 | FN 基线→候选 | 长误报点 基线→候选 |','|---|---:|---:|---:|']
        for fold in ('fold1','fold2'):
            for d,b in results['baseline'][fold]['devices'].items():
                c=results[decision['candidate']][fold]['devices'][d]
                lines.append(f"| {fold}/{d} | {b['fp']}→{c['fp']} | {b['fn']}→{c['fn']} | {b['long_false_positive_points']}→{c['long_false_positive_points']} |")
    lines += ['', '## 验证与证据边界', '- 通过所有候选六设备测试输出时间戳/行数/二元标签检查、137 点分块一致性、历史验证分块一致性、基线模型与产物哈希未变检查。',
        '- 每个候选的 fold1/metrics.json、fold2/metrics.json 保存逐设备指标、每个事件首次命中延迟与前 3/6 小时召回、全部误报段起止及长度。缺失事件延迟为 null，无异常块召回为 null。',
        '- 长误报段定义为至少 18 点（6 小时）。冷启动可能丢失块前报警状态，是固定的部署约定。全量模型相对历史模型的概率尺度迁移仍无保证。',
        '- P412B 只有一个故障事件，无法据此证明新故障识别能力。所有分数均为本地公式复现，不代表线上收益。',
        f"- 本次总耗时 {elapsed:.2f} 秒，主进程峰值 RSS {resources['peak_rss_mib']:.1f} MiB；子进程峰值 {resources['child_peak_rss_mib']:.1f} MiB（两者非同步总和）。详细推理时间见 results.json，特征数量见 resources.json。",'', '## 复现', '```bash',f'{sys.executable} train.py --experiment exp07 --data-dir {args.data_dir} --baseline-dir {args.baseline_dir} --output runs/exp07_repeat',f'{sys.executable} -m unittest discover -s tests -v','```','', '- protocol.json 在评估前冻结候选与门槛；fold1_selection.json 在 fold2 评估前保存选择。source/、changes.patch、data_hashes.json 保存版本与数据依据。']
    (out/'EXP07_实验结果.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(decisions,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
