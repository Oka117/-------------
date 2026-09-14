"""Finish EXP05 inference checks and write the evidence tables."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline import write_json, sha256

def main():
    out = ROOT/'runs/exp05_analysis'
    results = json.loads((out/'results.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    confirmation = json.loads((out/'confirmation.json').read_text())
    baseline = results['baseline']
    adopted=[]
    for n in sorted(set(selection['winners'].values())-{'baseline'}):
        stable=all(results[n][part]['aggregate']['score']>baseline[part]['aggregate']['score'] for part in ['forward','diagnostic'])
        stable=stable and all(results[n][part]['aggregate']['score']>=baseline[part]['aggregate']['score'] for part in ['july','august'])
        stable=stable and all(confirmation[f'{n}_seed{s}'][part]['aggregate']['score']>confirmation[f'baseline_seed{s}'][part]['aggregate']['score'] for s in [43,44] for part in ['forward','diagnostic'])
        if stable: adopted.append(n)
    write_json(out/'decision.json',dict(adopted=adopted,status='candidate_requires_review' if adopted else 'retain_baseline',
        rule='Predeclared positive forward and diagnostic gains across seeds 42/43/44; nonnegative July and August gains at seed42.'))
    verification = {}
    for name,r in results.items():
        folder = Path(r['folder'])
        command = [sys.executable,'-c',
            "import runpy,time,resource,json,platform; t=time.perf_counter(); runpy.run_path('predict.py',run_name='__main__'); print(json.dumps(dict(seconds=time.perf_counter()-t,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024))))",
            '--model-dir',str(folder),'--output',str(folder/'predictions')]
        completed = subprocess.run(command,cwd=ROOT,text=True,capture_output=True,check=True)
        (folder/'inference.log').write_text(completed.stdout)
        resources = json.loads(completed.stdout.splitlines()[-1])
        meta = json.loads((folder/'manifest.json').read_text())
        rows = {}
        for d,m in meta['models'].items():
            p = pd.read_csv(folder/'predictions'/f'{d}_predict.csv')
            prob = pd.read_csv(folder/'predictions'/f'{d}_probabilities.csv')
            test = pd.read_csv(ROOT/'data'/d/f'{d}_test.csv',usecols=['timestamp'])
            assert p.columns.tolist() == ['pump_id','timestamp','label']
            assert p.timestamp.equals(test.timestamp) and prob.timestamp.equals(test.timestamp)
            assert (p.pump_id == d).all() and p.label.isin([0,1]).all()
            np.testing.assert_array_equal(p.label,(prob.probability >= m['threshold']).astype(int))
            assert np.isfinite(prob.probability).all()
            for suffix,h in m['source_sha256'].items():
                assert sha256(ROOT/'data'/d/f'{d}_{suffix}.csv') == h
            rows[d] = len(p)
        verification[name] = dict(rows=rows,resources=resources,schema_timestamps_thresholds_data_hashes=True)
    write_json(out/'inference_verification.json',verification)
    lines = ['# EXP05 模型容量与训练轮数：实验结果','',
        '**结论：'+('存在通过预定筛选规则的候选：'+', '.join(adopted) if adopted else '本轮未采纳任何新配置，保留 baselinev1。')+'**','',
        '- 对照：`runs/baseline_v1/`；默认值为 min_data_in_leaf=100、num_leaves=15、300 轮。',
        '- 父实验：无。三个单因素实验均从 baselinev1 出发，原始特征、单位样本权重、原阈值选择策略不变。',
        '- 时间划分：4 月开始的 fold0、7 月开始的 fold1、9 月开始的 fold2；72 点隔离并保护事件边界。',
        '- 参数选择：仅以 fold0 有效监督概率校准阈值，按 fold1 整体分数在每个家族（含基线）中选择；并列优先基线。selection.json 在分析候选 fold2 指标前保存。基线重现一致性检查不用于参数选择。',
        '- fold2 阈值用 fold0+fold1 校准，算法及设备资格条件与原基线相同。fold2 已被历史实验使用，成绩属于开发诊断，不是独立测试或线上提分证明。',
        '- 7 月/8 月为 fold1 按事件安全边界划分的子块，使用同一个仅在 fold1 之前训练的模型和 fold0 阈值；不是新增独立故障或独立重训折。',
        '- 无 early stopping、无参数全网格、无设备定制参数。候选均 seed=42、4 线程。',
        '- 本地评分沿用现有复现公式，不是官方评分脚本；A 榜未提交。','',
        '## 单因素结果','',
        '| 配置 | fold1 早期评估 | 相对基线 | 7 月 | 8 月 | fold2 诊断 | 相对基线 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,r in results.items():
        f=r['forward']['aggregate']['score']; d=r['diagnostic']['aggregate']['score']
        lines.append(f"| {n} | {f:.6f} | {f-baseline['forward']['aggregate']['score']:+.6f} | {r['july']['aggregate']['score']:.6f} | {r['august']['aggregate']['score']:.6f} | {d:.6f} | {d-baseline['diagnostic']['aggregate']['score']:+.6f} |")
    lines += ['',f"早期筛选结果：{selection['winners']}。",'', '## fold2 整体指标与资源','',
        '| 配置 | Accuracy | 加权召回 | 训练秒数（含全量重训） | 验证推理秒数 | 全流程秒数 | 训练峰值 MiB | 测试推理流程秒数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in results.items():
        a=r['diagnostic']['aggregate']; s=r['resources']
        lines.append(f"| {n} | {a['accuracy']:.6%} | {a['weighted_recall']:.6%} | {s['fit_seconds']:.2f} | {s['validation_predict_seconds']:.2f} | {s['elapsed_seconds']:.2f} | {s['peak_rss_mib']:.1f} | {verification[n]['resources']['seconds']:.2f} |")
    lines += ['','训练资源为各训练子进程实测；全量拟合时间包含模型/重要性写出和数据校验值计算。测试推理流程含模块加载、读文件、模型加载、预测与 CSV 写出。测试推理峰值详见 inference_verification.json。所有方案的特征数保持各设备原有的 402/536 列。','',
        '## 随机种子确认','', '| 配置 | fold1 | 对配对基线差值 | fold2 | 对配对基线差值 |','|---|---:|---:|---:|---:|']
    for n,r in confirmation.items():
        seed=n.split('_seed')[-1]; b=confirmation['baseline_seed'+seed]
        f=r['forward']['aggregate']['score']; d=r['diagnostic']['aggregate']['score']
        lines.append(f"| {n} | {f:.6f} | {f-b['forward']['aggregate']['score']:+.6f} | {d:.6f} | {d-b['diagnostic']['aggregate']['score']:+.6f} |")
    if not confirmation:
        lines += ['','各家族早期筛选均未胜过基线，因此按预定规则不扩展随机种子。']
    for device in ['P202A','P106A','P310A','P310B','P412B','P601B']:
        lines += ['',f'## {device} 逐设备诊断','', '| 配置 | 分数 | FP | FN | 普通召回 | 加权召回 | 阈值 | 来源 |', '|---|---:|---:|---:|---:|---:|---:|---|']
        for n in results:
            df=pd.read_csv(out/n/'diagnostic_fold2_devices.csv'); r=df[df.device==device].iloc[0]
            def fmt(v):
                return 'NA' if pd.isna(v) else f'{v:.6f}'
            lines.append(f"| {n} | {fmt(r.score)} | {int(r.fp)} | {int(r.fn)} | {fmt(r.recall)} | {fmt(r.weighted_recall)} | {r.threshold:.9g} | {r.threshold_source} |")
    lines += ['','## 复现与产物','', '```bash','.venv/bin/python -m unittest discover -s tests -v', '.venv/bin/python experiments/exp05.py', '.venv/bin/python experiments/exp05_report.py','```','',
        '运行脚本拒绝覆盖已有实验目录；复现请在新的 checkout/output 位置运行。各配置的具体 train.py 参数在 experiments/exp05.py 的 CONFIGS 中；默认对照实际重训。',
        '- `runs/exp05_protocol/`：运行前实验协议、基线文件 SHA256、git HEAD/差异、执行源码快照、训练日志。',
        '- `runs/exp05_analysis/selection.json`：提前锁定的筛选结果。',
        '- `runs/exp05_analysis/results.json`：全部配置的整体、子块、误报段统计和资源。',
        '- `runs/exp05_analysis/<配置>/`：逐设备、逐事件（首次延迟、前 3/6 小时召回、整段加权召回）、逐误报段 CSV。未命中事件延迟留空并明确标记；无异常块的召回不填满分。',
        '- `runs/exp05_analysis/verification.json`：原始基线未改变、重训对照预测一致性。',
        '- `runs/exp05_analysis/inference_verification.json`：7 个主配置的六设备推理行数、时间戳、标签阈值和训练数据哈希检查。',
        '- `runs/exp05_baseline_control/`、`runs/exp05a_leaf_samples/`、`runs/exp05b_num_leaves/`、`runs/exp05c_rounds/`：配置、manifest、历史折/全量模型、重要性、验证概率、测试推理输出。',
        '- `runs/exp05_confirmation/`：早期胜出配置与配对基线的补充种子运行。','']
    recovery=ROOT/'runs/exp05_protocol/recovery.json'
    if recovery.exists():
        lines += ['本次实际先执行主脚本；7 个配置训练完成后，统计写出遇到 NumPy 整数 JSON 序列化错误。修复类型转换后执行 `.venv/bin/python experiments/exp05.py --resume`，核对训练源码/配置后复用完整运行。训练数据、模型与筛选规则未变；见 `runs/exp05_protocol/recovery.json`。','']
    (ROOT/'EXP05_实验结果.md').write_text('\n'.join(lines))
    print('Report and inference verification completed.')

if __name__ == '__main__':
    main()
