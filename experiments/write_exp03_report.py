"""Produce a reproducible EXP-03 report from saved metrics, not inferred forecasts."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_exp03 import RUNS
from baseline import write_json


def main():
    out=ROOT/'runs/exp03_analysis'
    read=lambda n:json.loads((out/n).read_text())
    selection=read('selection.json');results=read('results.json');forward=read('forward_results.json')
    inference=read('inference_resources.json');verification=read('verification.json')
    names={'baseline_v1':'baselinev1','exp03a_history_deviation':'03A 历史均值偏离','exp03b_lag_difference':'03B 历史点差分','exp03c_history_std':'03C 历史标准差','exp03d_standardized_deviation':'03D 标准化历史偏离'}
    base=results['baseline_v1']['aggregate'];chosen=selection['chosen']
    lines=['# EXP-03 独立实验结果','','日期：2026-09-14。状态：四组 seed=42 实验及六设备推理已完成。','',
    '- 固定对照：`runs/baseline_v1/`；父实验：无。没有叠加 EXP-01/02，也没有组合四组特征。',
    '- 代码：`temporal_features.py` 负责训练折内筛选与共享特征构造；`train.py --feature-kind` 选择特征族；`predict.py` 使用 manifest 与训练末尾历史；新增诊断、验证和复现实验脚本。',
    '- 所有原始特征保留。每个训练折额外训练同配置原始特征 LightGBM，仅在该折训练数据上取 gain 前 30；gain 并列按输入列顺序。最终全量模型重新进行同样筛选。',
    '- 单类折采用固定前 30 列回退，分类模型仍为原基线常数预测。没有用全量模型重要性筛选历史折。',
    '- 分数使用原本地公式 50×Accuracy+50×异常加权召回；官方跨设备汇总方式仍以官方明确规则为准。',
    '- LightGBM 参数、训练权重、原逐设备阈值策略及设备阈值资格不变；每个新模型用自己的折外概率重新校准。',
    '- A/B/C 分别增加 90 列，D 增加 60 列；P106A/P601B 原始 536 列，其余 402 列。',
    '- 模型、每折特征配方、每折模型、验证概率、原始数据 SHA256、最终历史上下文和推理输出均已保存。','',
    '## 特征与因果边界','',
    '| 组别 | 派生特征 | 窗口/滞后（每点 20 分钟） |','|---|---|---|',
    '| 03A | 当前值减过去均值 | 3 / 18 / 72 |',
    '| 03B | 当前值减过去点值 | 1 / 3 / 18 |',
    '| 03C | 过去窗口标准差 | 3 / 18 / 72 |',
    '| 03D | 当前值减过去均值，再除以过去标准差 | 18 / 72 |','',
    '历史范围等价于 `shift(1).rolling(window, min_periods=window)`；标准差 ddof=0。C/D 对每个窗口独立计算方差，D 的均值也逐窗口计算，避免增量滚动算法残留舍入误差。开头历史不足保留派生 NaN，原始 NaN/Inf 拒绝输入。D 的分母下界是对应训练折原始列总体标准差×0.001与 1e-6 中的较大值，没有用验证数据估计。',
    '验证可读取隔离区间已观测特征，但隔离区间标签不参与拟合。推理保存并使用最后 72 个训练原始点；若测试起点与训练末尾不是连续 20 分钟，则重置历史、保留初始派生 NaN，不跨缺口制造连续历史。测试文件内部非 20 分钟连续记录按原读取器拒绝。','',
    '## 验证协议与历史选择','',
    '首次 C/D 使用增量滚动标准差，实际数据回放发现截断历史会改变少量预测，因此旧运行保存在 `runs/exp03_superseded/`，不纳入下表；本报告 C/D 均为稳定窗口计算后的重跑结果。A/B 特征计算未改动。新增长恒定段的精确分块一致性回归测试。',
    '沿用原 4 月、7 月、9 月开始的事件完整时间块及至少 72 点训练隔离。fold0/1 为历史校准；fold2 已曾参与问题定位，本文称为开发诊断，不能视为新独立测试。',
    '预声明的特征族选择只使用 fold0 生成阈值、fold1 计算整体分数；相同分数优先基线。fold2 不用于选家族、窗口或种子。最终 fold2 的阈值按原基线方法使用 fold0+1 校准。原训练脚本先生成全部固定模型预测，但下面的自动选择函数只读取 fold0/1 预测来做决定。','',
    '| 方案 | fold0 校准→fold1 总分 | 相对同期基线 |','|---|---:|---:|']
    for n,r in forward.items():lines.append(f"| {names[n]} | {r['aggregate']['score']:.6f} | {r['aggregate']['score']-forward['baseline_v1']['aggregate']['score']:+.6f} |")
    lines += ['',f"历史选择结果：**{names[chosen]}**。选择依据与规则见 `selection.json`。",'',
    '## fold2 开发诊断','',
    '| 方案 | 总分 | 相对基线 | Accuracy | 异常加权召回 |','|---|---:|---:|---:|---:|']
    for n,r in results.items():
        a=r['aggregate'];lines.append(f"| {names[n]} | {a['score']:.6f} | {a['score']-base['score']:+.6f} | {a['accuracy']*100:.4f}% | {a['weighted_recall']*100:.4f}% |")
    lines += ['', '### 逐设备 FP / FN 与点级 AUC','',
    'AUC 仅反映该时间块点级排序，不是比赛评分，也不保证事件开始处识别良好。无异常设备 AUC/召回不可计算。','',
    '| 设备 | 方案 | FP | FN | AUC | 异常加权召回 |','|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'不可计算' if x is None else f'{x:.4f}'
    for d in ['P106A','P202A','P310A','P310B','P412B','P601B']:
        for n,r in results.items():
            s=next(v for v in r['devices'] if v['device']==d)
            lines.append(f"| {d} | {names[n]} | {s['fp']} | {s['fn']} | {fmt(s['auc'])} | {fmt(s['weighted_recall'])} |")
    lines += ['', '### 事件起始响应','',
    '| 设备 | 方案 | 事件点数 | 首次命中延迟（小时） | 前 3 小时召回 | 前 6 小时召回 |','|---|---|---:|---:|---:|---:|']
    for d in ['P202A','P310A','P310B','P601B']:
        for n,r in results.items():
            for e in r['events']:
                if e['device']==d:lines.append(f"| {d} | {names[n]} | {e['points']} | {'整段漏检' if e['missed_event'] else fmt(e['first_hit_delay_hours'])} | {fmt(e['first_3h_recall'])} | {fmt(e['first_6h_recall'])} |")
    lines+=['','短事件按实际长度评价。整段未检出延迟为 null；不是 0。起止时间及每段加权召回见各方案 `diagnostic_fold2_events.csv`。','',
    '### 相对总分的设备贡献','',
    '以下使用共同 fold2 分母分解变化，不相加设备自己的 score：每设备贡献=50×正确点数变化/全体点数+50×加权命中变化/全体异常权重。','',
    '| 设备 | 03A | 03B | 03C | 03D |','|---|---:|---:|---:|---:|']
    nsum=sum(v['rows'] for v in results['baseline_v1']['devices']);wsum=sum(v['weighted_total'] for v in results['baseline_v1']['devices'])
    for b in results['baseline_v1']['devices']:
        vals=[]
        for name in RUNS:
            v=next(v for v in results[name]['devices'] if v['device']==b['device'])
            vals.append(50*(v['correct']-b['correct'])/nsum+50*(v['weighted_hit']-b['weighted_hit'])/wsum)
        lines.append('| '+b['device']+' | '+' | '.join(f'{v:+.6f}' for v in vals)+' |')
    lines+=['','## 实测资源与验证','',
    '| 方案 | 训练全流程秒数 | 训练进程峰值 RSS MiB | 六设备测试推理秒数 |','|---|---:|---:|---:|']
    for name in RUNS:
        m=json.loads((ROOT/'runs'/name/'metrics.json').read_text())
        lines.append(f"| {names[name]} | {m['elapsed_seconds']:.2f} | {m['peak_rss_mib']:.2f} | {inference[name]['seconds']:.2f} |")
    lines+=['',
    '训练全流程含原始特征筛选模型、派生构造、各折训练与预测、最终全量训练及文件读写；推理计时含载入、构造和 CSV 写出。不能将本次资源与不同运行条件的旧基线作严格性能比较。各折模型推理时间另存 metrics.json。',
    '- 13 项自动测试通过，覆盖原评分/边界、手工特征数值、未来扰动、分块构造、训练测试接缝、缺口重置及标准化保护。',
    f"- 实际数据验证 {len(verification['checks'])} 个方案/设备组合通过；每个组合回放全部 3 折模型概率与标签，并核对实际全训练历史与保存的 72 点上下文构造、最终推理概率/标签、列顺序、测试时间戳/行数及训练折 epsilon。",
    '- 原 baselinev1 全部已有文件 SHA256 未变。','',
    '## 复现与输出','',
    '```bash','.venv/bin/python -m unittest discover -s tests -v','.venv/bin/python experiments/run_exp03.py','.venv/bin/python experiments/analyze_exp03.py','.venv/bin/python experiments/confirm_exp03.py','```','',
    '四个默认输出目录已存在，训练拒绝覆盖；重跑需直接调用 `train.py --feature-kind ... --output 新目录`。每个方案的推理调用形式：','',
    '```bash','.venv/bin/python predict.py --model-dir runs/exp03a_history_deviation --output runs/exp03a_history_deviation/predictions --threads 4','.venv/bin/python experiments/verify_exp03.py','.venv/bin/python experiments/write_exp03_report.py','```','',
    '其他三组同样替换目录名。`runs/exp03_logs/` 保存训练日志；`runs/exp03_protocol/` 保存预声明设置与基线校验值；`runs/exp03_analysis/` 保存选参、逐设备/事件/误报段、资源与验证记录。每个训练目录保存模型、每折特征选择、验证概率与 manifest。',
    '线上 A 榜：未提交。不能将本地变化加到旧 A 榜 81.9323 上。','']
    confirmation=read('seed_confirmation.json')
    lines += ['## 配对随机种子复核', '',
        '特征族只由 seed=42 的历史向前结果选定，然后固定该家族，分别与同种子的原始特征基线对照；不会按最后时间块重新挑选家族。', '',
        '| seed | 家族 | 基线历史分数 | 特征历史分数 | 历史差值 | 基线 fold2 | 特征 fold2 | fold2 差值 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    if chosen != 'baseline_v1':
        paired=[(42,forward['baseline_v1']['aggregate']['score'],forward[chosen]['aggregate']['score'],base['score'],results[chosen]['aggregate']['score'])]
        for seed in [43,44]:
            a=next(v for v in confirmation['results'] if v['seed']==seed and v['family']=='raw')
            b=next(v for v in confirmation['results'] if v['seed']==seed and v['family']==chosen)
            paired.append((seed,a['forward']['aggregate']['score'],b['forward']['aggregate']['score'],a['fold2']['aggregate']['score'],b['fold2']['aggregate']['score']))
        for seed,bf,cf,bh,ch in paired:
            lines.append(f'| {seed} | {names[chosen]} | {bf:.6f} | {cf:.6f} | {cf-bf:+.6f} | {bh:.6f} | {ch:.6f} | {ch-bh:+.6f} |')
        lines += ['', '确认种子每次并行运行一组基线与候选（各 4 线程），初筛及重跑资源占用不完全一致，耗时仅作本次运行记录。全部额外种子逐设备/事件/误报段及资源记录见 `seed_confirmation.json`；这些额外运行仅用于验证，未生成候选提交包。', '']
    else:
        lines += ['', '历史向前评估未选择任何新家族，因此未扩展种子。', '']
    lines += ['## 结论与下一步', '',
        '本次不替换原 baselinev1，也不自动组合特征族。历史选中的 03A 在 seed=42/43/44 的后续时间块均低于同种子基线；03D 虽然该块上涨，但历史评估较基线下降，且并未通过历史选参进入额外种子确认。不能按同一已见 fold2 重新选择 03D，再宣称它是独立验证的赢家。', '',
        '- 03A/03B：seed=42 历史分数略升，fold2 分别下降约 1.56/1.63 分；P601B 的更多命中伴随更多误报，排序 AUC 也未改善。',
        '- 03C：历史分数和 fold2 均下降。P601B 的点级 AUC 提升到约 0.576，但实际阈值产生大量误报，P310A 召回也退化；排序改善不自动等于比赛得分改善。',
        '- 03D：fold2 上涨约 0.584 分，主要来自 P310A 的加权召回改善与更早命中。P310A 首次报警由 4 小时降至 0，但前 3 小时召回仅 22.2%，不能将一次早期命中描述为持续稳定识别。P601B 未改善。',
        '- P202A：四组均未检出其 316 点异常；正常状态统计特征没有在本次设置下解决少异常监督问题。D 的点级 AUC 有上升，但回退阈值下仍全部漏检。',
        '- 当前前 30 原始 gain 特征、固定窗口及固定训练/阈值设置尚未提供稳定收益证据；不能推广为所有时序特征无效。若继续研究，应把新窗口、选择方法或阈值策略另列实验，保留原始独立对照。', '',
        '线上 A 榜未提交；没有生成新的正式提交包。默认 `--feature-kind raw` 保持不变。', '']
    (ROOT/'EXP03_实验结果.md').write_text('\n'.join(lines))
    (out/'实验结果.md').write_text('\n'.join(lines))
    print(ROOT/'EXP03_实验结果.md')


if __name__=='__main__':main()
