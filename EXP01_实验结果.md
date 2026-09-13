# EXP-01：共享分母阈值实验结果

日期：2026-09-13。状态：代码已实现，完成六设备训练与本地验证；线上未提交。

## 改动

- `metrics.py`：`choose_threshold()` 支持成对传入 `global_rows`、`global_weight`；省略时保留原目标。检查分母为有限正值并覆盖当前局部数据。
- `train.py`：新增 `--threshold-normalization device|global`，默认 `device` 保留 v1。全局分母只累计前两个时间块中有效监督模型的校准数据；设备资格和所有回退阈值保持原样。
- `metrics.json`、最终 `manifest.json` 保存模式、共享分母和数据范围。
- `tests/test_baseline.py`：补充整体目标与局部目标不同的例子、50 组随机暴力枚举比较、参数校验和旧行为兼容检查。

## 实际运行

在项目目录运行：

```bash
.venv/bin/python train.py --output runs/exp01_global_threshold --threshold-normalization global --threads 4 --rounds 300
.venv/bin/python predict.py --model-dir runs/exp01_global_threshold --output runs/exp01_global_threshold/predictions --zip
```

上述目录已生成；再次运行需指定新输出目录。

对照为 `runs/baseline_v1/`，使用相同数据、时间划分、seed=42、300 轮和 4 线程。共享分母 N=37,775，W=22,312；未纳入 fold2 和单类常量模型块。

## 结果

| 指标 | baselinev1 | EXP-01 | 变化 |
|---|---:|---:|---:|
| 有效监督校准块整体分数 | 78.0887 | 78.1906 | +0.1019 |
| 本地 fold2 整体分数 | 67.0528 | 67.9122 | +0.8594 |
| 本地 fold2 Accuracy | 87.4251% | 86.2200% | -1.2051 个百分点 |
| 本地 fold2 异常加权召回 | 46.6806% | 49.6044% | +2.9239 个百分点 |

只有 P601B 的阈值从 `0.009729547298222315` 降至 `0.007635118288682552`；其余五设备阈值未变。P601B 单设备 fold2 分数由 41.8830 变为 54.2503。

11 项自动测试通过。另核对全部 18 个验证 CSV，模型概率、时间戳和标签与 v1 完全一致；共享回退阈值及缺乏设备校准证据的三设备阈值保持一致。校准整体目标未下降。核对记录位于 `runs/exp01_global_threshold/verification.json`。

训练报告耗时约 73.1 秒，峰值 RSS 约 691 MiB。这些是本次运行资源记录，不能据此宣称阈值改动带来训练加速。

## 解释与限制

本次收益来自降低 P601B 阈值后增加异常召回，同时增加正常误报。符合本地整体目标的权衡，但仅一个后续时间块不足以证明稳定泛化。

现有 fold2 已在前期用于错误分析，因此这里属于开发诊断成绩。基线线上 A 榜 81.9323 来自用户反馈；EXP-01 尚无线上分数，不能将本地 +0.8594 加到该成绩上。

实现对齐当前本地 `aggregate()` 公式；官方跨设备汇总口径仍需以官方评分脚本或明确规则为准。
