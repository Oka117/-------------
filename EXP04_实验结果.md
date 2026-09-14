# EXP-04 实验结果

- 状态：已完成；收益稳定性证据不足，保留实验记录，暂不替换基线。
- 日期：2026-09-14 至 2026-09-15（Australia/Sydney）。
- 固定对照：`runs/baseline_v1/`；本轮无权重重跑：`runs/exp04_baseline_control/`。
- 父实验：无。从 baselinev1 独立出发，没有叠加 EXP01/02/03。
- 代码基点：`951003e`，源码快照、校验和、差异与运行日志位于 `runs/exp04_protocol/`。
- 唯一因素：训练样本权重。默认 `none`；A 的异常权重为 c；B 的多点事件权重为 `c × event_weight / 4`；正常权重为 1。单点事件 A/B 均设为 c，保证每个事件平均异常权重相同。验证评分中的单点事件权重仍为 7。
- 数据及模型：六设备、原始 402/536 维特征，300 轮、4 线程、seed=42，其他 LightGBM 参数不变。权重只用当前训练前缀标签计算，全量重训沿用同一规则。
- 时间划分：4 月、7 月、9 月边界，72 点隔离，并保护完整事件；实际设备边界与 baseline 完全相同。
- 选参：仅用有效监督 fold0 概率校准阈值，再用 fold1 总分分别选择 A/B 的 c；平分优先 c=2。先保存 `selection.json`，之后分析 fold2。A/B 均选 c=2。
- fold2 阈值：按原 baseline 算法使用 fold0+fold1 校准，设备有效校准事件不足 2 个时回退 pooled 阈值；每个新模型重新校准。
- 评估边界：fold1 是历史开发选参块，fold2 之前已用于开发诊断，均不是全新独立测试。总分采用本地复现公式，不是官方脚本。
- A 榜：未提交。没有根据测试预测比例调标签。

## 1. 历史向前筛选与后续诊断

| 配置 | fold1 总分 | 相对基线 | fold2 总分 | 相对基线 | fold2 Accuracy | fold2 加权召回 |
|---|---:|---:|---:|---:|---:|---:|
| baselinev1 | 70.486116 | +0.000000 | 67.052841 | +0.000000 | 87.4251% | 46.6806% |
| EXP-04A c=2 | 70.455775 | -0.030341 | 69.025323 | +1.972482 | 89.7535% | 48.2971% |
| EXP-04A c=4 | 69.785422 | -0.700695 | 68.399455 | +1.346614 | 89.4608% | 47.3381% |
| EXP-04B c=2 | 70.858824 | +0.372707 | 67.697519 | +0.644679 | 86.9553% | 48.4397% |
| EXP-04B c=4 | 70.817754 | +0.331638 | 65.710039 | -1.342801 | 87.6225% | 43.7975% |

- A c=2 的 fold2 最高，但 fold1 低于基线，不能只按 fold2 选择它为胜者。
- B c=2 在 seed=42 的两个评估块均提升；B c=4 在 fold2 退化，说明加大权重不一定更好。
- 同 c 比较：B 相对 A 在 fold1 更好，在 fold2 更差，没有证据认为前端加权普遍优于普通类别加权。

## 2. 逐设备、事件与误报

以下为 fold2；完整六设备、所有事件、全部误报段见 `runs/exp04_analysis/<配置>/` 下 CSV。无异常块的召回为空；整段未检出延迟为空并标记 missed_event=true。3/6 小时分别为前 9/18 个采样点，短事件按实际长度。误报持续时间按覆盖采样时长（点数 × 20 分钟），长误报段定义为至少 18 点。

| 设备 | 配置 | FP | FN | 加权召回 | ≥6小时误报段点数 |
|---|---|---:|---:|---:|---:|
| P202A | baselinev1 | 0 | 316 | 0.0000% | 0 |
| P202A | EXP-04A c=2 | 0 | 316 | 0.0000% | 0 |
| P202A | EXP-04B c=2 | 0 | 316 | 0.0000% | 0 |
| P310A | baselinev1 | 27 | 63 | 54.3410% | 0 |
| P310A | EXP-04A c=2 | 25 | 51 | 69.7341% | 0 |
| P310A | EXP-04B c=2 | 30 | 57 | 60.0835% | 0 |
| P310B | baselinev1 | 297 | 0 | 100.0000% | 290 |
| P310B | EXP-04A c=2 | 304 | 0 | 100.0000% | 291 |
| P310B | EXP-04B c=2 | 290 | 0 | 100.0000% | 290 |
| P601B | baselinev1 | 1090 | 54 | 30.4980% | 631 |
| P601B | EXP-04A c=2 | 747 | 62 | 19.0485% | 301 |
| P601B | EXP-04B c=2 | 1177 | 46 | 38.8652% | 700 |

| 设备 | 配置 | 首次报警延迟（小时） | 前3小时召回 | 前6小时召回 |
|---|---|---:|---:|---:|
| P310A | baselinev1 | 4.0000 | 0.0000% | 22.2222% |
| P310A | EXP-04A c=2 | 0.0000 | 33.3333% | 66.6667% |
| P310A | EXP-04B c=2 | 2.6667 | 11.1111% | 38.8889% |
| P601B | baselinev1 | 5.0000 | 0.0000% | 16.6667% |
| P601B | EXP-04A c=2 | 5.0000 | 0.0000% | 11.1111% |
| P601B | EXP-04B c=2 | 4.3333 | 0.0000% | 22.2222% |

- B c=2 的 fold1 加权召回收益来自 P310B 两段异常（合计 340 点），漏报从 21 点降至 0；其他有事件设备的加权命中不变。整体 FP 增加 37 点，收益并非来自 P601B 的 1322 点长异常，该事件原本已经全召回。
- B c=2 的 fold2 召回改善来自 P310A 和 P601B 各一段事件；P310A 提前 1小时20分，P601B 提前40分。但 P601B FP 增加 87，长误报段点数增加69，抵消部分收益。
- A c=2 的 fold2 改善主要体现为 P310A 召回提高、P601B 误报减少；P601B 加权召回反而从30.50%降至19.05%，不符合该设备起始漏检改善的目标。
- P202A 的316点事件始终整段漏报；P106A/P412B 的 fold2 没有异常，不能推断异常检出能力。fold1 的 P310A/P412B 为单类训练模型且整段漏报，证据不足。

## 3. 资源与复核

| 配置 | 训练流程秒数 | 训练峰值MiB | 推理秒数 | 推理峰值MiB |
|---|---:|---:|---:|---:|
| baselinev1 | 77.18 | 643.45 | 2.55 | 255.36 |
| EXP-04A c=2 | 73.65 | 723.02 | 2.81 | 270.12 |
| EXP-04A c=4 | 73.76 | 633.47 | 2.32 | 326.02 |
| EXP-04B c=2 | 72.87 | 733.09 | 2.31 | 305.64 |
| EXP-04B c=4 | 72.84 | 828.05 | 2.36 | 328.52 |

训练流程含读取数据、历史折训练/预测、校准及全量重训，不是纯模型训练耗时；baseline 使用本轮重跑资源。推理在独立进程测量，包含读取及写出。数值为本机实测，受系统负载影响。

- 默认无权重重跑的18份验证 CSV 与 baseline_v1 数值完全一致，基线所有文件哈希未变。
- 四个配置的数据哈希、特征顺序、时间戳、标签、事件评分权重、时间切分及固定参数均与基线一致。
- 权重审计覆盖全部历史训练折和全量训练；相同 c 的 A/B 总权重相同，见 `training_weight_audit.json`。
- 四个配置及基线已完成六设备推理，各38006行；校验了字段、逐行时间戳、设备名、有限概率与阈值标签一致性。未生成候选提交ZIP。
- 12项单元测试通过，包含权重总量/单点事件、训练前缀、非法权重、LightGBM实际接收权重及原基线测试；`git diff --check`通过。

## 4. 执行与产物

```bash
.venv/bin/python train.py --output runs/exp04_baseline_control --training-weight none --positive-weight 2 --threads 4 --rounds 300
# 实际分别执行四组：
.venv/bin/python train.py --output runs/exp04a_class_weight/c2 --training-weight class --positive-weight 2 --threads 4 --rounds 300
.venv/bin/python train.py --output runs/exp04a_class_weight/c4 --training-weight class --positive-weight 4 --threads 4 --rounds 300
.venv/bin/python train.py --output runs/exp04b_event_weight/c2 --training-weight event --positive-weight 2 --threads 4 --rounds 300
.venv/bin/python train.py --output runs/exp04b_event_weight/c4 --training-weight event --positive-weight 4 --threads 4 --rounds 300
.venv/bin/python experiments/exp04.py select
.venv/bin/python experiments/exp04.py diagnose
.venv/bin/python experiments/predict_exp04.py
```

各运行目录保存 config、manifest、模型、特征重要性、数据校验和、验证概率及预测。
分析入口 `experiments/exp04.py`；完整训练入口 `experiments/run_exp04.py`；推理核验入口 `experiments/predict_exp04.py`。新工作目录中可复现，非空目录不覆盖。
诊断文件包括 `forward_fold1_devices/events/false_alarm_segments.csv` 与 `diagnostic_fold2_devices/events/false_alarm_segments.csv`；整体指标保存为同名前缀 JSON。

## 5. 补充种子复核与最终结论

按预先冻结的 B c=2 补跑 seed=43、44，同种子无权重基线作对照；其他配置不变，不重新选择 c。

| seed | fold1 基线 | fold1 B c=2 | 差值 | fold2 基线 | fold2 B c=2 | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 70.486116 | 70.858824 | +0.372707 | 67.052841 | 67.697519 | +0.644679 |
| 43 | 70.474266 | 70.718815 | +0.244549 | 66.058257 | 64.858116 | -1.200141 |
| 44 | 70.668104 | 70.528402 | -0.139702 | 65.720583 | 67.280820 | +1.560237 |

三个种子的平均差值：fold1 +0.159185，fold2 +0.334925。但两个时间块都出现负向种子，均值不能证明稳定收益。

**结论：EXP04 已完成，尚无足够证据把训练加权设为新默认。** A c=2 在 seed=42 的 fold2 改善较大，但历史向前筛选没有超越基线，且牺牲了 P601B 召回；B c=2 有局部提前报警收益，但增加 P601B 误报、对随机种子敏感。保留默认无权重基线；EXP04 不直接进入 EXP08 的“已验证有效改动”名单。

后续若继续研究，应针对 P601B 的概率重叠与持续误报另设独立实验，并保留 P202A 未见故障的整段漏报问题。不得把本地诊断收益写成线上提分。

补充运行：对 seed=43/44 分别执行以下命令（MODE 为 none/event，SEED 为43/44）：

```bash
.venv/bin/python train.py --output runs/exp04_confirmation/MODE_seedSEED --training-weight MODE --positive-weight 2 --threads 4 --rounds 300 --seed SEED
.venv/bin/python experiments/confirm_exp04.py
# 新工作目录复现主实验及补充种子：
.venv/bin/python experiments/run_exp04.py --confirm
```

补充种子全量模型和验证概率在 `runs/exp04_confirmation/`；逐设备、事件与误报段在 `runs/exp04_analysis/none_seed43/` 等目录；总表为 `confirmation_results.json`。补充种子仅作稳定性验证，没有生成测试提交包。

| 补充运行 | 训练流程秒数 | 峰值MiB |
|---|---:|---:|
| none seed=43 | 73.91 | 743.50 |
| event seed=43 | 71.93 | 917.17 |
| none seed=44 | 71.97 | 691.84 |
| event seed=44 | 77.13 | 622.92 |
