# 泵设备异常预警 baseline v1

分设备 LightGBM 二分类，输入为当前时刻的原始 `feature_*`。CPU 训练，默认 4 线程；不使用时间戳作为特征，不使用测试数据拟合模型或选择阈值。各设备动态读取 402/536 列，无需人工补列。此版本尚未加入滚动统计、异常检测模型或集成。

## 快速开始

在当前项目目录执行。本机 `.venv` 已安装依赖，可直接跳到训练命令。新环境使用 Python 3.14（本机验证版本 3.14.7）：

```bash
# macOS 的 LightGBM 需要 OpenMP；本机已安装
brew install libomp
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

先训练两台泵：

```bash
.venv/bin/python train.py --devices P601B P310B --output runs/two_pumps_v1 --threads 4
```

训练全部六台泵：

```bash
.venv/bin/python train.py --data-dir data --output runs/my_baseline_v1 --threads 4 --rounds 300
```

推理（model-dir 与上面训练 output 对应）：

```bash
.venv/bin/python predict.py --data-dir data --model-dir runs/my_baseline_v1 --output runs/my_baseline_v1/predictions
```

路径相对于命令执行目录；路径含空格时加引号。`--data_dir` 也可作为 `--data-dir` 的别名。每次实验使用新输出目录，脚本拒绝覆盖非空目录。无需激活环境，明确使用 `.venv/bin/python` 即可。

## 验证与阈值

默认三个向前验证块：

| 验证块 | 时间范围 | 用途 |
|---|---|---|
| fold0 | 2024-04-01 至 2024-07-01 前 | 校准阈值 |
| fold1 | 2024-07-01 至 2024-09-01 前 | 校准阈值 |
| fold2 | 2024-09-01 至训练数据结束 | 最终留出评估 |

每折只使用该块之前的数据训练，并排除边界前 72 点（24 小时）。若边界会切开异常，则向前移到该异常的开始，避免同一异常进入两侧；这是使用已知标签设计验证划分，不是推理特征。实际起止时间记录在 `metrics.json`。例如 P106A 的首个边界会提前到 3 月 28 日。所有原始文件保持不变。

默认固定 300 轮、小树（15 个叶子）、叶节点最少 100 点、L2 正则化。不在最终留出块上 early stopping，也不基于留出分数自动调整参数。通过 `--rounds` 修改轮数后应视为新实验；反复看留出分数做选择会使其变为开发集，不再是独立测试。

每台设备若前两块中有至少 2 段来自有效监督模型的校准异常，则按官方公式选设备阈值；否则使用所有有效校准块汇总得到的统一阈值。统一阈值跨设备迁移并无校准保证，报告会标注来源。若所有有效校准块都没有异常，则明确退回 0.5。

训练折只有一种标签时，输出同一类别的常量预测，并在报告注明。这类折仍参与验证报告，但不用于阈值校准，避免把“没有训练过的分类器”当作正常模型。P310A、P412B 等设备会出现这种情况。

先完成时间验证和阈值选择，再使用全部训练标签重训最终模型。最终模型不能用来给旧验证段补做“验证预测”。留出 CSV 来自对应历史训练折模型。

## 评分含义

规则来源：[官方赛题](https://comp.kunlungpt.cn/player/competition/problem/2082685461163094018?eventId=2082273411774775297&index=1)。该实现依据已读取的网页公式，不是官方提供的评分脚本。

`Score = 50 × Accuracy + 50 × 异常加权召回率`。

同设备连续异常段长度 L>1 时，权重从 7 线性降到 1，单点异常权重为 7。汇总时先在各设备/时间块内计算权重，再累加分子、分母，不跨设备拼接异常。

- 无异常块的 `score`、`weighted_recall` 为 null，正常误报率仍可用。整体汇总只要含异常，就可计算总分。
- 阈值扫描按全部不同概率精确计算候选分数，不固定 0.5。分数并列优先较高阈值。
- `blocks[0:2]` 已参与阈值选择，属于校准成绩；`blocks[2]` 与 `holdout_aggregate` 才是本次未参与调参的留出成绩。
- 各设备留出异常很少，P106A/P412B 甚至没有异常，不能用一次留出分数证明上线泛化。总分还需与逐设备召回和误报率一起看。
- 最终留出分数不是线上 A/B 榜得分。官网要求 B 榜超过 80 分才具备晋级资格，不代表本基线能够达到。

## 输出

```text
runs/my_baseline_v1/
  config.json                 启动配置和依赖版本
  manifest.json               模型、特征顺序、阈值、训练数据 SHA256
  metrics.json                每折时间边界、指标、耗时和峰值内存
  models/
    P601B.txt                 全训练集拟合的 LightGBM 模型
    P601B_importance.csv      gain 特征重要性
  validation/
    P601B_fold0.csv           标签、概率、预测和异常权重
    P601B_fold1.csv
    P601B_fold2.csv
  predictions/
    P601B_predict.csv         pump_id,timestamp,label 三列
    P601B_probabilities.csv   概率，用于分析，不属于提交文件
```

以上以 P601B 示意，其余设备各自生成对应文件。

## 比赛结果导出

按用户选择，默认导出 `pump_id,timestamp,label` 三列。pump_id 的值来自六个设备文件夹名称。该表头是当前采用的约定，尚未取得官方样例确认；若平台要求不同名称，可用 `--pump-column` 覆盖。

生成三列结果与结果 ZIP：

```bash
.venv/bin/python predict.py --model-dir runs/my_baseline_v1 --output runs/my_baseline_v1/submission --zip
```

`result.zip` 仅包含本次模型涉及设备的 `*_predict.csv`，不包含概率文件。正式提交必须训练全部六台泵，检查 ZIP 含六个文件。当前 baseline_v1 已训练六台泵，打包脚本可生成含源码、依赖说明、配置及模型的代码包：

```bash
.venv/bin/python package_submission.py
```

输出为 `runs/baseline_v1/submission/submission_code.zip`，压缩包保留模型相对路径；不包含原始数据或 .venv。预测默认模型路径相对于 predict.py 定位，不依赖当前工作目录。解压并安装 requirements.txt 后，执行：

```bash
python predict.py --data_dir /absolute/path/to/data --output /absolute/path/to/new_result --zip
```

从头训练及推理（同样使用新输出目录）：

```bash
python train.py --data_dir /absolute/path/to/data --output /absolute/path/to/new_run --threads 4 --rounds 300
python predict.py --data_dir /absolute/path/to/data --model-dir /absolute/path/to/new_run --output /absolute/path/to/new_result --zip
```

旧 `runs/baseline_v1/predictions/` 中的两列预览保留用于比较，正式三列文件位于 `runs/baseline_v1/submission/`。所有结果均由模型推理生成，没有上传平台。

推理严格保留测试时间戳和原始行顺序，检查特征名与顺序、有限数值及训练/测试时间重叠。当前数据无缺失，因此发现异常输入时会直接报错，不悄悄填充或删行。

## 测试与后续实验

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖官方权重示例、单点异常、全对/全错、常量预测、无异常块、跨设备汇总、阈值扫描和事件边界。

下一步优先对 P601B/P310B 的起始漏检与正常误报做分析，再加入少量因果滚动特征。不要根据测试集预测异常比例手工指定标签，也不要直接将绝对时间作为故障捷径。

## EXP08 准入审计

已核对 EXP01–EXP07 最终分支结果，并补做 EXP01 的 fold0→fold1 阈值验证。当前缺少两项通过独立验证的有效改动，组合训练及移除消融暂不满足计划前提，保留 baselinev1。详见 [EXP08 实验结果](EXP08_实验结果.md)。

复核入口：`.venv/bin/python experiments/exp08.py --output runs/exp08_new_audit`。需要原始数据、七个本地实验分支及已有 baselinev1/EXP01 运行资产；使用新的输出目录。
