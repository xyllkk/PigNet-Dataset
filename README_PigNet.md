
# PigNet: Multi-step Forecasting of Pig Body Weight from Multi-source Data with Time-aware Attention and Depthwise-Separable Smoothing
猪体重多步预测模型：基于多源数据的时间感知注意力与深度可分离卷积平滑

[简体中文 | English]

## Overview 概览
PigNet 是一个面向养猪生产的数据驱动体重预测模型。它以 14 天滑动窗口汇聚多源数据（采食行为、年龄、初始体重、猪舍环境温湿气体），直接预测未来 7 天的日粒度体重序列。模型采用 LSTM 主干，并在读出端加入时间感知注意力（TAM）以区分不同预测步长的证据来源，同时沿“地平线轴”引入深度可分离卷积平滑模块（DSCSM），在不增加推理成本的前提下抑制步间抖动并提升稳定性。

PigNet is a data-driven forecasting model for commercial pig production. Using a 14-day sliding window of multi-source inputs (feeding behavior, age, initial body weight, and barn environmental variables), it directly predicts a 7-day sequence of daily body weight. An LSTM backbone is coupled with a Time-aware Attention Module (TAM) for horizon-specific readout and a Depthwise-Separable Convolutional Smoothing Module (DSCSM) along the forecast-horizon axis for lightweight cross-horizon consistency.

## Highlights 亮点
- 低成本落地：仅依赖 RFID 采食记录、食槽称重负载单元与猪舍环境传感器等常见配置即可运行。
- 步长感知读出：为每个预测步长分配可学习查询，聚合其最相关的时序证据，缓解跨步长异质性。
- 轻量平滑约束：在“地平线轴”上采用深度可分离 1D 卷积进行局部耦合与平滑，降低步间抖动。
- 评测友好：提供按猪分组的交叉验证范式与宏平均指标，避免个体信息泄露。

## Dataset 数据集
- 粒度：以天为单位的多变量时间序列。
- 必需字段（列名建议）：
  - pig_id, date(YYYY-MM-DD), age(days), init_bw(kg), bw(kg, 可选作监督检查)
  - feed_intake(kg), feed_duration(min), feed_freq(times/day)
  - temp(°C), rh(%), nh3(ppm), co2(ppm)
- 目录建议：
```
data/
  raw/                         # 原始记录（可选）
  processed/                   # 预处理后的按猪日表
    P0001.csv
    P0002.csv
    ...
```
- 单个 CSV 示例（逗号分隔）：
```
pig_id,date,age,init_bw,bw,feed_intake,feed_duration,feed_freq,temp,rh,nh3,co2
P0001,2025-06-11,71,30.2,35.4,2.43,42.0,18,27.3,58.1,5.2,870
P0001,2025-06-12,72,30.2,36.0,2.61,40.5,17,27.7,60.3,5.0,865
...
```
说明：训练时只需特征列，bw 列可在验证阶段用于度量。年龄与初始体重为强信号，环境变量对长期步长和高温季更敏感。

## Installation 安装
- Python ≥ 3.9
- PyTorch ≥ 2.1（CPU 或 GPU 均可）
- 依赖包见 requirements.txt（示例）

```
git clone https://github.com/yourname/PigNet.git
cd PigNet
conda create -n pignet python=3.9 -y
conda activate pignet
pip install -r requirements.txt
```

示例 requirements.txt（可按需增删）：
```
torch
torchvision
torchaudio
numpy
pandas
scikit-learn
tqdm
pyyaml
matplotlib
```

## Data Preparation 数据预处理
将原始多次到访级记录聚合为日粒度（同一天同一只猪的 feed_intake、feed_duration、feed_freq 求和，bw 取日均；环境变量取日均）。
提供的示例脚本可完成：缺失与异常值处理 → 日表聚合 → 训练/验证/测试划分（按 pig_id 分组）。

```
python tools/prepare_data.py \
  --input data/raw \
  --output data/processed \
  --min-days 21
```

## Configuration 配置
示例 YAML：configs/pignet.yaml
```
seed: 2025
window: 14           # 观测窗口长度 W
horizon: 7           # 预测步长 H
features: [feed_intake, feed_duration, feed_freq, temp, rh, nh3, co2, age]
static: [init_bw]
model:
  type: PigNet
  hidden_size: 128
  tam:
    n_heads: 4
    dropout: 0.1
  dscsm:
    kernel_size: 3
    dilation: 1
    dropout: 0.0
  film_se:
    film: true
    se_reduction: 8
train:
  epochs: 100
  batch_size: 128
  lr: 1e-3
  weight_decay: 1e-4
  smooth_loss:
    tv: 0.05
    curv: 0.01
eval:
  metrics: [rmse, r2]
  group_by: pig_id
```

## Quick Start 快速开始
训练：
```
python tools/train.py \
  --data data/processed \
  --config configs/pignet.yaml \
  --save runs/pignet
```

评估（生成整体与按地平线的 RMSE、R2）：
```
python tools/eval.py \
  --data data/processed \
  --config configs/pignet.yaml \
  --ckpt runs/pignet/best.ckpt
```

可选：与 TimesNet / LSTM / 树模型对比：
```
python tools/benchmark.py --data data/processed --configs configs/benchmarks/
```

## Expected Outputs 期望输出
- logs/ 与 runs/ 下自动保存曲线、配置与最优权重
- metrics.json：包含 All（7 日均值）与 h1–h7 的 RMSE、R2
- 可视化：单猪轨迹与预测对齐图（可选）

## Reproducibility 复现设置
- 划分：按 pig_id 分组的 K 折（防止个体信息泄露），或 train/val/test=90/10 的外层划分
- 窗口构建：日滑动窗，步长 1 天；严格在各划分内部构窗，避免时间泄露
- 指标：按猪宏平均（先算每只猪的指标，再在个体之间平均）

## Model Zoo 模型动物园（示例文件名）
- configs/pignet.yaml（主方法）
- configs/lstm.yaml
- configs/timesnet.yaml
- configs/lightts.yaml
- configs/xgboost.yaml, configs/lightgbm.yaml, configs/catboost.yaml, configs/rf.yaml

## Results 结果（占位示例）
表格中的数值请根据你的实验日志或论文最终结果替换。

RMSE（kg，按 h1–h7 与 7 日均值）：
```
| Model   |  h1 |  h2 |  h3 |  h4 |  h5 |  h6 |  h7 |  All |
|---------|-----|-----|-----|-----|-----|-----|-----|------|
| LSTM    |     |     |     |     |     |     |     |      |
| TimesNet|     |     |     |     |     |     |     |      |
| PigNet  |     |     |     |     |     |     |     |      |
```

R2：
```
| Model   |  h1 |  h2 |  h3 |  h4 |  h5 |  h6 |  h7 |  All |
|---------|-----|-----|-----|-----|-----|-----|-----|------|
| LSTM    |     |     |     |     |     |     |     |      |
| TimesNet|     |     |     |     |     |     |     |      |
| PigNet  |     |     |     |     |     |     |     |      |
```

消融：
```
| Variant             | All RMSE | All R2 |
|--------------------|----------|--------|
| LSTM               |          |        |
| LSTM + TAM         |          |        |
| LSTM + DSCSM       |          |        |
| PigNet (TAM+DSCSM) |          |        |
```

## Folder Structure 目录结构（建议）
```
PigNet/
  configs/
    pignet.yaml
    lstm.yaml
    ...
  tools/
    prepare_data.py
    train.py
    eval.py
    benchmark.py
  pignet/
    models/
      pignet.py
      lstm_backbone.py
      tam.py
      dscsm.py
      film_se.py
    data/
      dataset.py
      splits.py
    utils/
      metrics.py
      losses.py
      plotting.py
  data/ (ignored by git)
  runs/ (ignored by git)
  requirements.txt
  README.md
  LICENSE
```

## Citation 引用
如果本项目或数据对你有帮助，请引用：
```
@article{PigNet2025,
  title   = {PigNet: Multi-step forecasting of pig body weight from multi-source data using an LSTM with time-aware attention and depthwise separable convolutional smoothing},
  author  = {Xu, Yifei and Li, Xiaomeng and Yang, Ying and Li, Xiaocheng and Lin, Taiming and Wu, Zhenlong},
  journal = {preprint / journal under submission},
  year    = {2025}
}
```

## License 许可
建议使用 MIT 或 Apache-2.0 许可证。

## Contact 联系方式
欢迎提交 Issue 交流；或发送邮件至 maintainer@example.com（请替换为你的邮箱）。
