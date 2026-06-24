# GPU训练交接说明

本文档用于让后续接手的 agent 快速了解当前项目状态、已经完成的 GPU 训练线、最新实验结果、提交文件和后续优化方向。

## 项目和环境

- 项目目录：`D:\Study\wairc-2026`
- 用户 PyCharm 解释器：`D:\Develop\Miniconda\envs\deepl\python.exe`
- Python：`3.12.12`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- PyTorch：`2.11.0+cu128`
- CUDA 可用性已验证：`torch.cuda.is_available() == True`

验证命令：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出类似：

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

## 依赖

基础依赖文件：

```text
requirements.txt
```

当前内容：

```text
numpy>=1.26
```

GPU 依赖文件：

```text
requirements-gpu.txt
```

当前内容：

```text
--index-url https://download.pytorch.org/whl/cu128
torch
```

安装命令：

```powershell
cd D:\Study\wairc-2026
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m pip install -r requirements.txt
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m pip install --timeout 1200 --retries 10 -r requirements-gpu.txt
```

如果需要清华源安装普通包：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：CUDA 版 PyTorch 最稳仍使用 `https://download.pytorch.org/whl/cu128`。

## 已完成的代码变更

新增 GPU 训练相关文件：

- `src/torch_iq.py`
  - `IQDataset`：直接读取 `.npz` 原始 IQ 数据。
  - 把 3 个节点整理为 6 通道输入：`node0 I/Q + node1 I/Q + node2 I/Q`。
  - 缺失节点用 0 填充。
  - 额外输入 6 维 meta 特征：3 个 `has_node` + 3 个归一化 sample rate。
  - `IQCNN`：1D CNN 多标签模型。
  - 阈值搜索和指标函数：`exact_match_accuracy_multihot`、`macro_f1_score`、`find_best_threshold`。

- `src/train_torch_iq.py`
  - PyTorch GPU 训练入口。
  - 默认设备为 `cuda`，如果 CUDA 不可用会报错。
  - 使用 AMP 混合精度。
  - 使用 `BCEWithLogitsLoss` 做 9 类多标签训练。
  - 每个 epoch 在验证集上搜索最佳阈值。
  - 保存验证集 `exact_match_accuracy` 最好的 checkpoint。

- `src/predict_torch_iq.py`
  - 加载 `outputs\models\iq_cnn.pt`。
  - 使用 checkpoint 内保存的 `sequence_pairs` 和 `threshold`。
  - 对 public test set 生成比赛提交文件。

- `requirements-gpu.txt`
  - CUDA 版 PyTorch 安装配置。

- `README.md`
  - 已追加 RTX 4060 GPU 训练、预测、提交说明。

说明：原 baseline 流程未删除，仍可用：

- `src/train_baseline.py`
- `src/predict.py`
- `outputs\models\baseline.pkl`
- `outputs\submissions\submission_baseline.txt`

## 数据和任务理解

训练集：

```text
data_and_code\ai_radio_2026_qualifying_release\train
```

公共测试集：

```text
data_and_code_patch-1\test_public_v1.1
```

任务输出是 9 维 multi-hot 标签。公共测试集没有标签，所以不能在本地计算测试集准确率；本地评估只能依赖从训练集切出来的 validation split。

当前 split 逻辑沿用 baseline：

- `random_seed = 2026`
- `val_ratio = 0.2`
- 训练样本：`5999`
- 验证样本：`1501`

## Baseline 对照

baseline 指标文件：

```text
outputs\metrics\baseline_metrics.json
```

baseline 本地验证指标：

```text
local_exact_match_accuracy: 0.1905396402398401
```

baseline 使用的是手工特征 + `NearestCentroidClassifier`，没有使用 GPU。

## GPU 模型最新结果

最新训练命令对应配置：

```text
sequence_pairs: 65536
batch_size: 32
epochs_requested: 40
seed: 2026
device: cuda
amp: true
```

最新指标文件：

```text
outputs\metrics\iq_cnn_metrics.json
```

最新最佳结果：

```text
best_epoch: 37
best_validation.loss: 0.22386202837609195
best_validation.threshold: 0.45
best_validation.exact_match_accuracy: 0.47168554297135246
best_validation.macro_f1: 0.685044802900291
```

和 baseline 对比：

```text
baseline exact: 0.1905
GPU IQCNN exact: 0.4717
```

结论：当前 GPU 模型明显优于 baseline，应优先提交 GPU 模型生成的文件。

## 当前模型和提交文件

当前最佳模型：

```text
outputs\models\iq_cnn.pt
```

当前提交文件：

```text
outputs\submissions\submission_iq_cnn.txt
```

提交文件已通过格式验证：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.validate_submission outputs\submissions\submission_iq_cnn.txt
```

验证结果：

```text
Submission validation passed: outputs\submissions\submission_iq_cnn.txt
```

比赛网站应提交：

```text
outputs\submissions\submission_iq_cnn.txt
```

不要提交 `.pt` 模型，也不要提交 `metrics.json`。

## 常用命令

正式训练：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.train_torch_iq --sequence-pairs 65536 --epochs 40 --batch-size 32
```

生成提交：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.predict_torch_iq
```

验证提交格式：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.validate_submission outputs\submissions\submission_iq_cnn.txt
```

查看指标：

```powershell
Get-Content outputs\metrics\iq_cnn_metrics.json
```

查看 GPU：

```powershell
nvidia-smi -l 1
```

## GPU 利用率观察和瓶颈

观察到的现象：

- `nvidia-smi` 中 Python 进程占用约 `1.9GB` 显存。
- GPU 利用率是脉冲式的，大部分时间可能显示 `0%`，偶尔冲到 `80%+`。
- 这说明 CUDA 确实在跑，但 GPU 经常等待 CPU/磁盘准备下一批数据。

主要瓶颈：

```text
每个 batch 都要打开多个 .npz 文件 -> 解压/读取 numpy 数组 -> I/Q 下采样 -> 转 float32 -> 传到 GPU
```

当前数据管线比模型计算更容易成为瓶颈。

短期可尝试：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.train_torch_iq --sequence-pairs 65536 --epochs 40 --batch-size 64 --num-workers 4
```

如果 Windows 下 `num-workers 4` 不稳定，改成：

```text
--num-workers 2
```

如果显存溢出，优先降低：

```text
--batch-size
```

## 已知问题和注意事项

1. 公共测试集没有标签，不能本地评估 public test accuracy。
2. `Submission validation passed` 只代表格式正确，不代表比赛分数高。
3. 当前 `outputs\models\iq_cnn.pt` 会被新的训练覆盖；做实验时建议使用不同 `--model-path` 和 `--metrics-path`。
4. `.idea` 下存在一些 IDE 产生的变更，不应随意回滚，除非用户明确要求。
5. 当前模型只使用时域 I/Q 序列和简单 meta 信息，没有使用显式频域分支。

## 下一步优化建议

建议按优先级推进。

### 1. 保留当前最佳模型，做新实验时不要覆盖

当前最佳模型建议另存一份：

```powershell
Copy-Item outputs\models\iq_cnn.pt outputs\models\iq_cnn_pairs65536_seed2026_best.pt
Copy-Item outputs\metrics\iq_cnn_metrics.json outputs\metrics\iq_cnn_pairs65536_seed2026_best.json
```

### 2. 尝试更大 batch 和 worker

目标是提高 GPU 利用率和训练速度：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.train_torch_iq --sequence-pairs 65536 --epochs 40 --batch-size 64 --num-workers 4 --model-path outputs\models\iq_cnn_pairs65536_bs64.pt --metrics-path outputs\metrics\iq_cnn_pairs65536_bs64.json
```

### 3. 尝试更宽模型

当前默认 `width=64`。可试：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.train_torch_iq --sequence-pairs 65536 --epochs 40 --batch-size 32 --width 96 --num-workers 4 --model-path outputs\models\iq_cnn_pairs65536_w96.pt --metrics-path outputs\metrics\iq_cnn_pairs65536_w96.json
```

如果显存仍充足，可试 `--width 128`。

### 4. 多 seed 训练并集成

建议训练多个 seed：

```powershell
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.train_torch_iq --sequence-pairs 65536 --epochs 40 --batch-size 32 --width 96 --seed 2026 --model-path outputs\models\iq_cnn_w96_seed2026.pt --metrics-path outputs\metrics\iq_cnn_w96_seed2026.json
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.train_torch_iq --sequence-pairs 65536 --epochs 40 --batch-size 32 --width 96 --seed 2027 --model-path outputs\models\iq_cnn_w96_seed2027.pt --metrics-path outputs\metrics\iq_cnn_w96_seed2027.json
& "D:\Develop\Miniconda\envs\deepl\python.exe" -m src.train_torch_iq --sequence-pairs 65536 --epochs 40 --batch-size 32 --width 96 --seed 2028 --model-path outputs\models\iq_cnn_w96_seed2028.pt --metrics-path outputs\metrics\iq_cnn_w96_seed2028.json
```

后续可新增 ensemble predict：读取多个 checkpoint，对 sigmoid 概率取平均，再用验证集阈值或固定阈值生成 multi-hot。

### 5. 做 tensor 缓存

当前训练慢和 GPU 利用率不稳定的根因是每轮反复打开 `.npz`。建议新增预处理缓存：

```text
outputs\cache\iq_tensor_train_pairs65536.npy
outputs\cache\iq_tensor_val_pairs65536.npy
```

或使用 `.pt` 保存 tensor。训练时直接读取缓存，可以显著减少 CPU/磁盘瓶颈。

### 6. 加数据增强

可考虑：

- 随机时间裁剪，而不是固定 `linspace` 下采样。
- 随机幅度缩放。
- 轻微高斯噪声。
- 随机时间平移。
- I/Q 小幅扰动。

### 7. 加频域分支

当前模型只吃时域 IQ。无线电任务通常频域特征重要，可考虑：

```text
时域 IQ CNN 分支 + FFT/STFT 频域 CNN 分支 + meta 特征 -> 分类头
```

这可能比单纯加宽模型更有上限。

## 交接重点

当前最重要事实：

1. GPU 训练线已经跑通。
2. 当前最佳本地 exact match 是 `0.4717`，明显优于 baseline 的 `0.1905`。
3. 当前提交文件 `outputs\submissions\submission_iq_cnn.txt` 已生成并通过格式验证。
4. 下一步如果只是参赛提交，直接提交 `submission_iq_cnn.txt`。
5. 下一步如果继续优化，先保护当前最佳 checkpoint，再做 `width`、`batch-size`、`num-workers`、多 seed 和缓存实验。
