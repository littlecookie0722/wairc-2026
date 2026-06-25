# AI 无线电无人机识别冲高分版本

当前主线已经整理为：

```text
STFT 频谱图 + ImageNet 预训练图像模型 + k-fold ensemble
```

早期的手工特征 baseline 和原始 IQ CNN 已归档到：

```text
archived_baselines/
```

后续默认不再使用归档模型，除非需要回看历史实验。

## 数据目录

训练集，有标签：

```text
data_and_code/ai_radio_2026_qualifying_release/train/
```

公开测试集，无标签：

```text
data_and_code_patch-1/test_public_v1.1/
```

提交文件必须是 9 维 multi-hot 文本格式，提交前务必运行校验脚本。

## 安装依赖

CPU/通用依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

GPU/PyTorch 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

如果你使用的是 PyCharm/Conda 解释器，请把 `.\.venv\Scripts\python.exe` 换成你的解释器路径，例如：

```powershell
D:/Develop/Miniconda/envs/deepl/python.exe
```

## 推荐冲分流程

先跑一组 ResNet34 五折：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag r34 --arch resnet34 --epochs 40 --batch-size 16 --num-workers 2
```

说明：不传 `--fold` 时，脚本会顺序训练全部 5 折。

搜索 OOF 最优推理规则：

```powershell
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34
```

生成公开测试集提交：

```powershell
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34
```

校验提交格式：

```powershell
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt
```

## 多模型融合

训练 EfficientNet-B0 五折：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag b0 --arch efficientnet_b0 --epochs 40 --batch-size 24 --num-workers 2
```

训练 ConvNeXt-Tiny 五折：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag cnx --arch convnext_tiny --epochs 40 --batch-size 12 --num-workers 2
```

融合多组五折模型：

```powershell
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34 b0 cnx
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34 b0 cnx
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt
```

## 单模型试跑

如果只是想先确认环境、缓存和显存是否正常，可以跑单模型：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram --epochs 3 --batch-size 8 --num-workers 0 --max-samples 300
```

正式冲分仍建议使用 `src.train_spectrogram_kfold`。

## 详细说明

完整中文说明见：

```text
docs/冲高分STFT频谱图方案说明.md
```

