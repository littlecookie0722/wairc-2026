# 冲高分 STFT 频谱图方案说明

## 1. 当前项目状态

当前项目已经把早期基础模型归档到：

```text
archived_baselines/
```

归档内容包括：

- `nearest_centroid/`：手工特征 + 最近质心分类器，历史本地验证约 `0.1905`。
- `iq_cnn/`：原始 IQ 时序 1D CNN，历史最好本地验证约 `0.4717`。

现在 `src/` 只保留冲高分主线和必要通用工具：

```text
src/config.py
src/data.py
src/submission.py
src/validate_submission.py
src/spectrogram.py
src/train_spectrogram.py
src/train_spectrogram_kfold.py
src/search_spectrogram_kfold_thresholds.py
src/predict_spectrogram.py
src/predict_spectrogram_kfold.py
```

后续默认使用 STFT 频谱图方案。

## 2. 为什么换成 STFT 频谱图

原始 IQ 信号可以理解成一段无线电“录音”。

早期模型直接看时域波形，能学到一部分规律，但无线电信号的类别差异往往更明显地体现在频域：

- 哪些频段能量更强。
- 频谱纹理是否稳定。
- 有没有多个频谱峰。
- 信号随时间的频率变化。
- 不同节点看到的频谱形态是否不同。

STFT 的作用就是把一维时间信号变成二维图：

```text
横轴：时间
纵轴：频率
颜色/数值：能量强弱
```

这样无人机无线电信号就可以变成类似图片的频谱图，再交给成熟的图像模型学习。

## 3. 主线模型原理

每条样本最多有 3 个节点：

```text
iq_node0
iq_node1
iq_node2
```

每个节点的 IQ 数据先转成复数信号：

```text
I + jQ
```

然后对每个节点做 STFT，得到 3 张频谱图。模型输入可以理解为：

```text
3 个通道的频谱图
```

然后使用 ImageNet 预训练图像模型，例如：

```text
resnet34
efficientnet_b0
convnext_tiny
```

模型最后输出 9 个类别概率：

```text
[p0, p1, p2, p3, p4, p5, p6, p7, p8]
```

再用验证集搜索出来的推理规则，把概率转成提交要求的 9 维 multi-hot：

```text
[0, 1, 0, 0, 0, 0, 0, 1, 0]
```

## 4. 文件职责

### `src/spectrogram.py`

核心模型和数据处理：

- IQ 转 STFT 频谱图。
- 频谱图缓存。
- 数据增强 `SpecAugment`。
- 图像模型 `DroneClassifier`。
- 损失函数。
- 阈值/推理规则搜索工具。

### `src/train_spectrogram.py`

单模型训练脚本。

用途：

- 快速验证环境。
- 快速试一个模型配置。
- 不用于最终最高分方案。

输出：

```text
outputs/spectrogram/best_model.pth
outputs/spectrogram/best_rule.json
```

### `src/train_spectrogram_kfold.py`

k-fold 训练脚本。

这是当前冲高分主入口。

默认：

```text
n_splits = 5
arch = resnet34
```

如果不传 `--fold`，会顺序训练全部 5 折。

输出示例：

```text
outputs/spectrogram_kfold/best_model_r34_fold0.pth
outputs/spectrogram_kfold/best_model_r34_fold1.pth
outputs/spectrogram_kfold/best_model_r34_fold2.pth
outputs/spectrogram_kfold/best_model_r34_fold3.pth
outputs/spectrogram_kfold/best_model_r34_fold4.pth
outputs/spectrogram_kfold/oof_r34_fold0.npz
...
```

### `src/search_spectrogram_kfold_thresholds.py`

读取各 fold 的 OOF 预测结果，搜索最适合严格匹配分数的推理规则。

输出：

```text
outputs/spectrogram_kfold/best_rule_kfold.json
```

### `src/predict_spectrogram_kfold.py`

加载一组或多组 fold checkpoint，对公开测试集预测。

输出：

```text
outputs/submissions/submission_spectrogram_kfold.txt
```

### `src/validate_submission.py`

提交前校验格式。它只检查格式，不代表分数高。

## 5. 一组五折模型怎么跑

推荐第一组先用 ResNet34：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag r34 --arch resnet34 --epochs 40 --batch-size 16 --num-workers 2
```

这条命令会自动训练：

```text
r34 fold0
r34 fold1
r34 fold2
r34 fold3
r34 fold4
```

如果只想单独训练第 0 折：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --fold 0 --tag r34 --arch resnet34 --epochs 40 --batch-size 16 --num-workers 2
```

## 6. 多组五折模型怎么跑

一组 ResNet34：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag r34 --arch resnet34 --epochs 40 --batch-size 16 --num-workers 2
```

一组 EfficientNet-B0：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag b0 --arch efficientnet_b0 --epochs 40 --batch-size 24 --num-workers 2
```

一组 ConvNeXt-Tiny：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag cnx --arch convnext_tiny --epochs 40 --batch-size 12 --num-workers 2
```

这就是 3 组五折模型，总共 15 个 checkpoint。

## 7. 搜索规则和生成提交

只使用 ResNet34 五折：

```powershell
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt
```

使用 ResNet34 + EfficientNet-B0：

```powershell
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34 b0
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34 b0
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt
```

使用三组模型：

```powershell
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34 b0 cnx
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34 b0 cnx
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt
```

## 8. 输出文件说明

训练输出：

```text
outputs/spectrogram_kfold/
```

STFT 缓存：

```text
outputs/cache/stft/
```

提交文件：

```text
outputs/submissions/submission_spectrogram_kfold.txt
```

提交比赛平台时只提交 `.txt` 文件。

不要提交：

```text
.pth
.json
.npz
```

## 9. 显存和参数建议

如果显存不足，优先调小：

```text
--batch-size
```

常见建议：

| 模型 | 建议 batch size |
| --- | --- |
| resnet34 | 16 |
| efficientnet_b0 | 24 |
| convnext_tiny | 8 或 12 |
| resnet50 | 8 |

如果 Windows 下 DataLoader 不稳定，把：

```text
--num-workers 2
```

改成：

```text
--num-workers 0
```

## 10. 零基础训练顺序

第一步，确认环境能跑：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram --epochs 3 --batch-size 8 --num-workers 0 --max-samples 300
```

第二步，跑 ResNet34 五折：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag r34 --arch resnet34 --epochs 40 --batch-size 16 --num-workers 2
```

第三步，生成第一版高分提交：

```powershell
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34
.\.venv\Scripts\python.exe -m src.validate_submission outputs/submissions/submission_spectrogram_kfold.txt
```

第四步，再增加第二组模型：

```powershell
.\.venv\Scripts\python.exe -m src.train_spectrogram_kfold --tag b0 --arch efficientnet_b0 --epochs 40 --batch-size 24 --num-workers 2
.\.venv\Scripts\python.exe -m src.search_spectrogram_kfold_thresholds --tags r34 b0
.\.venv\Scripts\python.exe -m src.predict_spectrogram_kfold --tags r34 b0
```

第五步，线上提交后记录分数和使用的 tags。

## 11. 注意事项

- 公开测试集没有标签，不能在本地计算测试准确率。
- `validate_submission` 只检查格式，不检查预测质量。
- 每次训练新架构时使用不同 `--tag`，否则会覆盖同名模型。
- 每次提交前记录：
  - 使用了哪些 tags。
  - 使用了哪个 `best_rule_kfold.json`。
  - 线上分数是多少。
- 不要删除 `outputs/spectrogram_kfold/` 下的 OOF 文件；阈值搜索要用它们。

