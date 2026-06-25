# Archived Baseline Models

这个目录保存早期探索模型，默认后续不再作为主线使用。

## nearest_centroid

早期 CPU baseline：

- 手工统计/FFT 特征
- 最近质心分类器
- 本地验证 strict exact-match 约 `0.1905`

这条路线的意义是跑通数据读取、训练、预测和提交格式，但分数上限较低。

## iq_cnn

中间阶段 GPU baseline：

- 直接读取原始 IQ 时序
- 6 通道 1D CNN
- 已记录最好本地验证 strict exact-match 约 `0.4717`

这条路线明显强于最早 baseline，但当前冲高分主线已经转向 STFT 频谱图 + 预训练图像模型 + k-fold ensemble。

## 注意

这些文件从 `src/` 归档过来，保留历史参考价值。由于相对导入路径已经不在原包结构下，若要重新运行，需要先调整 import 或从历史版本恢复到 `src/`。

