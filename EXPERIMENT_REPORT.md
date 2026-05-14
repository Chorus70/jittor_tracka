# Jittor Track A Point Cloud Denoising Experiment Report

## 1. Objective And Constraints

本项目目标是完成第六届计图人工智能挑战赛 Track A 点云去噪任务。输入为含噪点云，输出同路径结构下的 `denoised.npy`。官方评价包含两个指标：

- CD: denoised point cloud 到 clean point cloud 的双向 Chamfer Distance 改善率。
- P2S: denoised points 到原始 mesh surface 的 point-to-surface distance 改善率。

最终分数为：

```text
score = 0.5 * CD_score + 0.5 * P2S_score
metric_score = clamp(100 * (1 - metric_pred / metric_noisy), 0, 100)
```

因此，模型不能只把点云强行平滑到曲面附近。过强的位移会破坏点云覆盖和细节，导致 CD 下降；过弱的位移又无法充分降低 P2S。后续所有实验都围绕这个 CD/P2S 平衡展开。

## 2. Repository And Data Inspection

初始检查内容包括：

- 阅读比赛 PDF，确认输出格式、数据结构、评分方式。
- 检查 `starter_code` 中的 Jittor 训练/推理框架。
- 检查 `dataset_train.tar`、`dataset_test_noisy`、`datalist`、训练脚本和配置文件。
- 解压并确认 `dataset_train` 可用，包含 ShapeNet mesh: `models/model_normalized.obj`。
- 确认测试集 `dataset_test_noisy` 有 200 个样本，每个样本包含 `noisy.npy`。
- 确认最终提交包需要包含 200 个 `denoised.npy`，路径格式为 `shapenet/<category>/<model_id>/denoised.npy`。

数据和代码均可用，但存在两个重要现实问题：

- 官方测试集 clean/mesh 不在服务器上，因此本地无法直接计算官方分数。
- 本地构造验证集的噪声分布与官方测试集并不完全一致，后来官方反馈证明本地验证高估了强去噪方案的效果。

## 3. Baseline Pipeline

项目原始框架采用 `run.py` 读取 YAML task config，加载 data/model/system/transform 配置后执行训练或预测。主要代码路径：

- `starter_code/run.py`: 统一入口。
- `starter_code/src/data/*`: 数据集、采样、增强和 datalist 处理。
- `starter_code/src/model/vm.py`: 主要神经网络和多个变体。
- `starter_code/src/system/spec.py`: 训练、保存 checkpoint、预测流程。
- `starter_code/src/system/vm.py`: 预测结果写出逻辑。
- `starter_code/evaluate.py`: 本地 CD/P2S 评测脚本。

原始输出逻辑会把模型预测的点位移加到 noisy points 上，并保存为 `denoised.npy`。

## 4. Environment And Training Setup

使用 Jittor CUDA 环境：

```bash
HOME=/data/qiaojiaxuan
JITTOR_HOME=/data/qiaojiaxuan/jittor_home
PATH=/data/qiaojiaxuan/miniconda3/envs/jt/bin:$PATH
```

MPI 多卡训练使用 OpenMPI + Jittor：

```bash
mpirun --mca opal_cuda_support 1 \
  --mca btl_vader_single_copy_mechanism none \
  --bind-to none -np <NPROC> \
  scripts/mpi_rank.sh python run.py --task <config>
```

训练时曾使用全部可用 GPU；推理阶段由于写出 200 个 5 万点结果及后处理较重，主要采用单进程/分块策略，避免 MPI dataloader shard 导致输出不完整。

## 5. Major Model Directions Explored

### 5.1 Robust / Fine-tune VM Baselines

基于原始 VM 模型训练和微调多个 checkpoint，例如 ckpt 19/24/29/34/39，并尝试不同推理配置：

- `predict_local_eval_mesh_20_vm_finetune_19.yaml`
- `predict_local_eval_mesh_20_vm_finetune_24.yaml`
- `predict_local_eval_mesh_20_vm_robust_29.yaml`
- `predict_local_eval_mesh_20_vm_robust_39.yaml`

观察：

- checkpoint 24 在某些局部验证上 loss 更低，但并不稳定对应更好的 CD/P2S。
- 继续从已有 checkpoint 微调会出现方向不一致的问题：旧解已经学习到某种位移分布，新目标或增强策略改变后，早期微调容易破坏旧解。
- 从头训练理论上更干净，但在当前时间和算力预算下成本过高，且本地验证分布不可靠，收益不确定。

### 5.2 Flow / Field / Surface / Normal Variants

尝试把模型输出解释为流场、局部曲面场、法向修正或多步迭代修正，相关配置包括：

- `vm_flow_ft.yaml`
- `vm_flow_latent_ft.yaml`
- `vm_flow_surface_latent_ft.yaml`
- `vm_field_ft.yaml`
- `vm_normal_ft.yaml`
- `vm_surface_ft.yaml`
- `vm_surface_teacher_ft.yaml`

代表性本地评估：

```text
flow_expanded local testcats:
CD_score  = 47.75
P2S_score = 60.93
score     = 54.34
```

结论：

- 这些方向能改善部分局部几何，但整体没有超过后处理增强的 finetune baseline。
- 多步 flow 在噪声强度较高的本地验证集上有效，但在低噪声官方分布上可能位移过大。

### 5.3 Score / DSM Directions

尝试 score matching / denoising score matching 思路，包括：

- `vm_score_dsm.yaml`
- `vm_score_dsm_cont.yaml`
- `vm_score_dsm_expanded_ft.yaml`
- `vm_score_anchor_ft.yaml`
- `vm_score_teacher_ft.yaml`
- `vm_score_hybrid_ft.yaml`

观察：

- 这类方法理论上适合多噪声强度，但当前实现和训练时间不足以收敛到可提交水平。
- 部分 one-step score 推理在本地 CD 上接近 noisy 或退化，没有成为最终主线。

### 5.4 PCT-Based Architectures

用户提供了 `pct_prev.py`，并指出 PCT (Point Cloud Transformer) 方向。参考 PCT 的 neighbor attention / point transformer 设计后，新增并训练了多个 PCT refiner 变体：

- `vm_pct.yaml`
- `vm_pct_denoise.yaml`
- `vm_pct_denoise_direct.yaml`
- `vm_pct_denoise_zero_residual.yaml`
- `vm_pct_flow_conditioned.yaml`
- `vm_refiner_pct_neighbor_pointgate_ft19.yaml`
- `vm_refiner_pct_neighbor_pointgate_pos_ft19.yaml`
- `vm_refiner_pct_neighbor_pointgate_surface_ft19.yaml`

主线完整训练：

```text
config: configs/task/train_vm_refiner_pct_neighbor_pointgate_pos_ft19_mpi.yaml
checkpoint: experiments/vm_refiner_pct_neighbor_pointgate_pos_ft19/checkpoint_e3_s600.pkl
```

设计逻辑：

- 使用 PCT 风格局部/全局点特征建模，提高对未见类别的几何泛化。
- 使用 point gate 控制 residual displacement，避免所有点被同等强度移动。
- 使用 position-aware neighbor features，让模型知道局部坐标结构。
- 以已有 ft19 预测作为 teacher/refiner 输入之一，而不是完全从 noisy 直接预测。

本地结果：

```text
PCT pointgate-pos e3_s600 local CD score: about 49.92
```

结论：

- PCT 方向架构上更合理，但当前训练轮数、数据构造和目标函数还不足以超过几何后处理主线。
- PCT 生成的 `result_mainline_pct_pointgate_pos_e3_s600_20260514_050041.zip` 未作为最终官方提交包。

## 6. Post-Processing And Ensemble Directions

由于神经网络输出存在过平滑/过位移问题，重点探索了几何后处理和 ensemble。

### 6.1 Local Tangent Projection

实现位置：

- `starter_code/scripts/postprocess_predictions.py`

核心逻辑：

1. 对每个点查询 kNN。
2. 对邻域点做 PCA。
3. 最小特征向量作为局部法向。
4. 将点沿法向投影回局部切平面附近。
5. 用 `beta` 控制投影强度。

相关函数：

- `tangent_project`
- `edge_aware_tangent_project`
- `weighted_mls_project`
- `bilateral_project`
- `normal_filter_project`

### 6.2 Edge-Aware Tangent Projection

动机：

- 普通 tangent projection 在尖锐边、薄结构和高曲率区域容易抹掉几何细节。
- 通过 PCA 特征值比值估计边缘/平面置信度，仅在更可靠的局部平面区域强投影。

核心参数：

```text
k: neighborhood size
beta: projection strength
edge_low / edge_high: confidence mapping range
clamp_scale: projection clipping scale
```

这个方向成为中后期效果最好的本地后处理分支之一。

### 6.3 Weighted MLS Projection

动机：

- 普通邻域均值容易受离群点影响。
- 使用距离权重估计局部加权中心和 covariance，提高局部平面估计稳定性。

核心函数：

```python
weighted_mls_project(pc, k, beta, sigma_scale=0.55, clamp_scale=1.5)
```

### 6.4 Blending And Category Selection

实现脚本：

- `scripts/blend_predictions.py`
- `scripts/blend_predictions_clipped.py`
- `scripts/select_predictions_by_category.py`
- `scripts/select_predictions_by_curvature.py`
- `scripts/select_predictions_by_delta_gate.py`
- `scripts/select_predictions_by_geometry_gate.py`

主要思想：

- 不同类别和几何复杂度对后处理强度敏感度不同。
- 对平面占比较大的类别可使用更强投影。
- 对薄结构/边缘复杂类别减少投影或使用 edge-aware 投影。
- 通过 delta gate/geometry gate 在多个候选目录之间选择单样本预测。

本地候选中较好的方向：

```text
results_local_eval_testcats_100_v6_geom_plan_gate:
CD_score  = 56.49
P2S_score = 73.01
score     = 64.75

results_local_eval_testcats_100_delta_gate_refiner_med0.006_p0.012_m0.100_edge_k64_a0.95_b0.30:
CD_score  = 56.31
P2S_score = 72.65
score     = 64.48

results_local_eval_testcats_100_category_select_v3:
CD_score  = 53.88
P2S_score = 71.26
score     = 62.57
```

这些分数来自本地构造验证集，不是官方分数。

## 7. Important Local Validation Problem

本地 `local_eval_testcats_100` 的噪声强度明显高于官方测试集。一个关键对比：

```text
local_eval_testcats_100 mean_CD_noisy ~= 0.00105067
official mean_CD_noisy              ~= 0.000246
```

这意味着本地验证集约为官方 CD 噪声量级的 4 倍以上。后果：

- 强去噪在本地高噪声样本上看起来有效。
- 同样强度用于官方低噪声样本时会过度移动点，CD 显著受损。
- 本地 64-65 分并不能直接外推到官方 64-65 分，更不能外推到 90 分。

后续根据官方反馈，新增了低噪声验证集生成参数：

```text
scripts/prepare_local_eval.py
--sigmas
--noise_types
--outlier_p
```

并新增低噪声快速网格脚本：

```text
scripts/grid_low_noise_projection.py
```

低噪声验证集用于判断官方噪声量级下的趋势，但仍不能替代官方 clean/mesh。

## 8. Official Submission Feedback

### 8.1 First Official Submission

提交包实际为旧主线：

```text
results_test_v6_geom_plan_gate -> result.zip
```

官方反馈：

```text
score          = 52.56
CD_score       = 35.66
P2S_score      = 69.47
mean_CD_pred   = 0.000161
mean_CD_noisy  = 0.000246
mean_P2S_pred  = 0.000082
mean_P2S_noisy = 0.000196
```

分析：

- P2S 接近 70，说明输出点确实更贴近 surface。
- CD 只有 35.66，说明点云覆盖/分布被破坏或位移过强。
- 关键问题不是完全没去噪，而是官方低噪声集上过处理。

### 8.2 Alpha Blend Toward Noisy

新增脚本：

```text
scripts/make_test_blends.py
```

核心公式：

```python
blended = noisy + alpha * (pred - noisy)
```

生成候选：

```text
alpha = 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85
```

其中 `alpha=1.0` 等价于旧主线，`alpha=0.0` 等价于 noisy 原始输入。这个方法不改变方向，只降低位移强度。

### 8.3 Second Official Submission

提交 `alpha=0.85`：

```text
results_test_v6_blend_noisy_official_a850.zip -> result.zip
```

官方反馈：

```text
score          = 59.37
CD_score       = 44.81
P2S_score      = 73.94
mean_CD_pred   = 0.000137
mean_CD_noisy  = 0.000246
mean_P2S_pred  = 0.000076
mean_P2S_noisy = 0.000196
```

分析：

- 相比 `alpha=1.0`，CD 从 35.66 提升到 44.81。
- P2S 从 69.47 提升到 73.94。
- 两项同时提升，强烈说明旧主线位移过强，适当回退到 noisy 方向是正确的。
- 但只靠全局 alpha 很可能不足以稳定达到 70，因为 mean 指标二次拟合显示 CD/P2S 最优点可能在 0.6-0.85 附近，而不是无限降低 alpha。

### 8.4 Current Candidate

当前工作目录里的 `result.zip` 已切换到：

```text
results_test_v6_blend_noisy_official_a650.zip
```

验证内容：

```text
files = 200
shape = (50000, 3)
dtype = float32
non-finite = 0
```

该候选尚未获得官方分数。选择理由：

- `alpha=0.85` 明确优于 `alpha=1.0`。
- 仅根据两个官方点线性外推会鼓励继续降低 alpha，但 noisy 本身得分为 0，因此真实曲线必然有峰值。
- 用 mean 指标做粗略二次拟合，CD 的较优区间可能靠近 0.6-0.7，P2S 较优区间靠近 0.8。
- 因此优先提交 `alpha=0.65` 作为第三个官方锚点，再决定是否转向自适应 alpha。

## 9. Code Logic Added During Exploration

### 9.1 `scripts/postprocess_predictions.py`

新增/扩展多种几何后处理：

- Tangent projection
- Edge-aware tangent projection
- Weighted MLS projection
- Bilateral projection
- Normal-filter projection

该脚本用于将模型输出进一步投影到局部几何结构附近。

### 9.2 `scripts/make_test_blends.py`

用于官方低噪声反馈后的 alpha 搜索。它读取已有预测和 noisy 输入，生成多个 blend result directory 和 zip：

```python
denoised = noisy + alpha * (pred - noisy)
```

这个脚本还支持指定 `--replace-result-alpha`，直接替换当前 `result.zip`，并备份旧结果。

### 9.3 `scripts/select_predictions_by_geometry_gate.py`

根据预测点云的 PCA 几何统计在 base/candidate 两个预测目录之间选择：

- curvature mean / p90
- planarity mean / p10

目的是对不同几何复杂度样本使用不同预测方案。

### 9.4 `scripts/select_predictions_by_delta_gate.py`

根据预测位移量、局部统计和阈值规则，在多个候选之间选择，避免某些样本被过度后处理。

### 9.5 `scripts/prepare_local_eval.py`

扩展参数：

```text
--sigmas
--noise_types
--outlier_p
```

用于构造不同噪声强度和噪声类型的本地验证集。该改动是在发现官方测试噪声远低于原本本地验证集后加入的。

### 9.6 `scripts/grid_low_noise_projection.py`

用于在低噪声验证集上快速搜索纯几何投影参数。初步结果显示：

- 小 beta 投影提升有限。
- 全量网格成本较高。
- 纯从 noisy 出发做几何投影不太可能直接解决 70 分目标。

## 10. Decision Process Summary

1. 先读题、检查数据和代码，确认项目可运行。
2. 训练/微调 baseline VM，发现 loss 和本地分数存在波动。
3. 尝试 flow/field/normal/surface/score/PCT 等架构方向。
4. PCT 架构更合理，但短期训练未超过后处理主线。
5. 发现后处理和单样本选择在本地验证上收益最大。
6. 提交本地最优 `v6_geom_plan_gate` 后，官方只得 52.56。
7. 根据官方 mean 指标判断：官方集噪声低，旧主线过处理。
8. 生成 alpha blend 候选，提交 `alpha=0.85` 后官方升到 59.37。
9. 当前选择 `alpha=0.65` 作为第三个官方 probe。
10. 如果 `alpha=0.65` 仍不足 70，下一步应做自适应 alpha，而不是继续盲目训练。

## 11. Recommended Next Steps

当前最有依据的下一步：

1. 提交当前 `result.zip` (`alpha=0.65`) 获取官方分数。
2. 若 `alpha=0.65` 比 `alpha=0.85` 好，则继续测试 `alpha=0.55` 或按类别降低 alpha。
3. 若 `alpha=0.65` 差于 `alpha=0.85`，峰值在 0.65-1.0 之间，测试 `alpha=0.75`。
4. 有三个以上官方点后，按 CD/P2S 分别拟合 alpha 曲线，再设计类别/几何自适应：

```text
flat/simple geometry: lower alpha, stronger denoising rollback
thin/edge-rich geometry: higher alpha, preserve surface projection
```

长期方向：

- 继续 PCT refiner，但训练目标必须匹配低噪声官方分布。
- 训练时混入低 sigma 噪声，并对 displacement 加更强约束。
- 损失函数需要同时约束 surface proximity 和 distribution preservation。
- 使用 validation noise schedule 接近官方 `mean_CD_noisy ~= 0.000246`。

## 12. Files Excluded From Git

为了避免 GitHub 仓库过大或泄漏生成数据，以下内容未纳入 Git：

- `dataset_train/`
- `dataset_train.tar`
- `dataset_test_noisy/`
- `starter_code/results*/`
- `starter_code/result*.zip`
- `starter_code/experiments/`
- `starter_code/logs/`
- `starter_code/local_eval*/`
- Jittor cache / Python cache

这些文件仍保留在服务器工作目录中，用于继续实验和提交。

