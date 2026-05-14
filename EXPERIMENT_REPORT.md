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

### 8.4 Current Deliverable Candidate

当前工作目录里的 `starter_code/result.zip` 已切换回已获得官方反馈的最佳包：

```text
results_test_v6_blend_noisy_official_a850.zip
```

验证内容：

```text
files = 200
shape = (50000, 3)
dtype = float32
non-finite = 0
sha256 = 22590557f6e7ad34d658b581b4939bfb7ba9ae6ca32c2540bf6873c75979abb2
```

该候选已有官方分数：

```text
score          = 59.37
CD_score       = 44.81
P2S_score      = 73.94
mean_CD_pred   = 0.000137
mean_CD_noisy  = 0.000246
mean_P2S_pred  = 0.000076
mean_P2S_noisy = 0.000196
```

选择理由：

- 这是目前所有已经获得官方反馈的提交中最高的分数。
- 后续低 alpha / 新候选官方反馈约为 59 分，没有形成明显突破。
- 新的 sklearn/AlphaGate 方向目前只在 official-like validation 上验证，尚未生成可直接信任的官方测试提交包。
- 因此当前可交付 `result.zip` 应优先保证已知有效，而不是提交未经官方验证的实验候选。

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
9. 进一步构造 official-like validation，把 `mean_CD_noisy` 校准到接近官方量级。
10. 在 official-like validation 上确认：全局 alpha 的收益有限，点级 alpha/gate 才是后续主线。
11. 通过 oracle 分析发现，沿 base 方向的每点最优 alpha 可把本地 official-like 分数从 64.76 提到 71.81，说明 base 方向不是核心瓶颈，核心是每点 step/gate 控制。
12. 训练 sklearn alpha gate 后，OOF 分数达到 66.16，证明自适应 gate 有真实泛化收益，但仍未接近 oracle。
13. 因此新增 `AlphaGateRefiner`，准备通过 Jittor 长训练学习正式的点级 alpha head。

## 11. Recommended Next Steps

当前最有依据的下一步：

1. 当前先交付已知官方最佳 `result.zip` (`alpha=0.85`)。
2. 启动 `AlphaGateRefiner` 长训练，固定 base VM 方向，只学习每点 alpha。
3. 训练结束后，对 official-like validation 预测并评估 CD/P2S。
4. 如果 AlphaGate checkpoint 超过 sklearn OOF 66.16，生成官方测试集预测包作为下一版提交。
5. 如果 AlphaGate 未超过 sklearn OOF，则说明训练数据/特征仍不足，需要把 sklearn gate 的几何特征或 oracle alpha 监督更直接地并入模型。

自适应 alpha 的设计依据：

```text
flat/simple geometry: lower alpha, stronger denoising rollback
thin/edge-rich geometry: higher alpha, preserve surface projection
```

长期方向：

- 继续 PCT refiner，但训练目标必须匹配低噪声官方分布。
- 训练时混入低 sigma 噪声，并对 displacement 加更强约束。
- 损失函数需要同时约束 surface proximity 和 distribution preservation。
- 使用 validation noise schedule 接近官方 `mean_CD_noisy ~= 0.000246`。

## 13. Update On 2026-05-14: Official-Like Validation And AlphaGate

### 13.1 New Official-Like Validation

新增脚本：

- `starter_code/scripts/make_noisy_as_pred.py`
- `starter_code/scripts/grid_noise_schedule.py`
- `starter_code/scripts/evaluate_candidate_table.py`

目标是构造更接近官方反馈的 validation。当前选用：

```text
starter_code/local_eval_official_low
sigmas      = 0.0050, 0.0055, 0.0060, 0.0065
noise_types = gaussian, laplace, anisotropic
outlier_p   = 0.0005
```

该 validation 的 noisy 统计：

```text
mean_CD_noisy  = 0.00026112
mean_P2S_noisy = 0.00017076
```

它比早期 `local_eval_testcats_100` 更接近官方 `mean_CD_noisy ~= 0.000246`。

### 13.2 Alpha And Weak Postprocess Results

在 official-like validation 上重新评估全局 alpha：

```text
a550  score=50.98
a650  score=56.21
a750  score=60.19
a850  score=62.93
a1000 score=64.76
```

这说明在该 validation 上 `alpha=1.0` 最好，但官方反馈中 `alpha=0.85` 更好，二者存在 domain gap。因此只靠全局 alpha 继续搜索风险很高。

新增 `adaptive_displacement_clip.py`，按 noisy 的局部 kNN 半径裁剪 base displacement：

```text
base_a1000      score=64.76 CD=53.97 P2S=75.55
clip_k16_s1000 score=65.00 CD=54.26 P2S=75.75
```

结论：异常位移裁剪有弱增益，但只有 +0.24，不能作为 70/90 分突破。

### 13.3 Oracle Alpha Analysis

新增 oracle 分析：固定 base 预测方向，只用 clean 计算每个点沿 base displacement 的最优 alpha。

```text
base_a1000                 score=64.76 CD=53.97 P2S=75.55
oracle_dir_alpha_clip015   score=71.81 CD=62.04 P2S=81.59
```

这是目前最重要的诊断结果：

- base displacement 方向仍有价值；
- 真正缺的是每点“走多远”的 step/gate；
- 如果能学习接近 oracle 的 alpha，70+ 是有现实空间的；
- 但 90+ 仍需要更强的模型、训练分布和官方反馈闭环。

### 13.4 Sklearn Alpha Gate

用户允许安装依赖后，在 `jt` 环境安装：

```text
scikit-learn==1.3.2
joblib
threadpoolctl
```

新增脚本：

- `starter_code/scripts/fit_alpha_gate_sklearn.py`
- `starter_code/scripts/fit_alpha_gate_ridge.py`
- `starter_code/scripts/ratio_alpha_gate.py`

使用测试时可得特征训练 alpha regressor：

- base displacement
- displacement norm
- noisy local radius
- displacement/radius ratio
- PCA linearity/planarity/scattering/anisotropy
- point radial position

结果：

```text
base_a1000         score=64.76 CD=53.97 P2S=75.55
sklearn_hgb_gate   score=66.59 CD=56.04 P2S=77.13
sklearn_rf_gate    score=66.72 CD=56.19 P2S=77.26
hgb_oof            score=66.16 CD=55.62 P2S=76.70
oracle_dir_alpha   score=71.81 CD=62.04 P2S=81.59
```

`hgb_oof` 使用 group K-fold，使每个样本由没见过该样本的 gate 预测，因此比 full-fit 更接近真实泛化。它仍比 base 高 +1.40，说明自适应 alpha 不是纯过拟合。

### 13.5 AlphaGateRefiner

新增模型：

- `starter_code/src/model/low_noise_refiner.py::AlphaGateRefiner`
- `starter_code/configs/model/alpha_gate_refiner_pct_neighbor.yaml`
- `starter_code/configs/system/alpha_gate_refiner.yaml`
- `starter_code/configs/task/debug_alpha_gate_refiner.yaml`
- `starter_code/configs/task/train_alpha_gate_refiner_fast.yaml`

核心公式：

```python
pc_base = frozen_vm(noisy)
alpha = AlphaGate(noisy, pc_base, pc_base - noisy, local_features)
denoised = noisy + alpha * (pc_base - noisy)
```

训练目标：

```text
loss_point    : denoised vs clean
loss_patch_cd : local patch CD
loss_alpha    : alpha vs clipped oracle alpha
loss_repulsion: point distribution preservation
```

这个设计刻意避免自由 residual direction，因为之前 residual/refiner 分支会破坏已有方向，导致分数低于 base。AlphaGate 只控制步长，直接对齐 oracle 分析。

Smoke test 已通过：

```text
task: configs/task/debug_alpha_gate_refiner.yaml
steps: 2
loss: about 0.026-0.027
CUDA enabled
```

Jittor 在 sandbox 内无法写 `/data/qiaojiaxuan/jittor_home` lock，因此 Jittor 训练需要在 sandbox 外运行。

### 13.6 Current Deliverables

当前可交付结果：

```text
starter_code/result.zip
source = starter_code/results_test_v6_blend_noisy_official_a850.zip
sha256 = 22590557f6e7ad34d658b581b4939bfb7ba9ae6ca32c2540bf6873c75979abb2
known official score = 59.37
```

当前尚未完成：

- AlphaGate 长训练；
- AlphaGate checkpoint 的 official-like validation；
- AlphaGate 官方测试集预测包；
- 官方提交闭环验证。

因此项目目标 `official_score > 90` 仍未达成，当前工作重点是用 AlphaGate 尝试突破 66-72 的区间。

### 13.7 AlphaGate Long-Run Evaluation

AlphaGate 长训练已完成：

```text
task       = configs/task/train_alpha_gate_refiner_fast.yaml
epochs     = 8
steps/ep   = 1200 effective steps
checkpoint = experiments/alpha_gate_refiner_pct_neighbor/checkpoint_7.pkl
final train mean loss = 0.017517
final EMA loss        = 0.017712
```

随后使用官方低噪本地验证划分预测：

```text
task = configs/task/predict_official_low_alpha_gate_final.yaml
output = results_official_low_alpha_gate_final
```

完整 CD/P2S 对比结果：

| name | final | CD | P2S | mean_CD_pred | mean_P2S_pred | mean_disp |
|---|---:|---:|---:|---:|---:|---:|
| rf_gate | 66.72 | 56.19 | 77.26 | 0.00011006 | 0.00003670 | 0.003942 |
| hgb_oof | 66.16 | 55.62 | 76.70 | 0.00011156 | 0.00003763 | 0.003916 |
| adaptive_clip_k16_s1000 | 65.00 | 54.26 | 75.75 | 0.00011560 | 0.00003986 | 0.004185 |
| base_a1000 | 64.76 | 53.97 | 75.55 | 0.00011644 | 0.00004022 | 0.004279 |
| alpha_gate_final | 59.74 | 50.01 | 69.46 | 0.00012834 | 0.00005287 | 0.003407 |

结论：

- AlphaGate 长训没有超过现有后处理/gate 路线，反而低于 base alpha。
- 它的 mean displacement 明显偏小，说明学到的 gate 过于保守，P2S 损失尤其差。
- 这不是 checkpoint 缺失或预测配置错误，`checkpoint_7.pkl` 存在且已被预测脚本加载。
- 当前不应使用 AlphaGate 结果替换 `starter_code/result.zip`。
- 目前本地验证最强仍是 `rf_gate` / `hgb_oof`，但 official 已验证过 `a850` 只有 59.37，因此本地验证到官方测试存在分布差异，后续需要新的官方反馈闭环或更可靠的验证拆分。

## 14. Files Excluded From Git

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
