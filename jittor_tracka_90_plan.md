# Jittor Track A 点云去噪任务 90+ 实施计划

## 0. 目标与判断标准

本计划目标不是继续做零散调参，而是把当前约 60 分的方案改造成一个可验证、可迭代、以官方指标为导向的工程方案。

最终目标：

```text
official_score > 90
CD_score       >= 88
P2S_score      >= 92
```

需要先明确：**超过 90 不能只靠 alpha blend 或几何后处理实现**。当前官方反馈已经显示，已有方案有一定 P2S 改善能力，但 CD 被过处理显著拖垮。要超过 90，必须重新对齐训练分布、位移控制方式和损失函数。

---

## 1. 当前任务约束

赛题本质是固定点数点云去噪：

- 输入：`noisy.npy`，shape 通常为 `(50000, 3)`。
- 输出：同目录结构下的 `denoised.npy`。
- 输出格式：`np.float32`，shape 必须与输入一致。
- 提交包：`result.zip`，内部路径类似：

```text
shapenet/<category>/<model_id>/denoised.npy
```

官方指标：

```text
score = 0.5 * CD_score + 0.5 * P2S_score
metric_score = clamp(100 * (1 - metric_pred / metric_noisy), 0, 100)
```

两个指标含义：

| 指标 | 关注点 | 对方案的要求 |
|---|---|---|
| CD | denoised point cloud 与 clean point cloud 的双向 Chamfer Distance | 点云分布和覆盖不能被破坏 |
| P2S | denoised points 到原始 mesh surface 的距离 | 点要更贴近真实曲面 |

因此，点云去噪不是单纯“投影到曲面”。如果只追求 P2S，容易把点云压到局部平面或局部曲面上，造成点分布坍缩、边缘抹平、覆盖破坏，CD 会下降。

---

## 2. 当前结果诊断

### 2.1 已有官方反馈

当前最关键的两个官方反馈：

| 提交 | CD_score | P2S_score | 总分 | 解释 |
|---|---:|---:|---:|---|
| `v6_geom_plan_gate` | 35.66 | 69.47 | 52.56 | 点更贴近 surface，但点云分布被破坏，CD 很低 |
| `alpha=0.85` | 44.81 | 73.94 | 59.37 | 回退到 noisy 后，CD/P2S 同时提升，说明原预测过强 |

由此得到两个结论：

1. **当前主线方向不是完全错误**：P2S 能到 70+，说明预测确实在靠近曲面。
2. **当前主线位移过强或方向不够精确**：CD 远低于 P2S，说明点云覆盖、局部密度和细节结构受损。

### 2.2 本地验证集失效

实验记录显示：

```text
local_eval_testcats_100 mean_CD_noisy ~= 0.00105067
official mean_CD_noisy              ~= 0.000246
```

本地验证噪声强度约为官方的 4 倍以上。这会导致：

- 强去噪在本地验证上显得有效；
- 同样强度在官方低噪声测试上变成过处理；
- 本地 64~65 分无法外推到官方；
- 继续围绕旧验证集优化会误导方向。

### 2.3 现有代码方向的局限

#### 后处理方向

当前 `postprocess_predictions.py` 中已有：

- tangent projection
- edge-aware tangent projection
- weighted MLS
- bilateral projection
- normal-filter projection

这类方法可以降低 P2S，但本质是局部几何投影，缺点是：

- 不能保证点云分布与 clean 一致；
- 高曲率、薄结构、尖锐边附近容易被抹平；
- 对低噪声官方测试集容易过处理；
- 参数 `k`、`beta` 对类别和形状敏感。

#### alpha blend 方向

当前 alpha blend 公式是：

```python
blended = noisy + alpha * (pred - noisy)
```

它只能缩放已有预测，不能修正预测方向错误，也不能区分：

- 不同类别；
- 平面区域与边缘区域；
- 高噪声点与低噪声点；
- 法向噪声与切向漂移。

因此 alpha blend 是短期 probe 手段，不是最终 90+ 方案。

#### PCT refiner 方向

当前 PCT refiner 方向合理，但还没有充分发挥作用，主要原因：

- 训练目标仍然偏 pointwise displacement；
- 没有直接优化 CD / point distribution preservation；
- 低噪声官方分布没有成为训练主分布；
- gate 还只是辅助控制，没有形成核心的 per-point 位移强度预测；
- refiner 依赖已有 teacher/base，若 base 位移方向偏强，refiner 容易继承错误。

---

## 3. 文献方法带来的设计原则

### 3.1 PointCleanNet

PointCleanNet 的核心是：先判断 outlier，再估计 correction vector，把 noisy point 投回 clean surface。

对本任务的启发：

- 可以继续采用“预测点位移”的框架；
- 但不能直接删除点，因为赛题要求点数和 shape 不变；
- correction vector 必须带置信度或强度控制，否则会破坏 CD。

### 3.2 Score-Based Point Cloud Denoising

Score-based 方法把点云去噪看作沿 score gradient 提升数据似然，即多步把点推向高概率曲面区域。

对本任务的启发：

- 多噪声强度建模是必要的；
- 测试时步长必须自适应；
- 对官方低噪声数据，不能默认使用强步长或多步迭代。

### 3.3 PCDNF / Joint Normal Filtering

PCDNF 强调 normal filtering 与 point denoising 的联合建模。

对本任务的启发：

- 应显式区分法向噪声和切向漂移；
- 法向修正有利于 P2S；
- 切向漂移容易破坏 CD，应被惩罚或抑制；
- normal / curvature / planarity 应作为 gate 输入或辅助监督。

### 3.4 IterativePFN

IterativePFN 把迭代过滤过程放进网络内部，而不是只在测试时重复调用单步网络。

对本任务的启发：

- 对高噪声场景，内部迭代有价值；
- 对官方低噪声场景，迭代必须由 gate 控制；
- 盲目多步迭代会加重过处理。

### 3.5 StraightPCF

StraightPCF 使用 VelocityModule 预测直线路径方向，并使用 DistanceModule 估计轨迹长度。

对本任务的启发最直接：

> 当前系统最大问题不是没有方向，而是缺少可靠的“走多远”控制。

因此下一版模型应从“直接输出 3D displacement”改为：

```text
direction head + step/distance head + confidence/gate head
```

---

## 4. 总体技术路线

推荐主线：

```text
官方低噪声验证集校准
    ↓
LowNoise Adaptive Refiner
    ↓
metric-aligned loss: point + patch CD + surface + tangent + repulsion
    ↓
弱后处理，不再强投影
    ↓
按样本/点级 gate 做自适应 ensemble
    ↓
官方提交
```

核心思想：

1. 用训练 mesh 构造与官方反馈一致的 validation。
2. 使用已有 baseline 作为候选输入，而不是完全废弃当前工作。
3. 新模型不直接预测自由 displacement，而是预测：
   - direction；
   - step/distance；
   - gate/confidence。
4. loss 直接对齐官方 CD/P2S，而不是只做 MSE/Huber。
5. 后处理只作为弱约束，避免继续破坏 CD。

---

## 5. 阶段一：构造 official-like validation

### 5.1 目标

构造一个本地验证集，使 noisy baseline 的统计量接近官方反馈：

```text
mean_CD_noisy  ~= 0.000246
mean_P2S_noisy ~= 0.000196
```

这一步优先级最高。没有可信验证集，后续所有训练和后处理都会被误导。

### 5.2 实施方式

基于已有 `scripts/prepare_local_eval.py` 扩展：

```bash
python scripts/prepare_local_eval.py \
  --mesh_dir ./dataset_train \
  --datalist ./datalist/validate.txt \
  --output_dir ./local_eval_official_low \
  --num_points 50000 \
  --limit 200 \
  --seed 2026 \
  --sigmas 0.0015,0.0020,0.0025,0.0030,0.0035 \
  --noise_types gaussian,anisotropic \
  --outlier_p 0.000
```

然后运行：

```bash
python evaluate.py \
  --pred_dir ./local_eval_official_low/noisy_as_pred \
  --gt_dir ./local_eval_official_low/gt \
  --noisy_dir ./local_eval_official_low/noisy \
  --mesh_dir ./dataset_train \
  --list ./local_eval_official_low/test.txt \
  --workers 8
```

需要额外写一个脚本 `scripts/make_noisy_as_pred.py`，把：

```text
noisy.npy -> denoised.npy
```

用于计算 noisy baseline。

### 5.3 搜索 noise schedule

建议搜索参数：

| 参数 | 搜索范围 |
|---|---|
| gaussian sigma | `0.0010 ~ 0.0040` |
| anisotropic scale | `0.5 ~ 1.8` |
| laplace 比例 | `0% ~ 30%` |
| outlier_p | `0 ~ 0.001` |
| clean sampling seed | 至少 3 个 seed |

保留三套验证集：

| 名称 | 用途 |
|---|---|
| `local_eval_official_low` | 主验证集，匹配官方低噪声分布 |
| `local_eval_official_low_shifted` | 轻微偏移，检查泛化 |
| `local_eval_medium` | 防止模型只适配极低噪声 |

### 5.4 验证集通过标准

只有当 noisy baseline 接近官方反馈时，后续实验才可信：

```text
abs(mean_CD_noisy - 0.000246) / 0.000246 < 20%
abs(mean_P2S_noisy - 0.000196) / 0.000196 < 20%
```

---

## 6. 阶段二：实现 LowNoise Adaptive Refiner

### 6.1 模型输入

每个 patch 输入：

```text
x_noisy        : noisy point coordinates
x_base         : base prediction, e.g. current alpha-best output
base_disp      : x_base - x_noisy
local_geom     : curvature, planarity, anisotropy, knn_radius
sigma_hat      : estimated local/global noise level
normal_hat     : local PCA normal
```

拼接后输入 encoder：

```text
[x_noisy, x_base, base_disp, local_geom, sigma_hat]
```

建议输入维度：

```text
3 + 3 + 3 + 4 + 1 = 14
```

### 6.2 模型输出

不要直接输出自由三维向量。改为三头输出：

```text
dir_raw_i   = HeadDir(feat_i)       # R^3
step_raw_i  = HeadStep(feat_i)      # R^1
gate_raw_i  = HeadGate(feat_i)      # R^1
```

变换为：

```python
dir_i  = normalize(dir_raw_i)
step_i = softplus(step_raw_i)
gate_i = sigmoid(gate_raw_i)
step_i = clamp(step_i, 0, max_step_scale * local_knn_radius_i)
disp_i = gate_i * step_i * dir_i
y_i    = x_noisy_i + disp_i
```

这样做的好处：

- direction 学方向；
- step 学强度；
- gate 学是否应该移动；
- clamp 防止低噪声场景下过处理。

### 6.3 Encoder 选择

推荐优先级：

1. `PCTNeighborFeatureExtraction`
2. `EdgePCTHybridFeatureExtraction`
3. 当前 `FeatureExtraction / EdgeConv` baseline

原因：

- PCTNeighbor 已经在你仓库中实现，改动成本低；
- neighbor attention 更适合 patch-level 局部结构；
- position embedding 对几何任务有帮助；
- EdgePCTHybrid 适合作为第二候选，保留 EdgeConv 局部归纳偏置。

### 6.4 文件改动建议

新建：

```text
starter_code/src/model/low_noise_refiner.py
starter_code/configs/model/low_noise_refiner_pct_neighbor.yaml
starter_code/configs/task/train_low_noise_refiner.yaml
starter_code/configs/task/predict_low_noise_refiner.yaml
starter_code/scripts/prepare_official_like_eval.py
starter_code/scripts/make_noisy_as_pred.py
starter_code/scripts/grid_noise_schedule.py
starter_code/scripts/evaluate_candidate_table.py
```

不要一开始大改 `VelocityModule`。建议先实现独立模型，减少和已有实验代码耦合。

---

## 7. 阶段三：训练目标设计

### 7.1 总损失

推荐：

```text
L = L_point
  + λ_cd      * L_patch_cd
  + λ_surface * L_surface
  + λ_tangent * L_tangent
  + λ_gate    * L_gate
  + λ_rep     * L_repulsion
  + λ_anchor  * L_anchor
```

### 7.2 `L_point`

合成数据时 noisy 与 clean 存在对应关系，使用：

```text
L_point = Huber(y_i - clean_i)
```

建议：

```text
huber_delta = 0.002 ~ 0.005
```

### 7.3 `L_patch_cd`

直接对齐官方 CD：

```text
L_patch_cd = mean_nn_dist_sq(y_patch, clean_patch)
           + mean_nn_dist_sq(clean_patch, y_patch)
```

实现时每个 patch 采样 512~2048 点即可，避免 50k 全量训练开销过大。

### 7.4 `L_surface`

用训练 mesh 上采样 dense surface points 近似 P2S：

```text
L_surface = mean_nn_dist_sq(y_i, surface_samples)
```

注意：

- 不建议权重过高；
- 否则会再次出现 P2S 高、CD 低的问题。

### 7.5 `L_tangent`

用 PCA normal `n_i` 分解位移：

```text
disp = y_i - x_noisy_i
normal_component  = dot(disp, n_i) * n_i
tangent_component = disp - normal_component
L_tangent = mean(||tangent_component||^2)
```

目的：

- 保留主要法向去噪；
- 抑制无意义切向漂移；
- 保护 CD 和局部采样分布。

### 7.6 `L_gate`

训练 gate 预测“移动是否带来改善”。定义目标：

```text
improve_i = ||x_noisy_i - clean_i|| - ||y_teacher_i - clean_i||
gate_target = 1 if improve_i > margin else 0
```

或者使用连续目标：

```text
gate_target = clamp(improve_i / tau, 0, 1)
```

作用：

- 低噪声点不要乱动；
- teacher 方向不可靠时降低 gate；
- 对官方低噪声数据尤其关键。

### 7.7 `L_repulsion`

防止点聚集：

```text
L_repulsion = mean(max(0, h - dist_to_knn)^2)
```

权重不要太大，只作为覆盖保持项。

### 7.8 `L_anchor`

约束输出不要离 noisy 太远：

```text
L_anchor = mean(max(0, ||y_i - x_noisy_i|| - max_step)^2)
```

其中：

```text
max_step = c * local_knn_radius
c = 0.3 ~ 1.0
```

---

## 8. 阶段四：训练计划

### 8.1 数据分布

训练 batch 中建议混合：

| 数据类型 | 比例 | 作用 |
|---|---:|---|
| official-like low noise | 70% | 对齐官方测试集 |
| shifted low noise | 20% | 防止过拟合单一 sigma |
| medium noise | 10% | 保留鲁棒性 |

不要再以高噪声为主。

### 8.2 训练阶段

#### Stage A：train heads only

冻结部分 encoder 或使用较小 LR，先让 step/gate 稳定：

```text
epochs: 3~5
lr: 1e-4
loss: L_point + L_anchor + L_tangent
```

#### Stage B：joint train

完整训练：

```text
epochs: 20~40
lr: 5e-5 ~ 1e-4
loss: full loss
```

#### Stage C：low-noise fine-tune

只在 official-like low noise 上微调：

```text
epochs: 5~10
lr: 1e-5 ~ 3e-5
increase λ_cd, λ_tangent
reduce λ_surface
```

目的：提升 CD，不让 P2S 导致过投影。

### 8.3 初始 loss 权重

建议起点：

```yaml
loss_point_weight:   1.0
loss_cd_weight:      0.5
loss_surface_weight: 0.15
loss_tangent_weight: 0.2
loss_gate_weight:    0.05
loss_rep_weight:     0.02
loss_anchor_weight:  0.1
```

若 CD 低、P2S 高：

```text
increase loss_cd_weight, loss_tangent_weight, loss_anchor_weight
reduce loss_surface_weight, postprocess_beta
```

若 CD 高、P2S 低：

```text
increase loss_surface_weight, max_step_scale, gate positive prior
```

---

## 9. 阶段五：推理与后处理

### 9.1 推理候选

保留多个候选目录：

```text
candidate_A_noisy
candidate_B_current_alpha_best
candidate_C_low_noise_refiner
candidate_D_low_noise_refiner_weak_projection
candidate_E_ensemble_adaptive
```

### 9.2 弱后处理

后处理只允许弱投影：

```text
project_method = edge_tangent
project_k      = 32 / 48 / 64
project_beta   = 0.05 / 0.10 / 0.15 / 0.20
edge_low       = 0.18
edge_high      = 0.55
clamp_scale    = 0.75 / 1.0
```

禁止再把强 projection 作为主线，除非 official-like validation 上 CD/P2S 同时提升。

### 9.3 自适应融合

按点融合：

```python
y = noisy + alpha_i * (refiner_pred - noisy)
```

其中 `alpha_i` 来自：

```text
gate_i * geometry_confidence_i * noise_confidence_i
```

geometry confidence 示例：

```text
flat/confident plane:   higher alpha
edge/high curvature:    lower alpha
thin structure:         lower alpha
large base displacement: lower alpha
```

按样本融合：

- flat/simple 类别：允许略强去噪；
- thin/edge-rich 类别：使用更保守 alpha；
- 如果某个样本估计噪声低于官方均值，优先保守。

---

## 10. 阶段六：实验矩阵

### 10.1 先做最小实验

| 实验 | 内容 | 预期 |
|---|---|---|
| E0 | 当前 alpha=0.65 / 0.75 / 0.85 在 official-like val 上评估 | 确定短期最佳 alpha |
| E1 | low-noise refiner，无 CD loss | 检查模型能否收敛 |
| E2 | E1 + patch CD loss | CD 应明显提升 |
| E3 | E2 + tangent loss | CD 继续提升，P2S 小幅下降或持平 |
| E4 | E3 + surface loss | P2S 提升，但观察 CD 是否受损 |
| E5 | E4 + weak projection | 只接受 CD/P2S 同时提升的配置 |

### 10.2 提交前本地门槛

如果 official-like validation 未达到：

```text
CD_score  >= 85
P2S_score >= 88
final     >= 86.5
```

不建议提交官方 probe，除非只是为了获得曲线锚点。

若要冲击 90，建议本地至少达到：

```text
CD_score  >= 88
P2S_score >= 92
final     >= 90
```

---

## 11. 具体代码任务清单

### 11.1 数据和验证脚本

必须实现：

```text
scripts/make_noisy_as_pred.py
scripts/grid_noise_schedule.py
scripts/evaluate_candidate_table.py
```

`make_noisy_as_pred.py` 功能：

```text
local_eval/noisy/**/noisy.npy
  -> local_eval/noisy_as_pred/**/denoised.npy
```

`grid_noise_schedule.py` 功能：

- 自动调用 `prepare_local_eval.py`；
- 自动构造 noisy-as-pred；
- 自动运行 `evaluate.py`；
- 输出每组 schedule 的 `mean_CD_noisy`、`mean_P2S_noisy`。

`evaluate_candidate_table.py` 功能：

- 批量评估多个 candidate dirs；
- 输出 CSV / Markdown 表格；
- 记录 CD_score、P2S_score、final、mean displacement。

### 11.2 模型文件

新建：

```text
src/model/low_noise_refiner.py
```

建议类：

```python
class LowNoiseAdaptiveRefiner(nn.Module):
    def __init__(self, cfg):
        ...

    def execute(self, batch):
        # returns y, disp, gate, step, direction
        ...
```

内部模块：

```text
encoder
head_dir
head_step
head_gate
loss functions
patch inference aggregator
```

### 11.3 配置文件

新建：

```text
configs/model/low_noise_refiner_pct_neighbor.yaml
configs/task/train_low_noise_refiner.yaml
configs/task/predict_low_noise_refiner.yaml
```

初始配置示例：

```yaml
__target__: LowNoiseAdaptiveRefiner
encoder_type: pct_neighbor
frame_knn: 24
num_train_points: 768
input_dim: 14
feat_embedding_dim: 384
pct_attention_channels: 96
pct_num_attention_layers: 4
pct_use_position_embedding: true
use_global_feature: true
use_input_coords: true
max_step_scale: 0.75
huber_delta: 0.003
loss_point_weight: 1.0
loss_cd_weight: 0.5
loss_surface_weight: 0.15
loss_tangent_weight: 0.2
loss_gate_weight: 0.05
loss_rep_weight: 0.02
loss_anchor_weight: 0.1
predict_patch_size: 1600
predict_patch_aggregation: disp_weighted
```

---

## 12. 决策规则

### 12.1 什么时候继续训练

满足以下条件时继续：

- official-like validation 上 CD 和 P2S 至少一个明显提升，另一个不明显下降；
- CD_score 高于当前 alpha baseline；
- mean displacement 不显著超过当前 alpha=0.85；
- 可视化没有明显塌缩、边缘糊掉、薄结构断裂。

### 12.2 什么时候回滚

出现以下情况时回滚：

- P2S 提升但 CD 大幅下降；
- 模型输出 mean displacement 过大；
- gate 大面积接近 1，失去选择能力；
- weak projection 让 CD 下降；
- validation 只在单一 seed 上有效。

### 12.3 官方提交策略

官方提交应该服务于验证假设，不要随机提交：

1. alpha 曲线尚不清楚时，提交 `alpha=0.65/0.75` 这类锚点。
2. 新 refiner 本地超过当前 alpha baseline 后，再提交一次。
3. 若官方反馈 CD 仍低，继续减小 step/gate，而不是增强 projection。
4. 若官方反馈 P2S 低、CD 高，增加 surface loss 或弱 projection。

---

## 13. 两周实施时间表

### Day 1-2：验证集修正

- 实现 noisy-as-pred；
- 实现 noise schedule grid；
- 得到 official-like validation；
- 对当前已有候选重新评估。

交付物：

```text
local_eval_official_low/
noise_schedule_results.md
candidate_baseline_table.md
```

### Day 3-5：LowNoiseAdaptiveRefiner 最小版

- 新建模型文件；
- 实现 direction / step / gate heads；
- 跑通 train / predict；
- 只用 `L_point + L_anchor + L_tangent`。

交付物：

```text
checkpoint_minimal.pkl
results_low_noise_refiner_minimal/
```

### Day 6-8：加入 metric-aligned loss

- 加 `L_patch_cd`；
- 加 `L_surface`；
- 加 `L_repulsion`；
- 对比 E1~E4。

交付物：

```text
ablation_low_noise_refiner.md
```

### Day 9-10：推理与弱后处理

- patch inference 聚合；
- weak edge projection；
- 自适应 alpha/gate 融合。

交付物：

```text
candidate_C_low_noise_refiner/
candidate_D_weak_projection/
candidate_E_adaptive_ensemble/
```

### Day 11-12：消融与可视化

- 按类别统计；
- 按几何复杂度统计；
- 检查边缘、薄结构、平面结构；
- 选择一个官方提交候选。

### Day 13-14：官方提交与反馈分析

- 提交最佳候选；
- 依据官方 CD/P2S 反馈调整：
  - CD 低：减 step，增 CD/tangent/anchor；
  - P2S 低：增 surface，略增 gate；
  - 两者都低：方向/验证集仍有问题。

---

## 14. 最终优先级排序

最重要的 5 件事：

1. **重建 official-like validation**。
2. **不再用高噪声 local eval 指导主线**。
3. **实现 direction + step + gate refiner**。
4. **加入 patch CD 和 tangent loss**。
5. **把后处理从主方法降级为弱约束**。

短期最可能涨分的是：

```text
官方低噪声验证 + per-sample/per-point adaptive alpha
```

长期最可能突破 90 的是：

```text
LowNoiseAdaptiveRefiner + metric-aligned training + weak geometry prior
```

---

## 15. 参考论文与对应启发

| 方法 | 核心思想 | 在本任务中的用法 |
|---|---|---|
| PointCleanNet | outlier detection + correction vector | 保留 correction vector 思路，但不能删点 |
| Score-Based Point Cloud Denoising | 估计 score，多步推向高概率曲面 | 引入 noise condition 和自适应步长，不盲目多步 |
| PCDNF | point denoising + normal filtering | 显式加入 normal / tangent 约束 |
| IterativePFN | 网络内部建模迭代过滤 | 只在 gate 控制下使用迭代思想 |
| StraightPCF | velocity direction + distance scalar | 采用 direction + step/distance + gate 三头设计 |

---

## 16. 一句话版本

当前 60 分的主要矛盾是：**模型能贴近曲面，但不懂得少动、按点动、沿法向动，因此 CD 被破坏**。下一步不应继续强后处理，而应先校准官方低噪声验证集，再训练一个带 direction / step / gate 的低噪声 refiner，并用 patch CD、surface、tangent、repulsion 等 loss 直接对齐官方指标。
