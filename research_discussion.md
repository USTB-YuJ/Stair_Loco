# 上下楼场景安全热力图预测研究方向讨论

## 核心思路

面向上下楼场景，输入深度图，**预测连续安全热力图**，引导机器人落脚点避开台阶边缘等危险区域，降低打滑/绊倒风险。目标机器人：G1。

**动机与痛点**：机器人在上下楼梯阶段猜不准台阶会摔，我们希望它**一步一步稳稳地走**——每一步都踩在安全区域上。

核心设计原则：
- **表示学习**：SegDecoder 从 CNN 中间特征预测 heatmap（辅助监督，MSE），梯度不回流到 Actor
- **预计算安全地图**：地形安全值离线计算为 `terrain_safety_map` 网格，GT heatmap 通过反投影查表生成
- **单次渲染**：深度图含本体遮挡，通过 `height_diff < 0.15m` 自然过滤身体像素

---

## 系统架构

```
预计算（一次性）：
  高度场 → nz × 0.5 + flatness × 0.5 → terrain_safety_map [rows, cols]

每帧：
  Warp 渲染（含 22 个身体 link mesh）→ depth_images [B, 64, 64] 米制
    │
    ├─→ 加噪 → warp_depth_buffer → CNN → GRU → depth_feature → Actor
    │
    └─→ 反投影查表 → GT safety_heatmap [B, 64, 64]
              ↑
         height_diff < 0.15m 过滤身体像素 + 竖直面
              │
              └─→ SegDecoder → pred_heatmap → seg_loss (MSE)

落脚点奖励：
  foot_world_xy → _get_foot_safety → terrain_safety_map 查表 → r_foot_safety
```

---

## 地形安全地图（terrain_safety_map）

### 数据结构

- **形状**：`[tot_rows, tot_cols]`，与 `height_samples` 同构
- **类型**：float32，范围 [0, 1]
- **分辨率**：每格 0.05m × 0.05m
- **索引约定**：`[col, row]`（即 `[px, py]`），与代码库中所有 terrain 查询一致

### 计算

```python
# 法向量 z 分量（中心差分）
nz = 1.0 / sqrt(1 + dz_dx² + dz_dy²)

# 局部平坦度（3×3 窗口方差，σ²=0.0004）
flatness = exp(-local_var / 0.02²)

# 融合（去掉 edge_score）
terrain_safety_map = 0.5 * nz + 0.5 * flatness
```

### 为什么去掉 edge_score

- 法向量已编码斜面危险程度，平坦度已编码边缘粗糙度
- edge_score 依赖深度分辨率 + 噪声，引入不必要的复杂度
- 预计算地图不受深度噪声影响，不受体遮挡影响

### 查询方式

```python
# 世界坐标 → 网格坐标 → 安全值
px = (world_x + border) / horizontal_scale
py = (world_y + border) / horizontal_scale
safety = terrain_safety_map[px, py]  # O(1) 查表
```

---

## GT 热力图生成（反投影查表）

### 流程

```
depth_images (体遮挡深度, 米制)
  → 反投影到世界坐标 (复用 warp 相机内参 + 位姿)
  → 查 terrain_safety_map[px, py] → 逐像素安全值
  → 计算 height_diff = |pts_world_z - terrain_h|
  → valid = (depth > 0) & (height_diff < 0.15)
  → safety_heatmap *= valid  (身体像素/竖直面 → 0)
  → crop + resize → warp_safety_heatmap_buffer
```

### height_diff 过滤机制

| 像素类型 | height_diff | 阈值 0.15 | 结果 |
|---------|------------|----------|------|
| 台阶踏面 | ≈ 0 | ✅ | 正常 safety 值 |
| 台阶 riser（竖直面） | 0.15-0.2m | ❌ | safety=0（正确） |
| 身体像素（腿/手臂） | 0.2-0.5m | ❌ | safety=0（正确） |
| 远处地形 | ≈ 0 | ✅ | 正常 safety 值 |

**单次渲染 + height_diff 自然过滤** = 无需 terrain-only 渲染，无 OOM，无额外标签。

---

## SegDecoder 辅助监督

### 网络结构

```python
class SegDecoder(nn.Module):
    # 输入: conv_feat [B, 64, 5, 5]（CNN backbone 中间特征）
    # ConvTranspose2d(64→32) → (32→16) → (16→1) → Upsample → Sigmoid
    # 输出: [B, H, W] ∈ [0, 1]
```

### 损失函数

```python
seg_loss = MSE(pred_heatmap, gt_safety_heatmap, reduction='none')
seg_loss = (seg_loss * body_mask).sum() / body_mask.sum().clamp(min=1)
loss += 0.1 * seg_loss
```

- **body_mask**：1=地形像素，0=身体像素（暂时全 1，身体像素因 height_diff 过滤已 safety=0）
- **梯度隔离**：seg_loss 只影响 conv_feat，不回流到 Actor

### 监控

- **Tensorboard**：`Loss/seg` 曲线
- **控制台**：`Seg loss: 0.001234`
- **play.py 可视化**：GT vs Pred vs |误差| 三列对比

---

## 本体遮挡渲染

### 实现

- **渲染器**：`warp_render_v3.py` 扩展 `load_body_meshes` + `_build_body_mesh`
- **覆盖**：22 个 G1 body link（左右 shoulder_pitch + elbow + hip/knee/ankle + pelvis）
- **原理**：每帧取 `rigid_body_states` 中 link 位姿 → 变换 STL 顶点 → 合并 warp mesh → 双 raycast（地形+身体 → min）

### 坐标约定

- rigid_body_states 四元数格式：XYZW → 转换为 WXYZ 供 `quat_apply`
- body mesh 顶点变换：`warp2gym^T @ (quat_apply(link_quat, v_local) + link_pos)`

---

## 落脚点安全奖励

### 实现

```python
def _reward_foot_safety(self):
    contact_moment = self.contact_filt & ~self.was_contact  # 触地边沿
    feet_xy = rigid_body_states[:, feet_indices, :2]         # 世界坐标
    safety = self._get_foot_safety(feet_xy).reshape(B, 2)
    self.was_contact = contact_now
    return (safety * contact_moment.float()).sum(dim=-1)


def _get_foot_safety(self, foot_xy):
    px = (foot_xy[:, 0] + border) / hs
    py = (foot_xy[:, 1] + border) / hs
    return self.terrain_safety_map[px, py]  # O(1) 查表
```

### 设计要点

- **触地瞬间奖励**：稀疏但精确，信用分配对准"落脚决策"
- **与 GT 同源**：落脚点和 heatmap GT 查同一张 `terrain_safety_map` → 观测-奖励语义对齐
- **权重**：`foot_safety = 1.0`

---

## 可视化

### 3D 地形热力覆盖层

- 方法：`_draw_terrain_safety_overlay()` → `gym.add_lines` 在地形表面画彩色十字
- 红色 = 危险 (safety→0)，绿色 = 安全 (safety→1)
- 采样间距：`safety_map_sample_spacing = 0.2m`（可在 config 调整）
- 开关：G1 config 中 `terrain.visualize_safety_map = True`

### play.py 预测对比

- Pred Safety vs GT Safety 并排显示
- 预测头输出通过 `actor_critic._last_pred_masks` 获取

---

## G1 深度配置

| 参数 | 值 |
|------|-----|
| 相机位姿 | [0.0576, 0.0175, 0.4299] m |
| FOV | 79.3°（与 H1 同款相机） |
| 渲染分辨率 | 64×64 |
| 输入尺寸 | 64×64 |
| near/far | 0 / 2 m |
| buffer_len | 3 |
| 体遮挡 | 22 links（肩膀/手肘/髋/膝/踝/pelvis） |
| 噪声 | 全部关闭（算法验证阶段） |

---

## 已实现代码改动

| 文件 | 改动 |
|------|------|
| `legged_gym/envs/base/legged_robot.py` | `terrain_safety_map` 预计算；GT heatmap 反投影查表 + height_diff 过滤；`_get_foot_safety` 单点查表；`_draw_terrain_safety_overlay` 3D 可视化；`warp_body_mask_buffer` |
| `legged_gym/utils/warp_render_v3.py` | `load_body_meshes` + `_build_body_mesh`；`render_depth` 支持 `link_pos_w/link_quat_w` |
| `rsl_rl/rsl_rl/modules/actor_critic_depth.py` | `SegDecoder` + `StackDepthEncoderGRU`；`act_inference` 存储 `_last_pred_masks` |
| `rsl_rl/rsl_rl/storage/rollout_storage_extra.py` | `gt_safety_heatmap` + `gt_body_mask` buffer |
| `rsl_rl/rsl_rl/algorithms/amp_ppo_multi.py` | seg_loss (masked MSE)；`last_seg_loss` 存储 |
| `rsl_rl/rsl_rl/runners/amp_on_policy_runner_multi.py` | 传递 heatmap + body_mask；tensorboard + 控制台 seg_loss 日志 |
| `legged_gym/scripts/play.py` | Pred vs GT heatmap 可视化；相机角度对齐 |
| `g1_loco/g1_16dof_loco_env.py` | `_reward_foot_safety` + `was_contact` buffer |
| `g1_loco/g1_16dof_loco_config.py` | `foot_safety` 奖励权重；`visualize_safety_map` + spacing |

---

## 论文定位

**主 claim**：量化证明通过安全热力图引导落脚点后，机器人打滑/绊倒概率下降。

**技术贡献**：
1. 预计算地形安全地图 + 反投影查表（替代逐像素三分量查询）
2. 辅助监督 SegDecoder + 本体遮挡渲染 + height_diff 自然过滤
3. 触地瞬间安全奖励 + 同一安全地图的观测-奖励对齐

---

## 待确认事项

- [ ] G1 Phase 1 训练：seg_loss 收敛、GT heatmap 可视化质量
- [ ] 评估 Tiny U-Net / 多尺度 skip SegDecoder：注意强 skip 可能让 decoder 依赖浅层边缘旁路，导致 actor 使用的 `depth_feature` 未真正编码 safety；后续可考虑加入 bottleneck safety head 或低分辨率辅助监督，强制 actor 可见特征学习安全语义
- [ ] body_mask 恢复（基于深度不连续检测，当前全 1）
- [ ] 与 baseline 对比方案
- [ ] 真机实验（G1 硬件）
- [ ] buffer_len 3 → 5
