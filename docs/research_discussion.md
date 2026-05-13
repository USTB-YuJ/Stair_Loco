# 上下楼场景落脚点规划研究方向讨论

## 核心思路

面向上下楼场景，输入深度图，分割每级台阶平面，引导机器人落脚点靠近台阶横向中心线，降低打滑/绊倒风险。目标机器人：G1。

---

## 数据流架构

```
深度图 buffer [B, T, H, W]
        │
        CNN backbone（逐帧，共享权重）
        ├─ conv_feat [B*T, 64, 5, 5]  ──→ SegDecoder → pred_mask [B, T, H, W]
        │                                   ↑ 辅助监督（训练时，所有T帧）
        └─ flatten → fc → [B*T, 128]
                    reshape → [B, T, 128]
                    │
                   GRU → depth_feature (128) → Actor
```

**关键设计决策：**
- 把编码器**特征**输入 Actor，而非分割结果
- 分割头接 CNN **中间特征图**（conv_feat [64, 5, 5]），保留空间信息
- 分割监督覆盖**所有 T 帧**，避免梯度被稀释（只监督最新帧时，梯度影响仅 1/T）
- 训练时辅助监督，推理时 SegDecoder 可不运行

---

## 台阶中心线 Heatmap（第二阶段）

用中心**线**替代中心**点**，原因：台阶横向自由度大，真正危险的是纵向太靠近边缘。

未来扩展：把台阶中心线 heatmap 作为第二输入通道和深度图一起送进 CNN，实现观测与奖励对齐。

---

## 落脚点奖励设计（第二阶段）

只在脚接触地面瞬间触发（利用 `contact_filt`），只惩罚**纵向**偏差：

```python
r_footstep = exp(-d_longitudinal² / σ²)   # σ ≈ 0.1m
```

---

## 标注生成方案

仿真中通过**深度图反投影 + 地形高度图查询**自动生成 per-pixel 台阶踏面标注：

1. 深度图反投影到世界坐标（利用相机内参 + 相机位姿）
2. 查询地形高度图（`height_samples`），计算点到地面的高度差
3. 计算地形法向量 z 分量（有限差分）
4. 判断条件：`height_diff < 0.06m` 且 `normal_z > 0.85` → 台阶踏面

优点：标注来自地形 GT，不受深度噪声影响，与深度图天然对齐。

---

## 训练流程（三阶段）

| 阶段 | 内容 | 目的 |
|------|------|------|
| 第一阶段（已实现） | CNN+GRU+SegDecoder，GT mask 辅助监督 | 让 CNN 学到台阶感知特征 |
| 第二阶段 | 加入落脚点奖励（GT 台阶中心线） | 验证控制侧可行性 |
| 第三阶段 | heatmap 双通道输入，感知-控制对齐 | 端到端性能验证 |

---

## 已实现的代码改动（第一阶段）

| 文件 | 改动 |
|------|------|
| `legged_gym/envs/base/legged_robot.py` | `warp_update_depth_buffer` 同步生成 gt_mask；新增 `_generate_stair_mask`、`_query_height_at_points`、`_query_normal_z_at_points` |
| `rsl_rl/modules/actor_critic_depth.py` | backbone 拆分暴露 conv_feat；新增 `SegDecoder`；`StackDepthEncoderGRU` 返回 `(depth_feature, pred_masks)` |
| `rsl_rl/storage/rollout_storage_extra.py` | 新增 `gt_mask` buffer |
| `rsl_rl/algorithms/amp_ppo_multi.py` | 加入 seg loss（BCE，权重 `seg_loss_coef=0.1`） |
| `rsl_rl/runners/amp_on_policy_runner_multi.py` | 传递 `warp_gt_mask_buffer` 到 transition |

---

## 论文定位（目标：CoRL）

**主 claim**：量化证明落脚点靠近台阶中心线后，打滑/绊倒概率的下降幅度（安全性量化）。

**技术贡献**：辅助分割监督提升 CNN 特征质量，体现在策略性能上（表示学习）。

**可强化方向：**
- 泛化性：扩展到任意结构化地面（台阶、石墩、窄梁），heatmap 作为统一接口
- 真机实验（G1 上下楼）：CoRL 必要条件

**与现有工作的区分点：**
区别于 ETH ANYmal 系列、MIT Parkour、Berkeley DreamWaQ 等视觉引导落脚点工作，核心差异在于辅助分割监督机制 + 安全性量化。

---

## 待确认事项

- [ ] 真机实验条件（G1 硬件可用性）
- [ ] 第一阶段验证：seg_loss 收敛情况，gt_mask 可视化质量
- [ ] 第二阶段：落脚点奖励的 `_get_dist_to_stair_centerline` 实现
- [ ] 与哪些 baseline 对比
