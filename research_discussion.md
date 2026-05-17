# 上下楼场景安全可踩性与踏面交互研究方向讨论

## 当前主线

面向 G1 上下楼场景，当前方法不再以 dense `SegDecoder` 热力图预测作为主线，而是围绕两个互补问题展开：

```text
1. 空间可踩性：从深度和本体状态中理解脚下哪些局部区域安全。
2. 台阶交互顺序：让训练过程产生沿路径、逐级处理楼梯的经验。
```

当前主线组件：

```text
terrain_safety_map
  -> 作为训练期 privileged label/reward source

Foot-Centric Safety Affordance Head
  -> 从 obs + history + depth 预测左右脚候选区域安全性
  -> 输出左右脚 affordance token 拼入 actor

foot_safety penalty-only reward
  -> touchdown 时用脚底面积评价落脚安全
  -> 只惩罚不安全落脚，不奖励安全落脚

Route-conditioned tread interaction（规划中）
  -> 路径线保证 stair episode 与楼梯充分交互
  -> tread progression / event-triggered landing region 管理逐级台阶交互顺序
```

Actor observation 和部署接口保持不变：

```text
actor input interface: obs + history + depth
num_observations: 57
不向 actor 输入 next target / current tread index / privileged terrain state
```

---

## 为什么不再以 SegDecoder 作为主线

早期方案是从 CNN 中间特征预测完整 `64x64` safety heatmap，并用 dense MSE 做辅助监督。这个方向现在降级为 ablation/debug，原因是：

- `SegDecoder` 是旁路监督，actor 不一定真正使用它学到的安全先验。
- 从 `[B, 64, 5, 5]` 深层特征恢复到 `64x64`，空间分辨率太粗，对台阶边缘等精细结构天然模糊。
- 存储完整 heatmap batch 增加显存和训练开销。
- Dense heatmap 监督容易把问题讲成 segmentation，而不是 locomotion 中的可部署安全落脚决策。

当前配置上也已经与这个判断对齐：

```python
algorithm.seg_loss_coef = 0.0
policy.enable_foot_affordance = True
```

保留 `SegDecoder` 的作用：

```text
1. debug 可视化：play 中对比 GT safety heatmap 和 predicted heatmap。
2. ablation：验证 dense auxiliary supervision 是否真的有帮助。
3. 不是当前主方法，不作为论文故事的核心贡献。
```

---

## Terrain Safety Map

`terrain_safety_map` 仍然是当前方法的基础训练信号。它不是 actor 的输入，而是用于生成 compact affordance label、落脚安全惩罚和 debug 可视化。

### 数据结构

- 形状：`[tot_rows, tot_cols]`，与 `height_samples` 同构。
- 类型：float32，范围 `[0, 1]`。
- 分辨率：每格 `0.05m x 0.05m`。
- 索引约定：`[px, py]`，与代码库 terrain 查询保持一致。

### 计算逻辑

```python
nz = 1.0 / sqrt(1 + dz_dx^2 + dz_dy^2)
flatness = exp(-local_var / 0.02^2)
terrain_safety_map = 0.5 * nz + 0.5 * flatness
```

设计理解：

- `normal_z` 编码斜面、竖直面、台阶 riser 等不可踩区域。
- `flatness` 编码局部粗糙度和边缘邻域风险。
- 当前主线不再单独监督 `edge_risk`，因为 safety 本身已经包含边缘/粗糙度风险。

---

## Foot-Centric Safety Affordance Head

### 目的

`FootAffordanceHead` 的目标不是定义真实落脚点，而是学习一个可部署的局部可踩性表示：

```text
如果左/右脚要落到 body-frame 中的这些候选脚底区域，哪些更安全？
```

它回答的是空间问题：

```text
where is safe to step?
```

它不回答时序问题：

```text
当前哪只脚应该摆动？
下一次落脚应该发生在哪一级台阶？
是否已经按顺序完成台阶交互？
```

因此 affordance 与后续 tread progression 的关系是：

```text
affordance = 空间可踩性先验
tread progression = 时序台阶交互先验
```

### 网络结构

候选点固定在 body frame 中，每只脚 `K=15` 个：

```text
x = [0.10, 0.25, 0.40, 0.55, 0.70] m
y = ±0.11 + [-0.08, 0.0, 0.08] m
```

输入：

```text
depth_feature[128]
body_feature[64] = MLP(obs57 + history_latent64)
candidate_geometry = (dx, dy, side)
```

输出：

```text
pred_affordance[B, 2, 15, 3]
  channel 0: safety
  channel 1: edge_risk   # 当前主线权重为 0
  channel 2: targetness  # 当前主线权重为 0

left_token[32], right_token[32]
  -> concat 到 actor input
```

当前 actor 输入维度：

```text
obs57 + history64 + depth128 + left_token32 + right_token32 = 313
```

### Loss

Env 每步生成 compact label：

```text
foot_affordance_labels[num_envs, 2, 15, 4]
  0: safety_label
  1: edge_risk_label
  2: targetness_label
  3: target_valid_mask
```

当前主线只监督 safety：

```python
safety_loss = MSE(pred_affordance[..., 0], labels[..., 0])
affordance_loss = safety_loss
total_loss += affordance_loss_coef * affordance_loss
```

当前配置：

```python
affordance_loss_coef = 0.1
affordance_safety_loss_weight = 1.0
affordance_edge_loss_weight = 0.0
affordance_target_loss_weight = 0.0
```

### Label 生成

`safety_label` 来自每个候选脚底矩形对 `terrain_safety_map` 的采样：

```text
candidate center -> world frame
foot box sampling -> safety_mean * valid_ratio
```

这比完整 heatmap 更轻量，也更贴近落脚决策。

### 当前担忧与待验证

- Affordance token 每一帧都存在，而真实落脚是 touchdown event，二者语义不同。
- Safety-only token 可能诱导策略保守贴地，而不是自然交替摆腿。
- Affordance loss 会影响共享 depth/history 表征，可能需要 stop-gradient ablation。
- 需要记录 double contact、single contact、air time、touchdown safety 等指标判断它是否改善真实步态。

---

## Foot Safety: 脚底面积 + 惩罚型落脚安全

当前 `foot_safety` 不再是“踩到安全区域给正奖励”，而是：

```text
只在 touchdown 时惩罚不安全落脚。
```

这样避免策略通过频繁触地或蹭地刷安全奖励。

### 当前实现

```python
def _reward_foot_safety(self):
    contact_moment = self.foot_touchdown
    safety = self._current_foot_sole_safety()
    unsafe_penalty = relu(foot_safety_threshold - safety) / foot_safety_threshold
    return -(unsafe_penalty * contact_moment.float()).sum(dim=-1)
```

### 脚底面积查询

- 使用当前 foot link quaternion 得到脚底朝向。
- 在脚底矩形区域采样 `terrain_safety_map`。
- 默认脚底框：`0.22m x 0.10m`。
- 默认采样：`5 x 3`。
- 当前安全值：`safety_mean * valid_ratio`。
- 默认阈值：`foot_safety_threshold = 0.55`。

### 仍需补充

当前只完成了用户指定的两个核心修改：

```text
1. 单点查询 -> 脚底面积查询
2. 正向安全奖励 -> 不安全惩罚
```

下一步还需要：

```text
valid touchdown gate:
  previous_air_time > min_air_time
  swing height > min_swing_height

metrics:
  touchdown_count_per_sec
  touchdown_safety_mean
  unsafe_touchdown_ratio
  double_contact_ratio / single_contact_ratio / no_contact_ratio
```

---

## 路径条件踏面交互规划

### 故事动机

随机速度命令在平地、坡地等连续地形上通常可行，但楼梯是离散结构。只给随机速度时，机器人可能：

```text
斜向擦过台阶
只在平台附近活动
一步跨过多级
下楼时直接冲下去
用滑动/蹭地/碰撞方式通过
```

这些行为可能让 episode 走到终点，却没有形成我们真正想要的能力：

```text
沿楼梯方向，按顺序、稳定地处理每一级踏面。
```

因此，下一步计划引入训练期的 route-conditioned stair interaction。它不向 actor 输入特权目标，而是组织训练经验、约束 curriculum 和提供 debug 可视化。

### Route Guidance

对 `env_class == 4` 的 stair episode：

```text
path_start = reset 后 base xy
path_end   = 当前 stair terrain 对侧有效通道内随机点
path_dir / path_normal / path_len 缓存
```

训练 command：

```text
沿 path_dir 给保守前向速度
根据 lateral_error 给横向回正速度
heading 指向 path_dir
```

非 stair 地形保持原随机速度命令。

路径线的作用：

```text
保证训练中机器人与楼梯发生严密、正向、持续的交互。
```

---

## Tread Progression Scheduler

我们不再把“目标落脚点”理解成全局预生成的一串固定脚印，而是把 stair task 看成有序踏面交互过程。

沿路径线解析出：

```text
tread_0, tread_1, tread_2, ...
```

训练期维护：

```text
current_tread_idx
last_landing_foot
next_required_foot
next_target_tread
skip / miss / uncertain 状态
```

成功标准从旧逻辑：

```text
走得远 -> 升级
```

改成：

```text
route completion + ordered tread completion + low skip/miss -> 升级
```

如果机器人下楼时直接跨过几级，即使到达楼梯底部，也不应推动 terrain curriculum 升级。

### Skip / Miss 处理原则

第一版建议温和处理：

```text
skip 不 early terminate
skip 可以给小 penalty 或只记录指标
skip episode 不允许 terrain level 升级
missed tread episode 不允许 terrain level 升级
uncertain 不推进，也不立刻判 skip
```

这样不会过早切死探索，但能避免“跳级冲下去”被当成学会下楼梯。

---

## Event-Triggered Alternating Landing Region

用户提出的新思路：不要全局生成所有脚印，而是根据真实落脚事件在线生成下一步目标区域。

核心逻辑：

```text
第一脚踏上楼梯不限制左右脚。

如果左脚有效落在第 k 级：
  生成右脚第 k+1 级 landing region。

如果右脚有效落在第 k+1 级：
  给奖励并推进，生成左脚第 k+2 级 landing region。

反之亦然。
```

这个机制的好处：

```text
1. 尊重策略自己的起步相位。
2. 自然诱导左右脚交替。
3. 管理的是台阶交互顺序，不重复 affordance 的可踩性建模。
4. 比全局固定 footstep target 更自适应。
```

目标应该是 landing region，而不是精确点：

```text
下一阶踏面中心区域
沿路径方向保留一定长度
横向根据左右脚 nominal offset 给范围
避开踏面前后边缘
```

事件分类建议：

```text
目标脚落在下一阶 landing region：奖励并推进
目标脚落在当前阶：允许 recovery，不推进
非目标脚重复接触当前阶：忽略
目标脚直接落到 k+2 或更远：skip
base 已越过下一阶但无有效 touchdown：miss
脚底 overlap 不明确：uncertain，不推进也不判 skip
```

需要避免的旧设计：

```text
不要求两只脚都踩同一级台阶后才推进。
不强制第一脚必须是指定左右脚。
不把 next target 输入 actor observation。
不把目标区域设计成新的 safety reward。
```

---

## 可视化与诊断指标

为了证明 route/tread 机制确实让策略经历正确的台阶交互顺序，需要记录：

```text
route_completion
ordered_tread_completion
skip_tread_count
missed_tread_count
max_tread_jump
valid_touchdown_ratio
touchdown_safety_mean
unsafe_touchdown_ratio
double_contact_ratio
single_contact_ratio
no_contact_ratio
mean_air_time
downstairs_overshoot_speed
```

Viewer/top-down debug 建议叠加：

```text
路径线
ordered tread bands
current tread index
active landing region
最近 touchdown 点
skip/miss/uncertain 标记
机器人 base 和左右脚位置
```

这些指标比单看 `rew_foot_safety` 更可靠，因为 `rew_foot_safety` 受到 touchdown 频率和 episode 长度影响。

---

## 当前代码状态

### 已实现 / 当前启用

| 模块 | 状态 |
|------|------|
| `terrain_safety_map` | 已实现，作为 label/reward/debug source |
| 自遮挡 depth 渲染 | 已实现，训练与部署一致地允许相机看到机器人本体 |
| `FootAffordanceHead` | 已实现，G1 默认启用 |
| compact `foot_affordance_labels` | 已实现，默认只用 safety 通道 |
| `affordance_loss` | 已实现，当前 safety-only |
| `foot_safety` penalty-only | 已实现，脚底面积采样 + 不安全惩罚 |
| `was_contact` / `foot_touchdown` 顺序 | 已处理，touchdown reward 后统一更新 |
| footstep guidance / target reward | 当前关闭，用于后续重设计 |

### 保留为 ablation/debug

| 模块 | 状态 |
|------|------|
| `SegDecoder` | 保留代码路径，默认 `seg_loss_coef=0.0` |
| full GT safety heatmap storage | 仅 `seg_loss_coef > 0` 时存储，默认不存 |
| play 中 GT/Pred heatmap 可视化 | 可用于 debug，不作为主训练监督 |

### 需要实现 / 重构

| 模块 | 目的 |
|------|------|
| valid touchdown gate | 避免蹭地/接触抖动污染 foot_safety 和 tread progression |
| contact diagnostics | 判断是否存在双脚蹭地、长 double stance、无效 touchdown |
| route guidance | stair-only 路径条件命令，保证与楼梯充分交互 |
| tread parser | 沿路径解析 ordered tread sequence |
| tread progression scheduler | 管理当前踏面、skip、miss、uncertain |
| event-triggered landing region | 根据真实落脚事件生成下一脚下一阶区域 |
| curriculum gate | skip/miss episode 不推动 stair terrain level 升级 |
| top-down debug | 验证路径、踏面、landing region 和 touchdown 分类是否正确 |

---

## 论文叙事草案

当前更合适的论文主张不再是“安全热力图预测”，而是：

```text
通过可部署的 foot-centric safety affordance 与训练期的 route-conditioned tread progression，
让双足机器人在上下楼梯时学习安全、逐级、有顺序的踏面交互。
```

可能的技术贡献：

1. **Foot-centric safety affordance**
   用 compact body-frame 候选脚底区域监督替代 dense heatmap 主监督，使 actor 直接获得可部署的局部可踩性 token。

2. **Penalty-only sole-area landing safety**
   用脚底面积和 touchdown 事件定义不安全落脚惩罚，避免安全正奖励诱导频繁触地/蹭地。

3. **Route-conditioned tread progression**
   用训练期路径和踏面推进机制组织 stair episode，避免随机速度训练中跳级、绕行、滑下楼梯也被当成成功。

4. **Event-triggered alternating landing region**
   不全局规定脚印，而是根据真实 touchdown 事件在线生成下一脚下一阶区域，诱导自然交替台阶步态。

---

## 消融实验建议

```text
A. baseline: random velocity + depth policy
B. A + foot_safety penalty-only
C. B + safety-only FootAffordanceHead
D. C + affordance stop-gradient ablation
E. C + no-token ablation
F. C + route guidance only
G. F + tread progression curriculum gate
H. G + event-triggered alternating landing region
```

重点观察：

```text
terrain level 是否稳定上升
下楼 skip/miss 是否降低
是否形成自然交替 gait
foot_safety penalty 是否降低
真实 touchdown safety 是否提高
double stance / single stance / no contact 比例
是否仍存在双脚蹭地或长时间贴地滑动
affordance safety loss 是否与真实 touchdown safety 一致
```

---

## 待确认事项

- [ ] 当前 `FootAffordanceHead` 是否需要 stop-gradient，避免 auxiliary loss 过强影响共享 depth/history 表征。
- [ ] `affordance_loss_coef=0.1` 是否过大，需要 `0.01 / 0.0 / no-token` 对照。
- [ ] 是否需要把 actor 使用 affordance token 的方式从强 concat 改成更弱的 residual/gating。
- [ ] 有效 touchdown 门控阈值如何设置：`min_air_time`、`min_swing_height`、contact threshold。
- [ ] `foot_safety` 的脚底区域 score 是否需要从 mean 改成 p10/min-like 保守统计。
- [ ] route guidance 是否影响非 stair 地形的速度命令分布，需保持 stair-only。
- [ ] tread parser 如何鲁棒处理上楼/下楼、平台、边缘和斜向路径。
- [ ] skip/miss curriculum gate 是否会导致早期训练卡住，需要温和 penalty 或 staged curriculum。
- [ ] 真机部署时自遮挡 depth、噪声和相机位姿误差对 affordance token 的影响。
