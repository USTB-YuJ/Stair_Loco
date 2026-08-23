# 2026-07-30 model_50100 策略

## 文件

- `policy.pt`：部署用 TorchScript 策略。
- `SIM2SIM_EXPERIMENTS.md`：该策略的视频、配置改动、效果和安全监测记录。

## 来源

- Checkpoint：
  `/root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-07-28_19-09-18_stair_nosing1cm_baseheight6_airtime08_from25800/model_50100.pt`
- Checkpoint SHA-256：
  `7363af43d29760f04893c5955b436b2a1cfdb1b1f33de6b73754e8b42a582dd0`
- Policy SHA-256：
  `0d09a76e11e2699a5c56f9e213c0e0132f411378b7a693b4dbae8f471ce43732`
- 接口：350 维历史观测输入，19 维动作输出。

## 当前代码状态

- 训练与 sim2sim 已加入零速步态相位冻结，阈值为 0.1。
- `model_50100.pt` 的权重是在未冻结零速相位的旧代码上训练得到的。
- 当前版本未恢复后续 guarded-v2 的关节目标硬限位和推力配置，不建议未经安全审查直接用于真机。

详细测试结果请查看 `SIM2SIM_EXPERIMENTS.md`。
