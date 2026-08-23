# `model_54000.pt` — phasefreeze01 策略

这是星期四晚间 20:25 左右录制的多速度 sim2sim 所使用的完整 checkpoint，
对应 `phasefreeze01_stair20_down20_simple60_from50100` 训练 run。

## 文件

- `policy.pt`：导出的 TorchScript 部署策略。
- `model_54000.pt`：原始训练 checkpoint，保留用于复现或继续分析。
- `training_code_diff.patch`：该训练 run 启动时相对于仓库基线的代码差异，
  用于还原当时的代码状态；当前正在运行的训练代码没有被覆盖。

## 来源

- Checkpoint：
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100/model_54000.pt`
- 场景：`scene_payload_horizontal_nosing.xml`，上下楼梯均带 1 cm 鼻口。
- 策略接口：350 维历史观测输入，19 维动作输出。
- 关键训练版本：零速指令下冻结步态相位；尚未加入后续 foot-separation continuation 的改动。

## 已知 sim2sim 表现

- 0.0 m/s：10 秒前移约 3.4 cm，base 高度约 0.994→0.988 m。
- 0.6 m/s：混合速度视频中最稳定，爬升至约 1.848 m。
- 0.8/1.0 m/s：发生自动 reset 后再次爬升。
- 1.2 m/s：仅为超出训练范围的仿真压力测试；曾爬到约 2.185 m，不能直接用于真机。

## 安全说明

该策略的混合速度测试出现关节目标和力矩越限，尤其是 1.2 m/s 段；
只能作为仿真策略归档，部署真机前必须重新做限位、力矩和低速安全验证。

## SHA-256

- Checkpoint：`f58ace084d6c988e0ea07e7ee54bb33b83692e652ed234e680cae52fcf3ef6d0`
- Policy：`dc88ce8ec097c3c51fab8c1392198230cf10e4f704b49ec5e376f42275d3e380`
