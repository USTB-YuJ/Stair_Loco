# `model_71700.pt` — foot-separation continuation

## 文件

- `policy.pt`：导出的 TorchScript 策略。
- `sim2sim_footsep_model71700_vx06_nosing_20s.mp4`：鼻口楼梯 0.6 m/s 录制。
- `sim2sim_footsep_model71700_vx06_nosing_20s.log`：运行日志和安全监测结果。

## 来源

- Checkpoint：
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_21-03-35_phasefreeze02_footsep_fix_from55000/model_71700.pt`
- 场景：`scene_payload_horizontal_nosing.xml`，上下楼梯均带 1 cm 鼻口。
- 策略接口：350 维历史观测输入，19 维动作输出。
- 指令：0.6 m/s，20 s，回到平地后自动 reset。

## 本次复现结果

- 两次爬升均达到约 1.534 m。
- 第一次在 11.72 s 回到平地 reset；第二次在视频结束前重复同样的爬升过程。
- 左/右踝目标角越限约 2.0417/1.6695 rad。
- 左/右踝力矩越限约 41.9009/41.0059；共保存 7 个关节监测图。
- 视频为 20.04 s、501 帧、640×480、25 fps。

## SHA-256

- Checkpoint：`c1efd78ef954ea0788623a461d0d835e69d808bc768a6cfa40c7ee1d9904bffe`
- Policy：`bd4f1b8811f2412b35b9825f0812fd6af822eef8921d5a32913fe499dec78da8`

该策略仍用于仿真验证，不建议未经限位和力矩安全审查直接上真机。
