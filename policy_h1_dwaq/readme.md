# H1 DWAQ 策略归档

每次导出的策略按“日期 + checkpoint”放入独立文件夹，避免不同版本混在一起。

- `model_xxx.pt`：训练 checkpoint，可用于 play、sim2sim 或继续训练。
- `policy.pt`：由 checkpoint 导出的 TorchScript 部署策略。
- 当前整理版本：`2026-07-30_model50100/`
