# H1 Payload Horizontal Best Checkpoints

Archived on: 2026-07-01

This folder keeps the best checkpoint from the current horizontal payload resume experiment and the base checkpoint required to reproduce the resume chain.

## Selected Best Model

- File: `best_model_40400.pt`
- Source: `logs/h1_dwaq/2026-06-28_15-59-06_payload_resume_m18000_upright25_angvel008/model_40400.pt`
- Selection criterion: highest `Train/mean_reward` among saved checkpoints in the run.
- TensorBoard metrics at checkpoint step 40400:
  - `Train/mean_reward`: `92.310`
  - `Train/mean_episode_length`: `959.7`
  - `Curriculum/terrain_levels`: `4.184`
  - `Policy/mean_noise_std`: `0.458`

Later checkpoints in the same run collapsed around step 47700+ with reward near `-3`, episode length around `58`, and terrain level `0`, so the final checkpoint is not used as the best policy.

## Resume Base Model

- File: `resume_base_model_18000.pt`
- Source: `logs/h1_dwaq/2026-06-12_11-47-51_resume_dwaq_low_level/model_18000.pt`
- This is the checkpoint referenced by `train.sh` for the payload resume experiment.

## Checksums

```text
d2d07b6f371636afbd2e4d3324040ef8eaa90db0d218c6d503d1b0bccea74e8c  best_model_40400.pt
81c8b53c77f3e075e5d8c1387037ebe0fbc918c7afb2cf281c661bf9ce8202df  resume_base_model_18000.pt
```
