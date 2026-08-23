# H1 DWAQ Sim2Sim Experiment Record

This file is the persistent record for H1 DWAQ videos. Append one entry for every
recorded run, including checkpoint, scene, command schedule, code/config state,
observed behavior, and safety-limit results.

## Exported policy

- Source checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-28_19-09-18_stair_nosing1cm_baseheight6_airtime08_from25800/model_50100.pt`
- Checkpoint SHA-256:
  `7363af43d29760f04893c5955b436b2a1cfdb1b1f33de6b73754e8b42a582dd0`
- TorchScript:
  `policy_h1_dwaq/policy.pt`
- Policy SHA-256:
  `0d09a76e11e2699a5c56f9e213c0e0132f411378b7a693b4dbae8f471ce43732`
- Interface: 350 history-observation inputs, 19 action outputs.

## Why `model_50100.pt` currently climbs better

This is an observed association, not a controlled single-variable ablation.

- It had trained to iteration 50100, versus iteration 28900 for the guarded-v2
  comparison checkpoint.
- Its curriculum used 20% ascending stairs with 1 cm noses, 20% ordinary
  descending stairs, and 60% simple terrain. The guarded-v2 run used 30%
  ascending nosing stairs, 30% descending nosing stairs, 5% flat, and 35%
  other terrain.
- The older setup used upright reward 2.5, base-height penalty -6.0 at 0.95 m,
  feet-air-time reward 0.8, and COM randomization of +/-5 cm on all axes.
- It did not contain the later joint-target hard clamp, target-limit/torque
  saturation penalties, zero-speed phase freeze, flat-only 40-100 N physical
  pushes, or the guarded-v2 optimizer reset and fixed 1e-4 learning rate.
- The absence of the hard target clamp allows more aggressive stair actions and
  is strongly associated with better climbing here, but it also produces large
  ankle target violations and torque saturation. This version is therefore not
  approved for direct real-robot deployment without a separate safety review.

## Video runs

### 2026-07-29 13:13 — historical baseline

- Video:
  `TienKung-Lab/training_logs/sim2sim_nosing_model50100_vx06_autoreset.mp4`
- Checkpoint: `model_50100.pt`
- Scene: `scene_payload_horizontal_nosing.xml`
- Command: 0.6 m/s for 30 s; reset after returning to flat.
- Result: base height 1.624 m at 4 s and 2.124 m at 6 s; returned to the
  approach flat and reset at 9.32 s. The behavior repeated consistently.
- Safety: ankle target excess reached 2.4566 rad left and 2.3185 rad right;
  ankle torque excess reached 61.73 and 67.62.

### 2026-07-30 — guarded-v2 comparison, 0.6 m/s

- Video:
  `TienKung-Lab/training_logs/sim2sim_guarded_v2_model28900_vx06_nosing_45s.mp4`
- Checkpoint: guarded-v2 `model_28900.pt`
- Command: 0.6 m/s.
- Result: reached the first nose, backed across the entrance threshold, and
  repeatedly reset without continuing up the staircase.
- Safety: target excess and torque excess were zero because of the hard target
  clamp; small ankle position-limit excess remained.

### 2026-07-30 — guarded-v2 comparison, 0.8 m/s

- Video:
  `TienKung-Lab/training_logs/sim2sim_guarded_v2_model28900_vx08_nosing_45s.mp4`
- Checkpoint: guarded-v2 `model_28900.pt`
- Command: 0.8 m/s.
- Result: destabilized near the first step; base height settled near 0.709 m
  and the robot did not continue climbing.
- Safety: target excess remained zero, but knee torque excess exceeded 438.

### 2026-07-30 — guarded-v2 comparison, zero speed

- Video:
  `TienKung-Lab/training_logs/sim2sim_guarded_v2_model28900_vx00_nosing_20s.mp4`
- Checkpoint: guarded-v2 `model_28900.pt`
- Command: 0.0 m/s.
- Result: base height stayed between 0.998 and 1.007 m with no joint or torque
  limit violations.

### 2026-07-30 — rollback verification, stand 10 s then climb at 0.6 m/s

- Video:
  `TienKung-Lab/training_logs/sim2sim_model50100_stand10_then_vx06_nosing_20s.mp4`
- Log:
  `TienKung-Lab/training_logs/sim2sim_model50100_stand10_then_vx06_nosing_20s.log`
- Checkpoint: `model_50100.pt`
- Scene: `scene_payload_horizontal_nosing.xml`
- Command: 0.0 m/s from 0-10 s, then 0.6 m/s from 10-20 s.
- Sim-only code change: added `--command-vx-switch-time`; training code was not
  changed.
- Standing result: base height changed from 0.991 m at 2 s to 0.986 m at
  10 s; forward drift at the switch was approximately 0.021 m.
- Climbing result: climbing was armed at 12.90 s; base height reached 1.505 m
  at 14 s and 1.785 m at 16 s; the robot returned to flat and reset at 17.48 s.
- Safety: target excess peaked at 2.8376 rad on the left ankle and 1.6579 rad
  on the right ankle. Knee torque excess reached 265.42 left and 217.85 right.

### 2026-07-30 — phase-freeze `model_51000.pt`, stand then climb

- Video:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51000_stand10_then_vx06_nosing_15s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51000_stand10_then_vx06_nosing_15s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100/model_51000.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command: 0.0 m/s from 0-10 s, then 0.6 m/s.
- Standing result: base height changed from 0.993 m at 2 s to 0.987 m at
  10 s. Forward drift at the switch was 0.028 m. The 5 s and 10 s frames show
  nearly unchanged posture, but drift is not yet lower than the pre-fine-tune
  checkpoint's 0.017 m.
- Climbing result: climbing armed at 12.54 s and base height reached 1.390 m
  at 14 s, but the robot then backed down to the approach flat and reset at
  15.04 s. It did not complete the staircase.
- Video duration: 15.66 s, covering the full standing segment and the first
  climbing attempt through automatic reset.
- Safety monitor summary was not finalized because the wall-clock recording
  timeout ended the process after the reset.

### 2026-07-30 — `model_51200.pt`, stand then 0.6/0.8 staged climb

- Video:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51200_stand10_vx06_step1_vx08_nosing_20s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51200_stand10_vx06_step1_vx08_nosing_20s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100/model_51200.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command: 0.0 m/s from 0-10 s, 0.6 m/s after 10 s, then 0.8 m/s after
  base height reached 1.15 m.
- Sim-only code change: added a second height-triggered command switch so the
  two-stage schedule is deterministic.
- Standing result: base height changed from 0.985 m at 2 s to 0.977 m at
  10 s; forward drift at the first switch was 0.021 m.
- The second switch occurred at x=1.058 m, h=1.151 m, t=12.86 s. Base height
  then fell to 1.132 m at 14 s and 1.057 m at 16 s. The robot returned to the
  entrance and reset at 17.64 s without completing the staircase.
- Safety: left/right ankle target excess reached 1.7868/1.7229 rad, ankle
  position excess reached 0.0538/0.0396 rad, and ankle torque excess reached
  43.6333/26.9249. Small hip-yaw position excess was also observed.
- Video duration: 20.02 s (1001 frames).

### 2026-07-30 — `model_51300.pt`, direct 0.8 m/s

- Video:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51300_vx08_nosing_20s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51300_vx08_nosing_20s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100/model_51300.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command: 0.8 m/s for 20 s, with automatic reset after returning to flat.
- Result: four nearly identical attempts. The robot armed the climb at 2.44 s
  and reached base height 1.530 m at 4 s, then returned to the entrance and
  reset at 4.92 s. Subsequent resets occurred at 9.84, 14.76, and 19.68 s.
  Direct 0.8 m/s climbs higher than the staged 0.6/0.8 command, but consistently
  backs down instead of continuing over the staircase.
- Safety: left/right ankle target excess reached 2.2084/1.8754 rad, ankle
  position excess reached 0.0775/0.0251 rad, and ankle torque excess reached
  95.4978/34.7309. Smaller hip-yaw, hip-roll, and elbow violations were also
  observed.
- Video duration: 20.02 s (1001 frames).

### 2026-07-30 — `model_51500.pt`, direct 1.0 m/s

- Video:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51500_vx10_nosing_20s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model51500_vx10_nosing_20s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100/model_51500.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command: 1.0 m/s for 20 s, with automatic reset after returning to flat.
- Result: base height reached 1.674 m at 4 s, which was higher than the direct
  0.8 m/s run, but the robot returned to the entrance and reset at 7.44 s. A
  second nearly identical attempt reset at 14.88 s; the third attempt was
  climbing at 1.674 m when the video ended. The additional speed improves
  upward progress but still does not produce a complete traversal.
- Safety: left/right ankle target excess reached 1.3350/1.4749 rad, ankle
  position excess reached 0.0893/0.0755 rad, and ankle torque excess reached
  56.6006/47.9705. Knee, shoulder, and elbow violations were also observed.
- Video duration: 20.02 s (1001 frames).

### 2026-07-30 — `model_54000.pt`, continuous 0.0/0.6/0.8/1.0/1.2 m/s schedule

- Video:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model54000_stand10_vx06_vx08_vx10_vx12_50s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-30_model50100/sim2sim_phasefreeze_model54000_stand10_vx06_vx08_vx10_vx12_50s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100/model_54000.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command schedule: 0.0 m/s from 0-10 s, 0.6 m/s from 10-20 s, 0.8 m/s from
  20-30 s, 1.0 m/s from 30-40 s, and 1.2 m/s from 40-50 s. Automatic reset
  remained enabled after a return to the approach flat. The 1.2 m/s segment is
  beyond the training command maximum of 1.0 m/s and is a sim-only stress test.
- Sim-only code change: added `--command-vx-schedule` for an arbitrary
  time-based command sequence and `--command-vx-limit` for an explicit test
  limit. The schedule clock continues across automatic resets. Training code
  and the learned policy were not changed by this recording feature.
- Standing result: base height decreased from 0.994 m at 2 s to 0.988 m at
  10 s; forward drift at the first switch was 0.034 m.
- 0.6 m/s result: the climb armed at 12.36 s and base height rose to 1.848 m
  by 20 s. This was the cleanest continuous segment in the run.
- 0.8 m/s result: after the switch at x=2.913 m, the robot backed down and
  reset at 25.08 s. On the second attempt it climbed rapidly to 1.831 m by
  30.02 s.
- 1.0 m/s result: after reaching 1.916 m at 31.1 s, the robot returned to the
  entrance and reset at 34.62 s. The next attempt reached 1.628 m by 38.6 s.
- 1.2 m/s stress-test result: the existing attempt reset at 41.30 s. A fresh
  attempt then reached base height 2.185 m by 49.3 s, but this does not imply
  hardware safety.
- Safety over the complete mixed-speed run: target excess peaked at 3.1856 rad
  on the left ankle and 5.3089 rad on the right ankle. The largest measured
  position-limit excess was 0.2012 rad on the left knee. Torque excess peaked
  at 540.6421 on the left knee, 349.3863 on the left hip pitch, and 283.1390
  on the right knee. The checkpoint is not approved for direct real-robot use.
- Video duration: 50.04 s (1251 frames, 640x480 at 25 fps).

### 2026-07-31 — foot-separation continuation `model_71700.pt`, 0.6 m/s

- Video:
  `policy_h1_dwaq/2026-07-31_model71700/sim2sim_footsep_model71700_vx06_nosing_20s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-31_model71700/sim2sim_footsep_model71700_vx06_nosing_20s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_21-03-35_phasefreeze02_footsep_fix_from55000/model_71700.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command: 0.6 m/s for 20 s with automatic reset after returning to the
  approach flat.
- Result: the first attempt armed at 3.02 s, reached base height 1.534 m at
  6 s, then returned to flat and reset at 11.72 s. The second attempt repeated
  the same pattern, reaching 1.534 m at 17.7 s before the video ended. It did
  not complete the staircase, but the behavior was repeatable and did not show
  the previous high-speed leg explosion.
- Safety: the monitor saved 7 joint plots. Left/right ankle target excess was
  2.0417/1.6695 rad, ankle torque excess 41.9009/41.0059, and left knee torque
  excess 20.1647. These are substantially below the previous `model_54000.pt`
  mixed-speed run, but still not suitable for direct real-robot deployment.
- Video duration: 20.04 s (501 frames, 640x480 at 25 fps).

### 2026-07-31 — `model_65000.pt`, continuous 0.0/0.6/0.8/1.0/1.2 m/s schedule

- Video:
  `policy_h1_dwaq/2026-07-31_model65000/sim2sim_footsep_model65000_stand10_vx06_vx08_vx10_vx12_50s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-31_model65000/sim2sim_footsep_model65000_stand10_vx06_vx08_vx10_vx12_50s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_21-03-35_phasefreeze02_footsep_fix_from55000/model_65000.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command schedule: 0.0 m/s from 0-10 s, 0.6 m/s from 10-20 s, 0.8 m/s from
  20-30 s, 1.0 m/s from 30-40 s, and 1.2 m/s from 40-50 s. Automatic reset
  remained enabled. The 1.2 m/s segment is outside the training range and is
  a sim-only stress test.
- Standing result: base height decreased from 0.992 m at 2 s to 0.986 m at
  10 s; forward drift at the switch was 0.027 m.
- 0.6 m/s result: first attempt reached 1.218 m at 16 s and reset at 16.52 s;
  the second attempt was still climbing at the 20 s boundary.
- 0.8 m/s result: first attempt reset at 23.02 s after reaching 1.392 m; the
  second attempt reached 1.784 m at 29 s and 1.874 m at the 30 s switch.
- 1.0 m/s result: the first attempt reset at 35.40 s and the second reset at
  38.22 s. It did not complete the staircase.
- 1.2 m/s result: one attempt reset at 41.28 s; a fresh attempt reached 1.705 m
  at 45.3 s before falling back to 1.378 m at 49.3 s.
- Safety over the complete mixed-speed run: 18 joint plots were saved. Largest
  target excesses were left hip yaw 2.5598 rad, left ankle 2.6944 rad, and
  right ankle 2.3004 rad. Largest torque excesses were left hip yaw 479.0040,
  torso 251.7435, and right knee 150.5123. The policy is not approved for
  direct real-robot deployment.
- Video duration: 50.04 s (1251 frames, 640x480 at 25 fps).

### 2026-07-31 — `model_60000.pt`, continuous 0.0/0.6/0.8/1.0/1.2 m/s schedule

- Video:
  `policy_h1_dwaq/2026-07-31_model60000/sim2sim_footsep_model60000_stand10_vx06_vx08_vx10_vx12_50s.mp4`
- Log:
  `policy_h1_dwaq/2026-07-31_model60000/sim2sim_footsep_model60000_stand10_vx06_vx08_vx10_vx12_50s.log`
- Checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_21-03-35_phasefreeze02_footsep_fix_from55000/model_60000.pt`
- Scene: `scene_payload_horizontal_nosing.xml`, with 1 cm noses on both
  ascending and descending stairs.
- Command schedule: 0.0 m/s from 0-10 s, 0.6 m/s from 10-20 s, 0.8 m/s from
  20-30 s, 1.0 m/s from 30-40 s, and 1.2 m/s from 40-50 s. Automatic reset
  remained enabled. The 1.2 m/s segment is outside the training range and is
  a sim-only stress test.
- Standing result: base height decreased from 0.987 m at 2 s to 0.981 m at
  10 s; forward drift at the switch was 0.026 m.
- 0.6 m/s result: the first attempt reached 1.749 m at 16 s, then clearly
  fell/returned and reset at 19.74 s (base height 0.453 m).
- 0.8 m/s result: resets occurred at 24.16 s and 28.40 s after reaching base
  heights of about 1.308 m and 1.207 m.
- 1.0 m/s result: resets occurred at 33.22 s and 39.30 s; the highest logged
  heights were about 1.248 m and 1.584 m.
- 1.2 m/s result: the attempt reset at 48.16 s; a fresh attempt reached about
  1.680 m at 45.3 s before falling to 1.400 m at 47.3 s.
- Safety over the complete mixed-speed run: 19 joint plots were saved. Target
  excess reached 11.9136 rad at the right ankle and 8.0742 rad at the left
  hip pitch; torque excess reached 1653.6449 at the left knee, 1416.0506 at
  the left hip pitch, and 941.6895 at the torso. This is materially worse than
  `model_65000.pt` and is not approved for direct real-robot deployment.
- Video duration: 50.04 s (1251 frames, 640x480 at 25 fps).

## Code changes after the `model_50100.pt` rollback

### 2026-07-30 — zero-speed gait-phase freeze

- Training and sim2sim now use the same standing-command threshold of 0.1.
- Below the threshold, the gait clock is frozen at left phase 0.0 and right
  phase 0.5 (double support). Moving commands keep the original 0.8 s gait
  period.
- This isolated change does not restore the later joint-target hard clamp,
  force-push randomization, guarded-v2 rewards, or optimizer changes.
- A non-video 10.2 s zero-command validation used `model_50100.pt` on
  `scene_payload_horizontal_nosing.xml`.
- Validation log:
  `TienKung-Lab/training_logs/sim2sim_model50100_phasefreeze_vx00_10s.log`
- Result: forward drift at 10 s was 0.017 m, versus 0.021 m before the freeze;
  base height was 0.985 m. Left/right ankle target excess was reduced to
  0.1118/0.0595 rad, with no position or torque-limit excess.
- Interpretation: the existing checkpoint was trained with a cycling clock at
  zero command, so inference-only freezing improves but cannot fully eliminate
  residual motion. Further training with the new observation behavior is
  required for the policy to learn a truly quiet stance.

### 2026-07-30 — soft foot-separation and crossing penalties

- Added `feet_crossing_humanoid`, which penalizes the H1 left foot losing its
  expected +y ordering relative to the right foot in the base frame. This
  catches a leg crossing even when the feet differ in x or z on a stair.
- Overrode the H1 DWAQ `feet_too_near` term from weight `-2.0`, threshold
  `0.30 m` to weight `-3.5`, threshold `0.32 m`.
- Added a `feet_y_distance` term with weight `-0.6` to keep the lateral foot
  spacing near the existing `0.299 m` nominal value.
- These are soft training rewards. MuJoCo/Isaac Lab foot self-collision stays
  enabled; no collision filtering was added to hide the problem.
- The latest `model_54000.pt` video showed the likely root cause: at high
  speed the policy produced large ankle target and torque violations, so these
  rewards should be combined with the existing target/torque safety review.
- The training process that was already running was not interrupted and will
  not use these new rewards; a new continuation must be started from a chosen
  checkpoint to train them.

### 2026-07-30 — foot-separation reward continuation from `model_55000.pt`

- The old phase-freeze run was stopped at its latest complete checkpoint,
  `model_55000.pt`, so it would not keep training with the old reward stack.
- An initial restart exposed an environment-interface mismatch before the
  first learning iteration (`G1DwaqEnv` exposes `feet_cfg.body_ids`). The
  reward functions were corrected to support both `feet_body_ids` and
  `feet_cfg.body_ids`; no checkpoint was written by the failed attempt.
- Completed continuation:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_21-03-35_phasefreeze02_footsep_fix_from55000`
- Source checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100/model_55000.pt`
- Target: 20,800 additional iterations, ending at the original total of
  75,800. The first successful iterations logged both
  `Episode_Reward/feet_crossing` and `Episode_Reward/feet_y_distance`.
- Final checkpoint: `model_75799.pt` (the runner's final saved iteration).
- Final observed window: reward about 74–80, episode length about 860–920,
  terrain level about 4.75. The last foot-crossing and lateral-spacing terms
  were approximately `-0.219` and `-0.071`, respectively. The run completed
  without NaN, out-of-memory, or abnormal training errors; the residual
  simulator process was terminated after the final checkpoint was saved.

## Training continuations

### 2026-07-30 — phase-freeze fine-tuning from `model_50100.pt`

- Status: running.
- Source checkpoint:
  `TienKung-Lab/logs/h1_dwaq/2026-07-28_19-09-18_stair_nosing1cm_baseheight6_airtime08_from25800/model_50100.pt`
- New run:
  `TienKung-Lab/logs/h1_dwaq/2026-07-30_17-19-13_phasefreeze01_stair20_down20_simple60_from50100`
- Console log:
  `TienKung-Lab/training_logs/train_phasefreeze01_stair20_down20_simple60_from50100.log`
- Schedule: iteration 50100 to 75800 (25700 additional iterations), 4096
  environments on GPU 0.
- Isolated behavior change: commands below magnitude 0.1 freeze the gait clock
  at the double-support phase. The command sampler supplies standing commands
  to 20% of environments.
- Terrain: 20% ascending stairs with 1 cm noses, 20% ordinary descending
  stairs, and 60% simple/other terrain.
- Initial health check at iteration 50134: mean reward 49.44, mean episode
  length 784.10, terrain level 0.7175, approximately 8.1 GB GPU memory, and no
  NaN, out-of-memory error, or abnormal exit.
- TensorBoard compares `phasefreeze01` against `model50100_source` on remote
  port 6006.
