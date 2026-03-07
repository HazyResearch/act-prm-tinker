# Training Report: -r0-s0

> Auto-generated 2026-03-07 01:10 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/hlurmghs)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `hlurmghs` |
| **Status** | crashed |
| **Started** | 2026-01-19T18:48:57Z |
| **Steps** | 77 |

## Latest Metrics

| Metric | Value |
|--------|-------|
| act_prm_sft_eval/best_batch | 59 |
| act_prm_sft_eval/best_metric | 0.160000 |
| act_prm_sft_eval/best_sampling_client_path | tinker://3a2ca9ac-0b2b-5267-b2f6-b96085c70aac:train:1/sampler_weights/act_prm_sft_eval_000059_best |
| act_prm_sft_eval/best_state_path | tinker://3a2ca9ac-0b2b-5267-b2f6-b96085c70aac:train:1/weights/act_prm_sft_eval_000059_best |
| act_prm_sft_eval/try_0/accuracy | 0.160000 |
| act_prm_sft_eval/try_0/correct | 16 |
| act_prm_sft_eval/try_0/correct_max | True |
| act_prm_sft_eval/try_0/correct_std | 0.366606 |
| act_prm_sft_eval/try_0/final_reward | 0.160000 |
| act_prm_sft_eval/try_0/final_reward_max | 1 |
| act_prm_sft_eval/try_0/final_reward_std | 0.366606 |
| act_prm_sft_eval/try_0/first_return | 0.142290 |
| act_prm_sft_eval/try_0/first_return_max | 0.900000 |
| act_prm_sft_eval/try_0/first_return_std | 0.326447 |
| act_prm_sft_eval/try_0/timesteps | 1 |
| act_prm_sft_eval/try_0/timesteps_max | 1 |
| act_prm_sft_eval/try_0/timesteps_std | 0 |
| act_prm_sft_eval/try_0/total | 100 |
| act_prm_sft_eval/try_0/total_max | 1 |
| act_prm_sft_eval/try_0/total_std | 0 |
| optim/lr | 0.000040 |
| time/assemble_training_data | 0.132485 |
| time/run_evals | 657.025462 |
| time/save_checkpoint | 2.717551 |
| time/total | 9.946629 |
| time/train | 7.089149 |

## Training Curves

### Act Prm Sft Eval / Best Batch

![Act Prm Sft Eval / Best Batch](plots/act_prm_sft_eval_best_batch.png)

### Act Prm Sft Eval / Best Metric

![Act Prm Sft Eval / Best Metric](plots/act_prm_sft_eval_best_metric.png)

### Act Prm Sft Eval / Best Sampling Client Path

![Act Prm Sft Eval / Best Sampling Client Path](plots/act_prm_sft_eval_best_sampling_client_path.png)

### Act Prm Sft Eval / Best State Path

![Act Prm Sft Eval / Best State Path](plots/act_prm_sft_eval_best_state_path.png)

### Act Prm Sft Eval / Try 0 / Accuracy

![Act Prm Sft Eval / Try 0 / Accuracy](plots/act_prm_sft_eval_try_0_accuracy.png)

### Act Prm Sft Eval / Try 0 / Correct

![Act Prm Sft Eval / Try 0 / Correct](plots/act_prm_sft_eval_try_0_correct.png)

### Act Prm Sft Eval / Try 0 / Correct Max

![Act Prm Sft Eval / Try 0 / Correct Max](plots/act_prm_sft_eval_try_0_correct_max.png)

### Act Prm Sft Eval / Try 0 / Correct Std

![Act Prm Sft Eval / Try 0 / Correct Std](plots/act_prm_sft_eval_try_0_correct_std.png)

### Act Prm Sft Eval / Try 0 / Final Reward

![Act Prm Sft Eval / Try 0 / Final Reward](plots/act_prm_sft_eval_try_0_final_reward.png)

### Act Prm Sft Eval / Try 0 / Final Reward Max

![Act Prm Sft Eval / Try 0 / Final Reward Max](plots/act_prm_sft_eval_try_0_final_reward_max.png)

### Act Prm Sft Eval / Try 0 / Final Reward Std

![Act Prm Sft Eval / Try 0 / Final Reward Std](plots/act_prm_sft_eval_try_0_final_reward_std.png)

### Act Prm Sft Eval / Try 0 / First Return

![Act Prm Sft Eval / Try 0 / First Return](plots/act_prm_sft_eval_try_0_first_return.png)

### Act Prm Sft Eval / Try 0 / First Return Max

![Act Prm Sft Eval / Try 0 / First Return Max](plots/act_prm_sft_eval_try_0_first_return_max.png)

### Act Prm Sft Eval / Try 0 / First Return Std

![Act Prm Sft Eval / Try 0 / First Return Std](plots/act_prm_sft_eval_try_0_first_return_std.png)

### Act Prm Sft Eval / Try 0 / Timesteps

![Act Prm Sft Eval / Try 0 / Timesteps](plots/act_prm_sft_eval_try_0_timesteps.png)

### Act Prm Sft Eval / Try 0 / Timesteps Max

![Act Prm Sft Eval / Try 0 / Timesteps Max](plots/act_prm_sft_eval_try_0_timesteps_max.png)

### Act Prm Sft Eval / Try 0 / Timesteps Std

![Act Prm Sft Eval / Try 0 / Timesteps Std](plots/act_prm_sft_eval_try_0_timesteps_std.png)

### Act Prm Sft Eval / Try 0 / Total

![Act Prm Sft Eval / Try 0 / Total](plots/act_prm_sft_eval_try_0_total.png)

### Act Prm Sft Eval / Try 0 / Total Max

![Act Prm Sft Eval / Try 0 / Total Max](plots/act_prm_sft_eval_try_0_total_max.png)

### Act Prm Sft Eval / Try 0 / Total Std

![Act Prm Sft Eval / Try 0 / Total Std](plots/act_prm_sft_eval_try_0_total_std.png)

### Optim / Lr

![Optim / Lr](plots/optim_lr.png)

### Time / Assemble Training Data

![Time / Assemble Training Data](plots/time_assemble_training_data.png)

### Time / Run Evals

![Time / Run Evals](plots/time_run_evals.png)

### Time / Save Checkpoint

![Time / Save Checkpoint](plots/time_save_checkpoint.png)

### Time / Total

![Time / Total](plots/time_total.png)

### Time / Train

![Time / Train](plots/time_train.png)
