# Training Report: longbench-v2-default-r5-s42

> Auto-generated 2026-03-07 01:10 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/5xfdtlhq)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `5xfdtlhq` |
| **Status** | crashed |
| **Started** | 2026-01-20T05:28:08Z |
| **Steps** | 0 |
| **env_config** | `longbench_v2/default` |
| **eval_env_config** | `None` |
| **trainer_config** | `qwen3_4b_pg_nstep` |
| **learning_rate** | `None` |
| **mini_batch_size** | `None` |
| **seed** | `42` |
| **replicate** | `5` |
| **group_size** | `None` |
| **hide_observations** | `None` |
| **actions_only** | `None` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| eval/best_batch | 0 |
| eval/best_metric | 0.262500 |
| eval/best_sampling_client_path | tinker://485b38c6-faa8-5c94-b513-8e9ea774f93b:train:0/sampler_weights/000000_best |
| eval/try_0/accuracy | 0.262500 |
| eval/try_0/correct | 21 |
| eval/try_0/correct_max | True |
| eval/try_0/correct_std | 0.439993 |
| eval/try_0/final_reward | 0.262500 |
| eval/try_0/final_reward_max | 1 |
| eval/try_0/final_reward_std | 0.439993 |
| eval/try_0/first_return | 0.181757 |
| eval/try_0/first_return_max | 1 |
| eval/try_0/first_return_std | 0.312487 |
| eval/try_0/timesteps | 1 |
| eval/try_0/timesteps_max | 1 |
| eval/try_0/timesteps_std | 0 |
| eval/try_0/total | 80 |
| eval/try_0/total_max | 1 |
| eval/try_0/total_std | 0 |
| optim/entropy | 0.383050 |
| optim/kl_sample_train_v1 | -0.007473 |
| optim/kl_sample_train_v2 | 0.042536 |
| optim/lr | 0.000040 |
| time/assemble_training_data | 5.838035 |
| time/compute_kl_sample_train | 0.926311 |
| time/run_evals | 233.197365 |
| time/save_checkpoint | 2.486813 |
| time/total | 800.770424 |
| time/train | 172.356932 |
| train/try_0/accuracy | 0.257812 |
| train/try_0/correct | 33 |
| train/try_0/correct_max | True |
| train/try_0/correct_std | 0.437430 |
| train/try_0/final_reward | 0.257812 |
| train/try_0/final_reward_max | 1 |
| train/try_0/final_reward_std | 0.437430 |
| train/try_0/first_return | 0.187972 |
| train/try_0/first_return_max | 0.900000 |
| train/try_0/first_return_std | 0.323673 |
| train/try_0/timesteps | 1 |
| train/try_0/timesteps_max | 1 |
| train/try_0/timesteps_std | 0 |
| train/try_0/total | 128 |
| train/try_0/total_max | 1 |
| train/try_0/total_std | 0 |

## Training Curves

### Eval / Best Batch

![Eval / Best Batch](plots/eval_best_batch.png)

### Eval / Best Metric

![Eval / Best Metric](plots/eval_best_metric.png)

### Eval / Best Sampling Client Path

![Eval / Best Sampling Client Path](plots/eval_best_sampling_client_path.png)

### Eval / Try 0 / Accuracy

![Eval / Try 0 / Accuracy](plots/eval_try_0_accuracy.png)

### Eval / Try 0 / Correct

![Eval / Try 0 / Correct](plots/eval_try_0_correct.png)

### Eval / Try 0 / Correct Max

![Eval / Try 0 / Correct Max](plots/eval_try_0_correct_max.png)

### Eval / Try 0 / Correct Std

![Eval / Try 0 / Correct Std](plots/eval_try_0_correct_std.png)

### Eval / Try 0 / Final Reward

![Eval / Try 0 / Final Reward](plots/eval_try_0_final_reward.png)

### Eval / Try 0 / Final Reward Max

![Eval / Try 0 / Final Reward Max](plots/eval_try_0_final_reward_max.png)

### Eval / Try 0 / Final Reward Std

![Eval / Try 0 / Final Reward Std](plots/eval_try_0_final_reward_std.png)

### Eval / Try 0 / First Return

![Eval / Try 0 / First Return](plots/eval_try_0_first_return.png)

### Eval / Try 0 / First Return Max

![Eval / Try 0 / First Return Max](plots/eval_try_0_first_return_max.png)

### Eval / Try 0 / First Return Std

![Eval / Try 0 / First Return Std](plots/eval_try_0_first_return_std.png)

### Eval / Try 0 / Timesteps

![Eval / Try 0 / Timesteps](plots/eval_try_0_timesteps.png)

### Eval / Try 0 / Timesteps Max

![Eval / Try 0 / Timesteps Max](plots/eval_try_0_timesteps_max.png)

### Eval / Try 0 / Timesteps Std

![Eval / Try 0 / Timesteps Std](plots/eval_try_0_timesteps_std.png)

### Eval / Try 0 / Total

![Eval / Try 0 / Total](plots/eval_try_0_total.png)

### Eval / Try 0 / Total Max

![Eval / Try 0 / Total Max](plots/eval_try_0_total_max.png)

### Eval / Try 0 / Total Std

![Eval / Try 0 / Total Std](plots/eval_try_0_total_std.png)

### Optim / Entropy

![Optim / Entropy](plots/optim_entropy.png)

### Optim / Kl Sample Train V1

![Optim / Kl Sample Train V1](plots/optim_kl_sample_train_v1.png)

### Optim / Kl Sample Train V2

![Optim / Kl Sample Train V2](plots/optim_kl_sample_train_v2.png)

### Optim / Lr

![Optim / Lr](plots/optim_lr.png)

### Time / Assemble Training Data

![Time / Assemble Training Data](plots/time_assemble_training_data.png)

### Time / Compute Kl Sample Train

![Time / Compute Kl Sample Train](plots/time_compute_kl_sample_train.png)

### Time / Run Evals

![Time / Run Evals](plots/time_run_evals.png)

### Time / Save Checkpoint

![Time / Save Checkpoint](plots/time_save_checkpoint.png)

### Time / Total

![Time / Total](plots/time_total.png)

### Time / Train

![Time / Train](plots/time_train.png)

### Train / Try 0 / Accuracy

![Train / Try 0 / Accuracy](plots/train_try_0_accuracy.png)

### Train / Try 0 / Correct

![Train / Try 0 / Correct](plots/train_try_0_correct.png)

### Train / Try 0 / Correct Max

![Train / Try 0 / Correct Max](plots/train_try_0_correct_max.png)

### Train / Try 0 / Correct Std

![Train / Try 0 / Correct Std](plots/train_try_0_correct_std.png)

### Train / Try 0 / Final Reward

![Train / Try 0 / Final Reward](plots/train_try_0_final_reward.png)

### Train / Try 0 / Final Reward Max

![Train / Try 0 / Final Reward Max](plots/train_try_0_final_reward_max.png)

### Train / Try 0 / Final Reward Std

![Train / Try 0 / Final Reward Std](plots/train_try_0_final_reward_std.png)

### Train / Try 0 / First Return

![Train / Try 0 / First Return](plots/train_try_0_first_return.png)

### Train / Try 0 / First Return Max

![Train / Try 0 / First Return Max](plots/train_try_0_first_return_max.png)

### Train / Try 0 / First Return Std

![Train / Try 0 / First Return Std](plots/train_try_0_first_return_std.png)

### Train / Try 0 / Timesteps

![Train / Try 0 / Timesteps](plots/train_try_0_timesteps.png)

### Train / Try 0 / Timesteps Max

![Train / Try 0 / Timesteps Max](plots/train_try_0_timesteps_max.png)

### Train / Try 0 / Timesteps Std

![Train / Try 0 / Timesteps Std](plots/train_try_0_timesteps_std.png)

### Train / Try 0 / Total

![Train / Try 0 / Total](plots/train_try_0_total.png)

### Train / Try 0 / Total Max

![Train / Try 0 / Total Max](plots/train_try_0_total_max.png)

### Train / Try 0 / Total Std

![Train / Try 0 / Total Std](plots/train_try_0_total_std.png)
