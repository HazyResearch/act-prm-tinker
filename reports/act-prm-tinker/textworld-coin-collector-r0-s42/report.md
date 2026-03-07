# Training Report: textworld-coin-collector-r0-s42

> Auto-generated 2026-03-07 01:10 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/8fa2pckf)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `8fa2pckf` |
| **Status** | crashed |
| **Started** | 2026-01-21T07:42:24Z |
| **Steps** | 33 |
| **env_config** | `textworld/coin_collector` |
| **eval_env_config** | `None` |
| **trainer_config** | `qwen3_4b_pg_nstep` |
| **learning_rate** | `None` |
| **mini_batch_size** | `None` |
| **seed** | `42` |
| **replicate** | `0` |
| **group_size** | `None` |
| **hide_observations** | `None` |
| **actions_only** | `None` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| eval/best_batch | 20 |
| eval/best_metric | 0 |
| eval/best_sampling_client_path | tinker://ec8c975e-487d-56a5-bb88-34dbf97a5c87:train:0/sampler_weights/000020_best |
| eval/try_0/accuracy | 0 |
| eval/try_0/action_prob | 0 |
| eval/try_0/action_prob_max | 0 |
| eval/try_0/action_prob_std | 0 |
| eval/try_0/correct | 0 |
| eval/try_0/correct_max | False |
| eval/try_0/correct_std | 0 |
| eval/try_0/final_reward | 0 |
| eval/try_0/final_reward_max | 0 |
| eval/try_0/final_reward_std | 0 |
| eval/try_0/first_return | 0 |
| eval/try_0/first_return_max | 0 |
| eval/try_0/first_return_std | 0 |
| eval/try_0/last_state_len | 5406 |
| eval/try_0/last_state_len_max | 6163 |
| eval/try_0/last_state_len_std | 1100.447364 |
| eval/try_0/timesteps | 1 |
| eval/try_0/timesteps_max | 1 |
| eval/try_0/timesteps_std | 0 |
| eval/try_0/total | 5 |
| eval/try_0/total_max | 1 |
| eval/try_0/total_std | 0 |
| optim/entropy | 0.310077 |
| optim/kl_sample_train_v1 | 0.000255 |
| optim/kl_sample_train_v2 | 0.001339 |
| optim/lr | 0.000040 |
| time/assemble_training_data | 11.308669 |
| time/compute_kl_sample_train | 1.588970 |
| time/run_evals | 94.618163 |
| time/save_checkpoint | 10.030605 |
| time/total | 676.482899 |
| time/train | 165.256722 |
| train/try_0/accuracy | 0 |
| train/try_0/action_prob | 0 |
| train/try_0/action_prob_max | 0 |
| train/try_0/action_prob_std | 0 |
| train/try_0/correct | 0 |
| train/try_0/correct_max | False |
| train/try_0/correct_std | 0 |
| train/try_0/final_reward | 0 |
| train/try_0/final_reward_max | 0 |
| train/try_0/final_reward_std | 0 |
| train/try_0/first_return | 0 |
| train/try_0/first_return_max | 0 |
| train/try_0/first_return_std | 0 |
| train/try_0/last_state_len | 4677.289062 |
| train/try_0/last_state_len_max | 6887 |
| train/try_0/last_state_len_std | 1409.935483 |
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

### Eval / Try 0 / Action Prob

![Eval / Try 0 / Action Prob](plots/eval_try_0_action_prob.png)

### Eval / Try 0 / Action Prob Max

![Eval / Try 0 / Action Prob Max](plots/eval_try_0_action_prob_max.png)

### Eval / Try 0 / Action Prob Std

![Eval / Try 0 / Action Prob Std](plots/eval_try_0_action_prob_std.png)

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

### Eval / Try 0 / Last State Len

![Eval / Try 0 / Last State Len](plots/eval_try_0_last_state_len.png)

### Eval / Try 0 / Last State Len Max

![Eval / Try 0 / Last State Len Max](plots/eval_try_0_last_state_len_max.png)

### Eval / Try 0 / Last State Len Std

![Eval / Try 0 / Last State Len Std](plots/eval_try_0_last_state_len_std.png)

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

### Train / Try 0 / Action Prob

![Train / Try 0 / Action Prob](plots/train_try_0_action_prob.png)

### Train / Try 0 / Action Prob Max

![Train / Try 0 / Action Prob Max](plots/train_try_0_action_prob_max.png)

### Train / Try 0 / Action Prob Std

![Train / Try 0 / Action Prob Std](plots/train_try_0_action_prob_std.png)

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

### Train / Try 0 / Last State Len

![Train / Try 0 / Last State Len](plots/train_try_0_last_state_len.png)

### Train / Try 0 / Last State Len Max

![Train / Try 0 / Last State Len Max](plots/train_try_0_last_state_len_max.png)

### Train / Try 0 / Last State Len Std

![Train / Try 0 / Last State Len Std](plots/train_try_0_last_state_len_std.png)

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
