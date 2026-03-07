# Training Report: act-prm-taubench-retail-500-fs1-r0-s42

> Auto-generated 2026-03-07 01:12 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/i57pl0ua)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `i57pl0ua` |
| **Status** | crashed |
| **Started** | 2026-01-30T05:06:38Z |
| **Steps** | 98 |
| **env_config** | `act_prm/taubench_retail_500_fs1` |
| **eval_env_config** | `None` |
| **model_config** | `hf_qwen3_4b_inst_2507` |
| **lora_config** | `r8_a16_qkvo` |
| **trainer_config** | `aprm_for_sft100` |
| **learning_rate** | `4e-05` |
| **mini_batch_size** | `None` |
| **gradient_accumulation_steps** | `None` |
| **seed** | `42` |
| **replicate** | `0` |
| **group_size** | `4` |
| **hide_observations** | `True` |
| **actions_only** | `True` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| eval/final_reward | 0.454505 |
| eval/final_reward_best | 0.454505 |
| eval/try_0/accuracy | 0 |
| eval/try_0/action_prob | 0.454505 |
| eval/try_0/action_prob_max | 0.999922 |
| eval/try_0/action_prob_std | 0.282786 |
| eval/try_0/correct | 0 |
| eval/try_0/correct_max | False |
| eval/try_0/correct_std | 0 |
| eval/try_0/final_reward | 0.454505 |
| eval/try_0/final_reward_max | 0.999922 |
| eval/try_0/final_reward_std | 0.282786 |
| eval/try_0/first_return | 0.454505 |
| eval/try_0/first_return_max | 0.999922 |
| eval/try_0/first_return_std | 0.282786 |
| eval/try_0/last_state_len | 1459.393617 |
| eval/try_0/last_state_len_max | 2960 |
| eval/try_0/last_state_len_std | 656.280188 |
| eval/try_0/timesteps | 1 |
| eval/try_0/timesteps_max | 1 |
| eval/try_0/timesteps_std | 0 |
| eval/try_0/total | 94 |
| eval/try_0/total_max | 1 |
| eval/try_0/total_std | 0 |
| optim/lr | 0.000040 |
| time/run_evals | 390.162901 |
| time/total | 388.258895 |
| train/advantage | 0.667303 |
| train/loss | 0.308760 |
| train/num_gen_tokens | 52 |
| train/ppl | 1.554688 |
| train/try_0/accuracy | 0 |
| train/try_0/action_prob | 0.492100 |
| train/try_0/action_prob_max | 0.999828 |
| train/try_0/action_prob_std | 0.254544 |
| train/try_0/correct | 0 |
| train/try_0/correct_max | False |
| train/try_0/correct_std | 0 |
| train/try_0/final_reward | 0.492100 |
| train/try_0/final_reward_max | 0.999828 |
| train/try_0/final_reward_std | 0.254544 |
| train/try_0/first_return | 0.492100 |
| train/try_0/first_return_max | 0.999828 |
| train/try_0/first_return_std | 0.254544 |
| train/try_0/last_state_len | 1426.097561 |
| train/try_0/last_state_len_max | 2805 |
| train/try_0/last_state_len_std | 613.236548 |
| train/try_0/timesteps | 1 |
| train/try_0/timesteps_max | 1 |
| train/try_0/timesteps_std | 0 |
| train/try_0/total | 164 |
| train/try_0/total_max | 1 |
| train/try_0/total_std | 0 |

## Training Curves

### Loss

![Loss](plots/loss.png)

### Reward

![Reward](plots/reward.png)

### Eval / Final Reward Best

![Eval / Final Reward Best](plots/eval_final_reward_best.png)

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

### Optim / Lr

![Optim / Lr](plots/optim_lr.png)

### Time / Run Evals

![Time / Run Evals](plots/time_run_evals.png)

### Time / Total

![Time / Total](plots/time_total.png)

### Train / Advantage

![Train / Advantage](plots/train_advantage.png)

### Train / Num Gen Tokens

![Train / Num Gen Tokens](plots/train_num_gen_tokens.png)

### Train / Ppl

![Train / Ppl](plots/train_ppl.png)

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
