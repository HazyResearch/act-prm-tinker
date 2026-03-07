# Training Report: act-lm-tw-cooking-fsc-ms1-mt4k-rgen1_sl_al_eg5-s0

> Auto-generated 2026-03-07 01:11 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/b3yf49d9)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `b3yf49d9` |
| **Status** | crashed |
| **Started** | 2026-01-28T18:42:05Z |
| **Steps** | 5121 |
| **env_config** | `act_lm/tw_cooking_fsc_ms1_mt4k` |
| **eval_env_config** | `textworld/cooking_game` |
| **model_config** | `hf_qwen3_4b_inst_2507` |
| **lora_config** | `r8_a16_qkvo` |
| **trainer_config** | `pt_sft_gen5` |
| **learning_rate** | `0.001` |
| **mini_batch_size** | `32` |
| **gradient_accumulation_steps** | `32` |
| **seed** | `0` |
| **replicate** | `gen1_sl_al_eg5` |
| **group_size** | `None` |
| **hide_observations** | `True` |
| **actions_only** | `None` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| actions_data_lm_save_path | ./logs/act_lm_tw_cooking_fsc_ms1_mt4k/Qwen3_4B_Instruct_2507/act-prm-tinker-isas=0-reru=0-enco=act_lm_tw_cooking_fsc_ms1_mt4k-trco=pt_sft_gen5-mona=Qwen3_4B_Instruct_2507-moco=hf_qwen3_4b_inst_2507-loco=r8_a16_qkvo-gracst=32-hiob=1-evenco=tw_cooking_game-lera=0_001-mibasi=32-se=0-re=gen1_sl_al_eg5/eval_lm_action_metrics.csv |
| eval/lm_best_ppl | 1.035206 |
| eval/lm_longest | 6 |
| eval/lm_nll | 0.034600 |
| eval/lm_ppl | 1.035206 |
| eval/lm_probs | 0.965992 |
| eval/lm_success | 0 |
| eval/ro_rollout_eval_eval/try_0/final_reward | 0 |
| eval/ro_rollout_eval_eval/try_0/final_reward_best | 0 |
| rollout_eval_eval/try_0/accuracy | 0 |
| rollout_eval_eval/try_0/action_prob | 0 |
| rollout_eval_eval/try_0/action_prob_max | 0 |
| rollout_eval_eval/try_0/action_prob_std | 0 |
| rollout_eval_eval/try_0/correct | 0 |
| rollout_eval_eval/try_0/correct_max | False |
| rollout_eval_eval/try_0/correct_std | 0 |
| rollout_eval_eval/try_0/final_reward | 0 |
| rollout_eval_eval/try_0/final_reward_max | 0 |
| rollout_eval_eval/try_0/final_reward_std | 0 |
| rollout_eval_eval/try_0/first_return | 0.233313 |
| rollout_eval_eval/try_0/first_return_max | 0.810000 |
| rollout_eval_eval/try_0/first_return_std | 0.303661 |
| rollout_eval_eval/try_0/last_state_len | 6943.400000 |
| rollout_eval_eval/try_0/last_state_len_max | 12262 |
| rollout_eval_eval/try_0/last_state_len_std | 2785.775411 |
| rollout_eval_eval/try_0/timesteps | 1 |
| rollout_eval_eval/try_0/timesteps_max | 1 |
| rollout_eval_eval/try_0/timesteps_std | 0 |
| rollout_eval_eval/try_0/total | 10 |
| rollout_eval_eval/try_0/total_max | 1 |
| rollout_eval_eval/try_0/total_std | 0 |
| train/loss | 1.304688 |
| train/ppl | 3.687500 |
| train/weight | 0.999975 |

## Training Curves

### Loss

![Loss](plots/loss.png)

### Actions Data Lm Save Path

![Actions Data Lm Save Path](plots/actions_data_lm_save_path.png)

### Eval / Lm Best Ppl

![Eval / Lm Best Ppl](plots/eval_lm_best_ppl.png)

### Eval / Lm Longest

![Eval / Lm Longest](plots/eval_lm_longest.png)

### Eval / Lm Nll

![Eval / Lm Nll](plots/eval_lm_nll.png)

### Eval / Lm Ppl

![Eval / Lm Ppl](plots/eval_lm_ppl.png)

### Eval / Lm Probs

![Eval / Lm Probs](plots/eval_lm_probs.png)

### Eval / Lm Success

![Eval / Lm Success](plots/eval_lm_success.png)

### Eval / Ro Rollout Eval Eval / Try 0 / Final Reward

![Eval / Ro Rollout Eval Eval / Try 0 / Final Reward](plots/eval_ro_rollout_eval_eval_try_0_final_reward.png)

### Eval / Ro Rollout Eval Eval / Try 0 / Final Reward Best

![Eval / Ro Rollout Eval Eval / Try 0 / Final Reward Best](plots/eval_ro_rollout_eval_eval_try_0_final_reward_best.png)

### Rollout Eval Eval / Try 0 / Accuracy

![Rollout Eval Eval / Try 0 / Accuracy](plots/rollout_eval_eval_try_0_accuracy.png)

### Rollout Eval Eval / Try 0 / Action Prob

![Rollout Eval Eval / Try 0 / Action Prob](plots/rollout_eval_eval_try_0_action_prob.png)

### Rollout Eval Eval / Try 0 / Action Prob Max

![Rollout Eval Eval / Try 0 / Action Prob Max](plots/rollout_eval_eval_try_0_action_prob_max.png)

### Rollout Eval Eval / Try 0 / Action Prob Std

![Rollout Eval Eval / Try 0 / Action Prob Std](plots/rollout_eval_eval_try_0_action_prob_std.png)

### Rollout Eval Eval / Try 0 / Correct

![Rollout Eval Eval / Try 0 / Correct](plots/rollout_eval_eval_try_0_correct.png)

### Rollout Eval Eval / Try 0 / Correct Max

![Rollout Eval Eval / Try 0 / Correct Max](plots/rollout_eval_eval_try_0_correct_max.png)

### Rollout Eval Eval / Try 0 / Correct Std

![Rollout Eval Eval / Try 0 / Correct Std](plots/rollout_eval_eval_try_0_correct_std.png)

### Rollout Eval Eval / Try 0 / Final Reward

![Rollout Eval Eval / Try 0 / Final Reward](plots/rollout_eval_eval_try_0_final_reward.png)

### Rollout Eval Eval / Try 0 / Final Reward Max

![Rollout Eval Eval / Try 0 / Final Reward Max](plots/rollout_eval_eval_try_0_final_reward_max.png)

### Rollout Eval Eval / Try 0 / Final Reward Std

![Rollout Eval Eval / Try 0 / Final Reward Std](plots/rollout_eval_eval_try_0_final_reward_std.png)

### Rollout Eval Eval / Try 0 / First Return

![Rollout Eval Eval / Try 0 / First Return](plots/rollout_eval_eval_try_0_first_return.png)

### Rollout Eval Eval / Try 0 / First Return Max

![Rollout Eval Eval / Try 0 / First Return Max](plots/rollout_eval_eval_try_0_first_return_max.png)

### Rollout Eval Eval / Try 0 / First Return Std

![Rollout Eval Eval / Try 0 / First Return Std](plots/rollout_eval_eval_try_0_first_return_std.png)

### Rollout Eval Eval / Try 0 / Last State Len

![Rollout Eval Eval / Try 0 / Last State Len](plots/rollout_eval_eval_try_0_last_state_len.png)

### Rollout Eval Eval / Try 0 / Last State Len Max

![Rollout Eval Eval / Try 0 / Last State Len Max](plots/rollout_eval_eval_try_0_last_state_len_max.png)

### Rollout Eval Eval / Try 0 / Last State Len Std

![Rollout Eval Eval / Try 0 / Last State Len Std](plots/rollout_eval_eval_try_0_last_state_len_std.png)

### Rollout Eval Eval / Try 0 / Timesteps

![Rollout Eval Eval / Try 0 / Timesteps](plots/rollout_eval_eval_try_0_timesteps.png)

### Rollout Eval Eval / Try 0 / Timesteps Max

![Rollout Eval Eval / Try 0 / Timesteps Max](plots/rollout_eval_eval_try_0_timesteps_max.png)

### Rollout Eval Eval / Try 0 / Timesteps Std

![Rollout Eval Eval / Try 0 / Timesteps Std](plots/rollout_eval_eval_try_0_timesteps_std.png)

### Rollout Eval Eval / Try 0 / Total

![Rollout Eval Eval / Try 0 / Total](plots/rollout_eval_eval_try_0_total.png)

### Rollout Eval Eval / Try 0 / Total Max

![Rollout Eval Eval / Try 0 / Total Max](plots/rollout_eval_eval_try_0_total_max.png)

### Rollout Eval Eval / Try 0 / Total Std

![Rollout Eval Eval / Try 0 / Total Std](plots/rollout_eval_eval_try_0_total_std.png)

### Train / Ppl

![Train / Ppl](plots/train_ppl.png)

### Train / Weight

![Train / Weight](plots/train_weight.png)
