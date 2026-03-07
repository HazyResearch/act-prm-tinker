# Training Report: act-lm-snorkel-finance-aprm-fs1-45-rgen0_sl-s0

> Auto-generated 2026-03-07 01:10 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/45vt7o71)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `45vt7o71` |
| **Status** | crashed |
| **Started** | 2026-01-26T21:58:19Z |
| **Steps** | 6686 |
| **env_config** | `act_lm/snorkel_finance_aprm_fs1_45` |
| **eval_env_config** | `None` |
| **model_config** | `hf_qwen3_4b_inst_2507` |
| **lora_config** | `r16_a32_qkvo` |
| **trainer_config** | `pt_sft_gen5` |
| **learning_rate** | `4e-05` |
| **mini_batch_size** | `None` |
| **gradient_accumulation_steps** | `None` |
| **seed** | `0` |
| **replicate** | `gen0_sl` |
| **group_size** | `None` |
| **hide_observations** | `True` |
| **actions_only** | `None` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| eval/best_ppl | 2.020177 |
| eval/eval_idx | 12 |
| eval/gen_longest_per_task | 1.622222 |
| eval/gen_success_per_task | 0 |
| eval/nll | 0.703874 |
| eval/ppl | 2.021570 |
| eval/probs | 0.494665 |
| eval/step_act_acc | 0.317728 |
| eval/task_longest | 0 |
| eval/task_success | 0 |
| train/loss | 0.589844 |
| train/ppl | 1.804688 |
| train/weight | 0.951803 |

## Training Curves

### Loss

![Loss](plots/loss.png)

### Eval / Best Ppl

![Eval / Best Ppl](plots/eval_best_ppl.png)

### Eval / Eval Idx

![Eval / Eval Idx](plots/eval_eval_idx.png)

### Eval / Gen Longest Per Task

![Eval / Gen Longest Per Task](plots/eval_gen_longest_per_task.png)

### Eval / Gen Success Per Task

![Eval / Gen Success Per Task](plots/eval_gen_success_per_task.png)

### Eval / Nll

![Eval / Nll](plots/eval_nll.png)

### Eval / Ppl

![Eval / Ppl](plots/eval_ppl.png)

### Eval / Probs

![Eval / Probs](plots/eval_probs.png)

### Eval / Step Act Acc

![Eval / Step Act Acc](plots/eval_step_act_acc.png)

### Eval / Task Longest

![Eval / Task Longest](plots/eval_task_longest.png)

### Eval / Task Success

![Eval / Task Success](plots/eval_task_success.png)

### Train / Ppl

![Train / Ppl](plots/train_ppl.png)

### Train / Weight

![Train / Weight](plots/train_weight.png)
