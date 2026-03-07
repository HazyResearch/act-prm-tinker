# Training Report: act-lm-taubench-airline-gt-mt4k-rgen1_sl_al-s0

> Auto-generated 2026-03-07 01:11 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/sggsxb9a)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `sggsxb9a` |
| **Status** | crashed |
| **Started** | 2026-01-27T20:04:29Z |
| **Steps** | 8342 |
| **env_config** | `act_lm/taubench_airline_gt_mt4k` |
| **eval_env_config** | `None` |
| **model_config** | `hf_qwen3_4b_inst_2507` |
| **lora_config** | `r8_a16_qkvo` |
| **trainer_config** | `pt_sft_gen5` |
| **learning_rate** | `0.001` |
| **mini_batch_size** | `32` |
| **gradient_accumulation_steps** | `32` |
| **seed** | `0` |
| **replicate** | `gen1_sl_al` |
| **group_size** | `None` |
| **hide_observations** | `True` |
| **actions_only** | `None` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| eval/best_ppl | 1.516918 |
| eval/eval_idx | 21 |
| eval/gen_longest_per_task | 0.087500 |
| eval/gen_success_per_task | 0 |
| eval/nll | 0.416680 |
| eval/ppl | 1.516918 |
| eval/probs | 0.659232 |
| eval/step_act_acc | 0.268293 |
| eval/task_longest | 0.075000 |
| eval/task_success | 0 |
| train/loss | 1.640625 |
| train/ppl | 5.156250 |
| train/weight | 0.656100 |

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
