# Training Report: act-lm-snorkel-finance-gt-mt4k-rgen3_sl_al_eg5-s0

> Auto-generated 2026-03-07 01:13 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/vnvedgny)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `vnvedgny` |
| **Status** | crashed |
| **Started** | 2026-02-02T01:08:42Z |
| **Steps** | 18573 |
| **env_config** | `act_lm/snorkel_finance_gt_mt4k` |
| **eval_env_config** | `None` |
| **model_config** | `hf_qwen3_8b` |
| **lora_config** | `r8_a16_qkvo` |
| **trainer_config** | `pt_sft_gen5` |
| **learning_rate** | `0.001` |
| **mini_batch_size** | `32` |
| **gradient_accumulation_steps** | `32` |
| **seed** | `0` |
| **replicate** | `gen3_sl_al_eg5` |
| **group_size** | `None` |
| **hide_observations** | `True` |
| **actions_only** | `True` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| actions_data_gen_save_path | ./logs/act_lm_snorkel_finance_gt_mt4k/Qwen3_8B/act-prm-tinker-isas=0-reru=0-enco=act_lm_snorkel_finance_gt_mt4k-trco=pt_sft_gen5-moco=hf_qwen3_8b-loco=r8_a16_qkvo-gracst=32-mainidle=2048-acon=1-hiob=1-lera=0_001-mibasi=32-nuevgesa=100-se=0-re=gen3_sl_al_eg5/eval_gen_action_metrics.csv |
| actions_data_lm_save_path | ./logs/act_lm_snorkel_finance_gt_mt4k/Qwen3_8B/act-prm-tinker-isas=0-reru=0-enco=act_lm_snorkel_finance_gt_mt4k-trco=pt_sft_gen5-moco=hf_qwen3_8b-loco=r8_a16_qkvo-gracst=32-mainidle=2048-acon=1-hiob=1-lera=0_001-mibasi=32-nuevgesa=100-se=0-re=gen3_sl_al_eg5/eval_lm_action_metrics.csv |
| eval/gen_longest | 1.058824 |
| eval/gen_success | 0 |
| eval/lm_best_ppl | 1.913314 |
| eval/lm_longest | 0 |
| eval/lm_nll | 0.649760 |
| eval/lm_ppl | 1.915081 |
| eval/lm_probs | 0.522171 |
| eval/lm_success | 0 |
| train/loss | 0.150391 |
| train/ppl | 1.164062 |
| train/weight | 0.590490 |

## Training Curves

### Loss

![Loss](plots/loss.png)

### Actions Data Gen Save Path

![Actions Data Gen Save Path](plots/actions_data_gen_save_path.png)

### Actions Data Lm Save Path

![Actions Data Lm Save Path](plots/actions_data_lm_save_path.png)

### Eval / Gen Longest

![Eval / Gen Longest](plots/eval_gen_longest.png)

### Eval / Gen Success

![Eval / Gen Success](plots/eval_gen_success.png)

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

### Train / Ppl

![Train / Ppl](plots/train_ppl.png)

### Train / Weight

![Train / Weight](plots/train_weight.png)
