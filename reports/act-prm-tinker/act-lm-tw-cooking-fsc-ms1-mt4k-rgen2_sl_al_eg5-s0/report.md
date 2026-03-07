# Training Report: act-lm-tw-cooking-fsc-ms1-mt4k-rgen2_sl_al_eg5-s0

> Auto-generated 2026-03-07 01:13 UTC from [W&B run](https://wandb.ai/hazy-research/act-prm-tinker/runs/tjfbkbnu)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `tjfbkbnu` |
| **Status** | crashed |
| **Started** | 2026-02-02T20:00:56Z |
| **Steps** | 373 |
| **env_config** | `act_lm/tw_cooking_fsc_ms1_mt4k` |
| **eval_env_config** | `textworld/cooking_game` |
| **model_config** | `hf_qwen3_4b_inst_2507` |
| **lora_config** | `r8_a16_qkvo` |
| **trainer_config** | `pt_sft_gen5` |
| **learning_rate** | `0.001` |
| **mini_batch_size** | `32` |
| **gradient_accumulation_steps** | `32` |
| **seed** | `0` |
| **replicate** | `gen2_sl_al_eg5` |
| **group_size** | `None` |
| **hide_observations** | `True` |
| **actions_only** | `True` |

## Latest Metrics

| Metric | Value |
|--------|-------|
| actions_data_lm_save_path | ./logs/act_lm_tw_cooking_fsc_ms1_mt4k/Qwen3_4B_Instruct_2507/act-prm-tinker-isas=0-reru=0-enco=act_lm_tw_cooking_fsc_ms1_mt4k-trco=pt_sft_gen5-mona=Qwen3_4B_Instruct_2507-moco=hf_qwen3_4b_inst_2507-loco=r8_a16_qkvo-gracst=32-acon=1-hiob=1-evenco=tw_cooking_game-lera=0_001-mibasi=32-se=0-re=gen2_sl_al_eg5/eval_lm_action_metrics.csv |
| eval/lm_best_ppl | 1.865579 |
| eval/lm_longest | 0 |
| eval/lm_nll | 0.623572 |
| eval/lm_ppl | 1.865579 |
| eval/lm_probs | 0.536027 |
| eval/lm_success | 0 |
| train/loss | 0.660156 |
| train/ppl | 1.937500 |
| train/weight | 0.999954 |

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

### Train / Ppl

![Train / Ppl](plots/train_ppl.png)

### Train / Weight

![Train / Weight](plots/train_weight.png)
