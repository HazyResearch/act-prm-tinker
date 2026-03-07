# TextWorld Environment

## Default Setup

Following [BALROG](https://github.com/balrog-ai/BALROG/tree/main/balrog/environments/textworld) and [Intelligent Go-Explore](https://github.com/conglu1997/intelligent-go-explore/tree/main), we use pre-generated games for the `coin_collector`, `the_cooking_game`, and `treasure_hunter` tasks.

First create a directory for the game files (which should be the same as `dataset_config.cache_dir` in the `configs/environments/textworld/*.yaml` environment config). 

Then, download the game files from [https://github.com/conglu1997/intelligent-go-explore/tree/main/textworld/tw_games](https://github.com/conglu1997/intelligent-go-explore/tree/main/textworld/tw_games). BALROG provides a convenient Google Drive link, so we can set things up in a single command:

```bash
mkdir /scr/mzhang/data/textworld && cd /scr/mzhang/data/textworld  # change to your own directory
curl -L -o tw-games.zip 'https://drive.google.com/uc?export=download&id=1aeT-45-OBxiHzD9Xn99E5OvC86XmqhzA'
unzip tw-games.zip
```

## Custom Task Generation

We can also generate custom TextWorld tasks via the `generate_tasks.py` script. See section-by-section example commands below.

### Dependencies patch

For cooking game, we may need to do a package-level patch for newer numpy versions. 

Currently, in `.venv/lib/python3.12/site-packages/textworld/challenges/tw_cooking/cooking.py`
L1016 may have something like:

```python
nb_distractors = abs(int(rng_objects.randn(1) * 3 + nb_ingredients))
```

With newer numpy versions, this will raise a scalar issue. We should instead change it to:

```python
nb_distractors = abs(int(rng_objects.randn() * 3 + nb_ingredients))
```

### More notes on level specification

#### Coin Collector

Paper's Difficulty Modes (Section 5.1)

┌───────────┬───────────────────────────┬───────────────┬────────────────┐
│   Mode    │        Distractors        │  Total Rooms  │   TextWorld    │
│           │                           │               │  Level Range   │
├───────────┼───────────────────────────┼───────────────┼────────────────┤
│ Easy      │ 0 off-chain rooms         │ 1 ×           │ 1–100          │
│ (mode 0)  │                           │ quest_length  │                │
├───────────┼───────────────────────────┼───────────────┼────────────────┤
│ Medium    │ 1 distractor per room on  │ 2 ×           │ 101–200        │
│ (mode 1)  │ the chain (dead ends)     │ quest_length  │                │
├───────────┼───────────────────────────┼───────────────┼────────────────┤
│ Hard      │ 2 distractors per room,   │ 3 ×           │ 201–300        │
│ (mode 2)  │ randomly placed           │ quest_length  │                │
└───────────┴───────────────────────────┴───────────────┴────────────────┘

The paper uses "level" to mean quest length (L5, L10, L15, L20, L25, L30), and
the TextWorld --level flag encodes both quest length and mode:

```python
TextWorld level = (n_distractors * 100) + quest_length
```

So for `quest_length=10`: `easy=10`, `medium=110`, `hard=210`.

Generation Commands

The paper tested quest lengths `{5, 10, 15, 20, 25, 30}`. Here are commands for
10 games at each level and mode:

```bash
# Quest lengths 5, 10, 15, 20, 25
# --- EASY (no distractors) ---
for QL in 5 10 15 20 25; do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/easy/ \
    --task coin_collector --n 20 --seed_start 0 --level $QL
done

# --- MEDIUM (1 distractor per chain room, simple dead ends) ---
for QL in 5 10 15 20 25; do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/medium/ \
    --task coin_collector --n 20 --seed_start 0 --level $((100 + QL))
done

# --- HARD (2 distractors per chain room, randomly placed) ---
for QL in 5 10 15 20 25; do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/hard/ \
    --task coin_collector --n 20 --seed_start 0 --level $((200 + QL))
done
```

For example, `--level 210` generates a hard game with `quest_length=10`, `30` total
rooms, and `randomly placed` distractors — matching the paper's "L10, hard"
setting.

### Only Last Action

```bash
# Quest lengths 5, 10, 15, 20, 25
# --- EASY (no distractors) ---
for QL in 5 10 15 20 25; do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/easy_last_act/ \
    --task coin_collector --n 20 --seed_start 0 --level $QL --only_last_action
done

# --- MEDIUM (1 distractor per chain room, simple dead ends) ---
for QL in 5 10 15 20 25; do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/medium_last_act/ \
    --task coin_collector --n 20 --seed_start 0 --level $((100 + QL)) --only_last_action
done

# --- HARD (2 distractors per chain room, randomly placed) ---
for QL in 5 10 15 20 25; do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/hard_last_act/ \
    --task coin_collector --n 20 --seed_start 0 --level $((200 + QL)) --only_last_action
done
```

#### Treasure Hunter

Treasure Hunter levels range from 1 to 30, split into three tiers:

Level Breakdown

┌───────┬────────┬───────┬──────────────┬─────────────────────────────────┐
│ Level │  Mode  │ Rooms │ Quest Length │           Description           │
├───────┼────────┼───────┼──────────────┼─────────────────────────────────┤
│       │        │       │ 1 → 5        │ Empty rooms, no doors, 0        │
│ 1–10  │ Easy   │ 5     │ (linspace,   │ distractors                     │
│       │        │       │ rounded)     │                                 │
├───────┼────────┼───────┼──────────────┼─────────────────────────────────┤
│       │        │       │ 2 → 10       │ Closed doors & containers (may  │
│ 11–20 │ Medium │ 10    │ (linspace,   │ need opening), 10 distractor    │
│       │        │       │ rounded)     │ containers/supporters           │
├───────┼────────┼───────┼──────────────┼─────────────────────────────────┤
│       │        │       │ 3 → 20       │ Locked doors & containers (keys │
│ 21–30 │ Hard   │ 20    │ (linspace,   │  placed in inventory), 20       │
│       │        │       │ rounded)     │ distractor                      │
│       │        │       │              │ containers/supporters           │
└───────┴────────┴───────┴──────────────┴─────────────────────────────────┘

The quest length within each tier is computed via `np.round(np.linspace(start, end, 10))`, so the exact values are:

- Easy (levels 1–10): quest lengths [1, 1, 2, 2, 3, 3, 3, 4, 4, 5]
- Medium (levels 11–20): quest lengths [2, 3, 4, 4, 5, 6, 7, 7, 8, 10]
(rounding of `linspace(2,10,10)`)
- Hard (levels 21–30): quest lengths [3, 5, 7, 8, 10, 12, 13, 15, 17, 20]
(rounding of `linspace(3,20,10)`)

Unlike coin collector (which is just navigation), treasure hunter quests chain
actions like `go`, `open`, `unlock`, and `take`. There's also a wrong object —
picking it up loses the game.

Generation Commands

```bash
# --- EASY (levels 1-10) ---
for LVL in $(seq 1 10); do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/easy/ \
    --task treasure_hunter --n 10 --seed_start 0 --level $LVL
done

# --- MEDIUM (levels 11-20) ---
for LVL in $(seq 11 20); do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/medium/ \
    --task treasure_hunter --n 10 --seed_start 0 --level $LVL
done

# --- HARD (levels 21-30) ---
for LVL in $(seq 21 30); do
  uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
    --cache_dir ./data/textworld/hard/ \
    --task treasure_hunter --n 10 --seed_start 0 --level $LVL
done

Or if you want a single representative level per mode (e.g. mid-range difficulty within each tier):

# Easy (level 5): 5 rooms, quest length ~3
uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir ./data/textworld/easy/ \
--task treasure_hunter --n 100 --seed_start 0 --level 5

# Medium (level 15): 10 rooms, quest length ~5
uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir ./data/textworld/medium/ \
--task treasure_hunter --n 100 --seed_start 0 --level 15

# Hard (level 25): 20 rooms, quest length ~10
uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir ./data/textworld/hard/ \
--task treasure_hunter --n 100 --seed_start 0 --level 25
```

#### The Cooking Game

Easy / Medium / Hard

There's no official easy/medium/hard, but the First TextWorld Problems
competition and the generate script in this repo (generate_tasks.py:19-21)
suggest a reasonable progression as below.

Note that as in Intelligent-Go-Explore (where BALROG gets its `the_cooking_game` tasks), 
we set `take` to be the same as `recipe` [by default](https://github.com/conglu1997/intelligent-go-explore/blob/821ad194080a30b1df7055fc6250cf45ccfcb477/textworld/misc/make_cooking.py#L51).

```bash
# Easy — few ingredients, small map, no processing:
uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir ./data/textworld/easy/ \
--task cooking_game --n 100 --seed_start 0 \
--recipe 1 --go 6 --cook

# Medium — more ingredients, larger map, cooking and cutting:
uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir ./data/textworld/medium/ \
--task cooking_game --n 100 --seed_start 0 \
--recipe 2 --go 6 --open --cook --cut

# Hard — many ingredients, large map, all skills:
uv run python ./src/act_prm/environments/textworld/generate_tasks.py \
--cache_dir ./data/textworld/hard/ \
--task cooking_game --n 100 --seed_start 0 \
--recipe 3 --go 12 --open --cook --cut --drop
```