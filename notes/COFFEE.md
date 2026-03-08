# Project: The "Coffee Experiment"

This document summarizes the **"Coffee Experiment"** (Chess Environment), as transcribed from project whiteboards.

**Subject:** Reasoning in a Chess Environment
**Goal:** Validate if learning to "think" (generate reasoning traces $\tau$) from demonstrations outperforms standard baselines.

### 1. Hypothesis
Learning to think from demonstrations of chess actions will perform better than:
1.  **SFT (Action-only):** Fine-tuning on the action dataset alone.
2.  **Standard RL:** Pure Reinforcement Learning on the Chess environment.
3.  **Matched SFT:** Fine-tuning on the "full" thought-action dataset without iterative refinement.

### 2. Proxy & Metrics
* **Primary Metric:** Performance of the model trained via different methods (Win Rate / Elo).
* **Evaluation Set:** Held-out / New games.

### 3. Implementation Protocol
**Step 1: Environment Implementation**
* Implement `ChessEnv`.
* Interface: Returns `state`, `action`, `next_observation`.

**Step 2: Observed Data Collection**
* Use a teacher model (`Qwen-3-[Size]`) to play chess games.
* Store trajectory: `action`, `thought` ($\tau$), and game metadata.

**Step 3: Experiment Setup**
* Train base model (e.g., `Qwen-3-4B`) using collected thoughts ($\tau$) and actions.
* Define parameters and hyperparameters.

### 4. Expected Results
We anticipate the following outcomes:

**A. Method Comparison (Bar Chart)**
* **Y-Axis:** Win Rate / Performance.
* **X-Axis:**
    * SFT (Action Only)
    * Standard RL
    * **ACT-PRM (Ours)** $\leftarrow$ *Expected highest performance*
    * ACT + Thought

**B. Training Dynamics (Line Chart)**
* **Y-Axis:** Reward ($R$) or $P(action)$.
* **X-Axis:** Training Iterations.
* **Trend:** Steady improvement in probability/reward over time.