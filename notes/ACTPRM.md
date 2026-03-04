# Project: ACT-PRM 

This document summarizes the theoretical framework for **ACT-PRM**, as transcribed from project whiteboards.

### 1. Typical RL (Baseline)
In standard Reinforcement Learning for generation, the process is modeled as:
* **Input:** Query ($Q$)
* **Generation:** The model generates a sequence of states ($O$) and actions ($a$).
    * $Q \rightarrow (O_1, a_1) \rightarrow (O_2, a_2) \dots$
* **Reward:** A terminal reward is assigned based on the final output (Binary: Correct/Incorrect).

### 2. ACT-PRM (Proposed Method)
ACT-PRM acts as a "Process Reward Model" that explicitly models the relationship between reasoning ("thought") and action.

**Core Mechanism:**
The probability ratio is defined as:

$$
\frac{R_1}{R_2} = P(\text{action} \mid \text{thought}, \text{state})
$$

**Data Definitions:**
We distinguish between two types of Supervised Fine-Tuning (SFT) data structures:
* **Full Chain ($SFT$):** Includes the reasoning trace.
    * Format: $Q, \hat{\text{thought}}, \text{action}, \dots$
* **Action-Only ($SFT_{act}$):** Includes only the actions.
    * Format: $Q, \text{action}_1, \text{action}_2, \dots$

### 3. Iterative Training Loop
The core training loop (for $i$ in $MAX\_ROUNDS$):

1.  **Sample:** Sample thoughts/reasoning traces from the current model $\theta_i$.
2.  **Generate:** Create a new SFT dataset based on these samples.
3.  **Train:** Train policy $\phi$ on the new SFT dataset to update parameters to $\phi'$.
4.  **Evaluate:** Run evaluation of $\phi'$ on the RL Evaluation Set.

---
