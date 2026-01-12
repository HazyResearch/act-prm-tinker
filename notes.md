# Action Process Reward Models (Action-PRMs)

Michael Zhang (last updated 1/7/2026)

---

Building and hill-climbing on RL environments is great. But what happens if our workflows are too long and our rewards too sparse? In this note, we'll explore one alternative: using easy-to-collect, offline demonstrations of past successful trajectories (e.g., the actions recorded in a process log when a human employee currently has to do the task), to guide our LLMs along the way. 

While supervised finetuning (SFT) on these observed actions alone may not provide enough learning signal, our key idea is that this recorded sequence only represents half the picture. Missing from this context are a human's *implicit* or internal thoughts between the recorded actions, which may provide the connecting context needed to predict future explicit actions.

As a result, we'll explore how to use LLMs to *infer* these thoughts from the actions---or at least generate useful analogues---to fill in this missing context. We motivate our design under two perspectives:

* As **Reinforcement Learning (RL)**, we can train LLMs to generate useful thoughts preceding an observed action, using the subsequent perplexity or likelihood over the observed action as a dense, "unbiased" process reward.  

* As **Expectation-Maximization (EM)**, we can train LLMs to learn the most likely **latent thoughts** behind the *observed actions* in a successful workflow trajectory.

Rather than have humans arduously write these thoughts down (or assume that everyone has an internal monologue), we can then train LLMs to more effectively complete tasks from easy-to-collect, action-only demonstrations.


<!-- how to use cheap demonstrations of successful trajectories (e.g., process logs of humans doing the task already), to guide our LLMs along the way.  -->

<!-- Can we use cheap, demonstrations of successful trajectories (e.g., via human process logs), even in *off-line* manners to guide our LLMs along the way? -->

---

## Intro + Problem

Towards turning LLMs into useful "agents", we'd like to provide scalable ways to train these models to complete long and tedious tasks---ones that humans might begrudgingly do today. Mechanically, this can involve having an LLM think and take actions (e.g., generate tool calls, reply to users) iteratively over multiple steps. Then, the final outcome of these actions determines a successful task completion or not.

To do this, building environments and training with RL is one option. After all, these tasks are *technically verifiable* (did the LLM complete the task or not?). However, completing long multi-turn workflows can introduce several challenges, especially for the popular policy gradient methods of today:

* **Sparse rewards / credit assignment**. While successful task completion is one true reward signal, verifying success is only possible at the end. For long multi-turn workflows, this single sparse reward exacerbates a classic ***credit assignment*** problem. Here, it can be difficult to determine which actions contributed to the final outcome (one poor action resulting in failure can noisily penalize perfectly reasonable prior actions).

<!-- * **Designing "unbiased" proxy rewards**.  -->
* **Providing informative intermediate signal**. Alternatively, we could try to *engineer* or *shape* reward functions to provide more frequent reward signals, e.g., based on the immediate observations following an action. However, as these rewards only serve as intermediate proxies for the final outcome, they can often be *hacked*. Designing "unbiased" functions where local rewards align with final success remains its own [tedious challenge](https://archives.argmin.net/2018/04/16/ethical-rewards/#:~:text=Shaping%20rewards%20is%20hard).

* **Maintaining scalable training**. Finally, towards automating a large variety of useful yet undesirable tasks, we'd like to be able to easily implement + deploy our training for arbitrary workflows, e.g., by reducing the amount of human effort needed to provide training signals. Existing reward-based systems---even when automated (e.g., rubric-based LLM-as-judges, or process reward models)---require additional manual design or training data curation. Can we instead have methods where collecting this training signal aligns with humans already having to do the task? 

### SFT / BC as an alternative?

Meanwhile, because we are *aiming to distill* a human's ability to complete a workflow into an LLM, we could consider supervised finetuning (SFT) over sequences of expert demonstrations, *i.e.,* *behavioral cloning* (BC). 

By collecting successful demonstrations of a human completing a task, and simply training an LLM over text-based representations of these trajectories, we can ostensibly provide dense next-token prediction signals by simply having a human worker complete the task. 

However, doing so raises its own challenges.

1. Remaining faithful to the "full trajectory"---and recording not just actions but a human's chain-of-thoughts and reasoning steps---introduces additional annotation effort separate from task completion. This can be difficult to scale, limiting effectiveness especially when (1) BC often relies on having large amounts of training data, and (2) our training data size is directly capped by the number of collected samples.

2. Meanwhile, limiting this data to action-only demonstrations can remove this annotation burden, but may not provide informative-enough learning signal. Consider a task that involves navigating a series of web pages. We can record the sequence of actions automatically as events in a process log (e.g., URLs visited, coordinates clicked, etc.). But without the surrounding context motivating the sequence of events, predicting the next action can be challenging.

---

## Our Approach

Instead, we'll consider an alternative approach, where we aim to train LLMs to generate thoughts and actions, from easy-to-collect, action-only demonstrations.

### Setup - multi-turn RL

To ground our approach, we start with the following multi-turn RL setup. For each timestep $t$, our model processes as input context a state $s_t$, generates an action $a_t$, and receives a reward $r_t$. We then transition to the next state $s_{t+1}$ via a next-timestep observation $o_{t+1}$. We use the simple case where new states are a concatenation of the prior state, action, reward, and new observation:
$$
\begin{aligned}
s_1 &= (o_1,) \\
s_2 &= (o_1, a_1, r_1, o_2) \\
&\ldots \\
s_{t} &= (o_1, a_1, r_1, \ldots, a_{t-1}, r_{t-1}, o_t)
\end{aligned}
$$

and let trajectory $\tau$ be a complete sequence of observations, actions, and rewards, associated with a task completion (or failure), *i.e.,* 
$$
\tau = (o_1, a_1, r_1, o_2, a_2, r_2, \ldots, o_T, a_T, r_T, o_{T+1})
$$
for last timestep $T$. By default, we tie rewards to task completion (success or failure at the final timestep), so
$$
r_t = \begin{cases}
1 & \text{if } t = T \text{ and } \tau \text{ succeeds} \\
0 & \text{otherwise}
\end{cases}
$$

Concretely, we can think of:

* $s_1 = o_1$ as the system prompt, initial task instruction, and any additional environment context,
* $a_1$ as the text generated by the model in response (*i.e.,* thoughts and explicit actions ($z_1, x_1$)),
* $o_2$ as some feedback or response from the environment (tool call response or user message),  
* $s_2$ as just the concatenation of these prior components (*e.g.,* represented as the processed list of dicts following a Hugging Face Transformers chat template),

with similar instantiations for subsequent $t \in [T]$.

**Tying objectives to task success**  

Like most RL setups, for any timestep $t$ we want our model to generate actions that maximize the expected *return* $J_t$, or (discounted) sum of future rewards. Considering our sparse rewards, no discounting, and letting $R_t$ and $\tau_t$ be the return and trajectory at starting at $t$, we have
$$
\begin{aligned}
R_t &= r_t + r_{t+1} + \ldots + r_T \\
J_t &= \mathbb{E}_{\tau_t \sim p_\theta(\tau_t)} \left[ R_t \right] \\
% J_t &= \mathbb{E} \left[R_t \right] \\
&= \mathbb{E}_{\tau_t \sim p_\theta(\tau_t)} \Big[ \sum_{t'=t}^T r_{t'} \Big] \\
&= \mathbb{E}_{\tau_t \sim p_\theta(\tau_t)} \Big[ \mathbb{1}(\tau_{t} \text{ succeeds}) \Big] \\
&= P(\tau_t \text{ succeeds})
\end{aligned}
$$
In other words, at each timestep $t$, given everything the model's seen before, we want to generate actions that maximize the probability of a successful task completion.

---

### Training to infer thoughts given successful demonstrations

Now, we'll back out an RL algorithm for our setting. 

To recap, we're given access to a sequence of *explicit actions* $x_t$ and outcomes $o_{t+1}$ from successful human task completions, i.e.,
$$
\tau \text{ where } p(\tau_\text{succeeds} \mid o_1, a_1, \ldots, o_T, a_T) = 1
$$
However, for each action, rather than the full $a_t = (z_t, x_t)$, we only observe $a_t = (x_t,)$, resulting in "partial" trajectories:
$$
% \tau^x = o_1, (\cdot, x_1), \ldots, o_T, (\cdot, x_T)
\dot{\tau} = (o_1, \cdot\;, x_1, \ldots, o_T, \cdot\;, x_T)
$$

We could SFT LLMs over $\dot{\tau}$, as generating the explicit actions $x_t$ is enough to reach a success state. However, just $x_1, \ldots, x_t$ alone may not provide enough context to predict $x_{t+1}$. 

So we're interested in learning to generate the best interweaving inferred thoughts $\hat{z}_1, \hat{z}_2, \ldots, \hat{z}_T$ to fill in the complete trajectory:
$$
\hat{\tau} = (o_1, \hat{z}_1, x_1, \ldots, o_T, \hat{z}_T, x_T)
$$
where best is determined by the $z$ that make the observed sequence of observations and explicit actions $\hat{\tau}$ most likely.

#### Finding rewards by maximizing our expectations

To find $z$, we'll first connect this objective to Expectation Maximization (EM), *i.e.,* finding the latent thoughts that make the observed sequence of explicit actions $x$ most likely. We'll then describe how we can implement this as policy gradient RL, using a likelihood-based reward function over these actions.

**EM Setup**  

Recall that EM iterates over two eponymous steps:

1. **Expectation (E)**: where we define $p_{\theta^{(n)}}(z \mid x, s)$ as a posterior over the latent thought $z$ given the explicit action $x$, current state $s$ and latest model weights $\theta^{(n)}$; and use this to write an expectation over the *complete action* log-likelihood:

$$
\mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid x, s)} \left[ \log p_\theta(x, z \mid s) \right]
$$

2. **Maximization (M)**: where we update $\theta$ to maximize our expectation:

$$
\theta^{(n + 1)} = \arg \max_\theta\; \mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid x, s)} \left[ \log p_\theta(x, z \mid s) \right]
$$

We'll iterate between **E** and **M** until convergence, *i.e.,* finding LLM parameters $\theta^*$ and thoughts $z$ that maximize the likelihood of observed successful human actions $x$. 

Following this framing in practice:

* In **E**, we'll sample many potential thoughts $z$ from the current state and LLM (*generating or sampling rollouts*). We use these samples to compute rewards, and construct a training objective (*the policy gradient*).

* In **M**, we'll update the LLM weights according to this objective.

To make this concrete, we'll next show how to get the rewards and training objective from our EM framing.

**From expectation maximization to policy gradient**  

Our goal is to show an equivalence between the maximization equation above and policy gradient with certain rewards. We can then use this to motivate a practical training algorithm from the EM framework. 

Recall that our expectation is defined as
$$
% \begin{aligned}
% \mathbb{E}_{z \sim p(z \mid x, s, \theta^{(n)})} \left[ \log p(x, z \mid s, \theta) \right]
% &= \int p(z \mid x, s, \theta^{(n)}) \log  p(x, z \mid s, \theta) \;dz
% \end{aligned}
\begin{aligned}
\mathbb{E}_{z \sim \color{blue}{p_{\theta^{(n)}}(z \mid x, s)}} \left[ \log p_\theta(x, z \mid s) \right]
&= \int \color{blue}{p_{\theta^{(n)}}(z \mid x, s)} \log  p_\theta(x, z \mid s) \;dz
\end{aligned}
$$

From Bayes' Rule, we have
$$
p_{\theta^{(n)}}(z \mid x, s) = \frac{p_{\theta^{(n)}}(x \mid z, s) \cdot p_{\theta^{(n)}}(z \mid s)}{p_{\theta^{(n)}}(x \mid s)}
$$

So plugging this in gives us
$$
\begin{aligned}
\mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid x, s)} \left[ \log p_\theta(x, z \mid s) \right]
&= \int \frac{p_{\theta^{(n)}}(x \mid z, s) \cdot \color{blue}{p_{\theta^{(n)}}(z \mid s)} }{p_{\theta^{(n)}}(x \mid s)} \log  p_\theta(x, z \mid s) \;dz \\
&= \int \color{blue}{p_{\theta^{(n)}}(z \mid s)} \cdot \Big( \frac{p_{\theta^{(n)}}(x \mid z, s)}{p_{\theta^{(n)}}(x \mid s)} \log  p_\theta(x, z \mid s) \Big) \;dz \\
&= \mathbb{E}_{z \sim \color{blue}{p_{\theta^{(n)}}(z \mid s)}} \left[ \frac{p_{\theta^{(n)}}(x \mid z, s)}{p_{\theta^{(n)}}(x \mid s)} \log  p_\theta(x, z \mid s) \right]
\end{aligned}
$$

Recall that we're trying to maximize this expectation via $\theta$. So we can write our objective as 
$$
\begin{aligned}
J(\theta) 
&= \mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid s)} \left[ \frac{p_{\theta^{(n)}}(x \mid z, s)}{p_{\theta^{(n)}}(x \mid s)} \log  p_\theta(x, z \mid s) \right] \\
\Rightarrow
\nabla_\theta J(\theta) 
&= \mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid s)} \left[ \frac{p_{\theta^{(n)}}(x \mid z, s)}{p_{\theta^{(n)}}(x \mid s)} \nabla_\theta \log  p_\theta(x, z \mid s) \right]
\end{aligned}
$$

This starts to look *a lot* like a policy gradient objective. Define reward $r(z) = \dfrac{p_{\theta^{(n)}}(x \mid z, s)}{p_{\theta^{(n)}}(x \mid s)}$, and we're there!

* As a brief interpretation note, $r(z)$ already hints at what we're rewarding. Ignoring the normalizing denominator for now, we score a given thought $z$ by how likely it is that the observed explicit action $x$ follows.

**Practical Empirics**

<!-- At this point, we still need to resolve a couple empirical computations.  -->
<!-- Finally, we'll resolve a couple empirical computations to back-out a concrete training algorithm.  -->
Finally, we'll derive a sampling-based version of the above to back-out a concrete training algorithm.

<!-- First, while we can't compute $p_\theta() -->
Let's first define the *unnormalized* reward $\tilde{r}(z) = p_{\theta^{(n)}}(x \mid z, s)$. Then, making the marginalization explicit in our reward's denominator:
$$
\begin{aligned}
p_{\theta^{(n)}} (x \mid s)
&= \int p_{\theta^{(n)}} (x \mid z, s)\; p_{\theta^{(n)}}(z \mid s) \; dz \\
&= \mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid s)} \Big[ p_{\theta^{(n)}} (x \mid z, s) \Big] \\
&= \mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid s)} [ \tilde{r}(z)]
\end{aligned}
$$

Now in both the reward denominator and our maximization objective, we're computing expectations over $p_{\theta^{(n)}}(z \mid s)$, *i.e.,* the latent thought distribution parameterized by our LLM in a given state. We can then estimate both $p_{\theta^{(n)}} (x \mid s)$ and $J(\theta)$ by sampling $G$ thoughts per state, and using this same batch of generations to compute these quantities as empirical means:

$$
\begin{aligned}
p_{\theta^{(n)}} (x \mid s)
&\approx \frac{1}{G} \sum_{g=1}^G \tilde{r}(z^{(g)}) \\ 

\nabla_\theta J(\theta) 
&= \mathbb{E}_{z \sim p_{\theta^{(n)}}(z \mid s)} \left[ \frac{\color{magenta}{p_{\theta^{(n)}}(x \mid z, s)}}{\color{blue}{p_{\theta^{(n)}}(x \mid s)}} \nabla_\theta \log  p_\theta(x, z^{(g)} \mid s) \right] \\

&\approx \frac{1}{G} \sum_{g=1}^G \frac{\color{magenta}{\tilde{r}(z^{(g)})}}{\color{blue}{\frac{1}{G} \sum_{g'=1}^G \tilde{r}(z_{g'})}} \nabla_\theta \log  p_\theta(x, z^{(g)} \mid s) \\

&\approx \sum_{g=1}^G \frac{\tilde{r}(z^{(g)})}{\sum_{g'=1}^G \tilde{r}(z_{g'})} \nabla_\theta \log  p_\theta(x, z^{(g)} \mid s)
\end{aligned}
$$

So with empirical reward $\bar{r}(z^{(g)})$, 
$$
\bar{r}(z^{(g)}) 
= \frac{\tilde{r}(z^{(g)})}{\sum_{g'=1}^G \tilde{r}(z_{g'})}
= \dfrac{p_{\theta^{(n)}}(x \mid z^{(g)}, s)}{ \sum_{g'=1}^G p_{\theta^{(n)}} (x \mid z_{g'}, s)}
$$
we recover the policy gradient
$$
\nabla_\theta J(\theta) \approx \sum_{g=1}^G \bar{r}(z^{(g)}) \nabla_\theta \log p_\theta(x, z^{(g)} \mid s)
$$

---

### Practical Implementation (it's just policy gradient)

In practice, we can implement this as a standard RL algorithm. 

**Generation (Expectation Step)**

1. Starting with the first prompt state, we first sample thoughts $z^{(g)}_t$ from an LLM given the current state $s_t$.
2. We then compute a per-timestep reward $r_t$ based on the LLM's likelihood over the given explicit action $x_t$, given the prior context $(s_t, z^{(g)}_t)$.
3. We then pick the highest-reward thought $z^{(g)}_t)$, and transition to the next state $s_{t+1} = (s_t, z^{(g)}_t, x_t, r_t, o_{t+1})$.
4. We repeat this process until we reach the final timestep $T$, saving each of these trajectory steps for training next.

**Policy Update (Maximization Step)**  

We then train our LLM with standard policy gradient methods, *i.e.,* 
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau_t \sim p_\theta(\tau_t)} \left[ \nabla_\theta R_t \log p_\theta(x_t, z_t^{(g)} \mid s_t) \right]
$$
where $R_t$ is some function of the original likelihoods $\tilde{r}(z_t^{(g)} ) = p(x_t | z_t^{(g)} , s_t)$.

While the EM math above tells us one option, we can consider a couple different choices:

1. Even though we have a multi-turn trajectory, at each step we may aim to generate thoughts that primarily maximize the likelihood of that step's observed action. In this case, a contextual bandits setup may suffice:

    * $R_t = \bar{r}(z_t^{(g)} ) = \dfrac{p_{\theta^{(n)}}(x \mid z_t^{(g)}, s)}{ \sum_{g'=1}^G p_{\theta^{(n)}} (x \mid z_t^{(g')}, s)}$  

    * $R_t = \tilde{r}(z_t^{(g)} ) = p_{\theta^{(n)}}(x \mid z_t^{(g)}, s)$ (unnormalized rewards may work too)

2. We can also consider the proper return as a sum of the individual rewards: 

    * $R_t = \sum_{t'=t}^T \bar{r}(z_t^{(g)} )$ or $R_t = \sum_{t'=t}^T \tilde{r}(z_t^{(g)})$  

    This may be motivated since earlier thoughts do add to the context and impact the generation of later ones.  

3. Orthogonally, instead of using the raw likelihoods as rewards, we can do some group-based normalization such as in GRPO. In practice, this may help by giving us *negative* returns, allowing us to actively downweight the worse half of thoughts in a batch.
    * For example, after sampling multiple potential thoughts in step 2., we subtract the mean of the group from each likelihood:  
    $
    \tilde{r}_\text{grpo}(z_t^{(g)}) = \tilde{r}(z_t^{(g)}) - \frac{1}{G}\sum_{g' = 1}^{G} \tilde{r}(z_t^{(g')})
    $