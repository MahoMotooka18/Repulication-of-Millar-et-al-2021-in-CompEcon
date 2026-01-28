---
title: "Section 3 Math Notes"
paper: "Deep learning for solving dynamic economic models (Maliar, Maliar, Winant, 2021, JME)"
scope: "Section 3 — Deep learning solution method"
language: "English"
---

# Section 3 Math Notes (All Equations)

Note: Some symbols are corrupted by PDF extraction, so I reconstruct them
in standard notation consistent with the paper’s definitions. When a symbol
is ambiguous in the screenshot, I keep the equation correct and clarify the
role in words.

## 3.1 Unified expectation-form objective

### (16) Expected objective
$$
\min_{\theta\in\Theta}\; \mathcal{L}(\theta)
= \min_{\theta\in\Theta}\; \mathbb{E}_{\omega}\big[\xi(\omega;\theta)\big].
$$
- $\omega=(m,s,\varepsilon)$ (or a longer stack including future shocks)
  collects the random variables used by the objective.
- $\xi(\omega;\theta)$ embeds Euler/Bellman equations, constraints, etc.
- Solving the model becomes minimizing a single objective.

### (17) Empirical risk (sample average)
$$
\min_{\theta\in\Theta}\; \mathcal{L}_n(\theta)
= \min_{\theta\in\Theta}\; \frac{1}{n}\sum_{i=1}^{n}\xi(\omega_i;\theta).
$$
- Approximate the expectation with $n$ simulated draws.
- Unlike standard regression, the draws are re-sampled during training.

### Algorithm 1: DL algorithm for solving dynamic economic models
Initialize:
1) Construct theoretical risk $\mathcal{L}(\theta)=\mathbb{E}_{\omega}[\xi(\omega;\theta)]$
   (lifetime reward, Euler, or Bellman objective).
2) Define empirical risk $\mathcal{L}_n(\theta)=\frac{1}{n}\sum_{i=1}^n\xi(\omega_i;\theta)$.
3) Choose a neural-network topology for $\varphi(\cdot;\theta)$ (and $V(\cdot;\theta)$
   if the Bellman objective is used).
4) Initialize $\theta$.

Train (repeat until convergence):
1) Simulate the model to generate a fresh minibatch $\{\omega_i\}_{i=1}^n$
   using the current policy.
2) Compute the gradient $\nabla_{\theta}\mathcal{L}_n(\theta)$.
3) Update parameters $\theta \leftarrow \theta - \lambda_k \nabla_{\theta}\mathcal{L}_n(\theta)$.
4) Stop if $\|\theta^{k+1}-\theta^{k}\|$ is small.

Assess accuracy on an out-of-sample simulation batch.

## 3.2 Geometric intuition for the curse of dimensionality

### (18) Volume ratio: hypersphere vs hypercube
$$
V_d =
\begin{cases}
\dfrac{(\pi/2)^{(d-1)/2}}{1\cdot3\cdot5\cdots d}, & d=1,3,5,\ldots \\
\dfrac{(\pi/2)^{d/2}}{2\cdot4\cdot6\cdots d}, & d=2,4,6,\ldots
\end{cases}
$$
- $V_d$ is the volume of a $d$-sphere relative to a $d$-cube.
- $V_d$ shrinks rapidly with $d$, explaining grid inefficiency.

### Replication notes on data generation
- The paper stresses simulating data where the solution lives: the ergodic
  set implied by the current policy. This avoids wasting samples in regions
  never visited in equilibrium.
- In practice: simulate long paths, discard burn-in, then sample states
  from the ergodic distribution to build training minibatches.

## 3.3 Gradient optimization of expectation objectives

### (19) Mini-batch gradient descent update
$$
\theta^{k+1} \leftarrow \theta^{k} - \lambda_k\,\frac{1}{n}\sum_{i=1}^{n}\nabla_{\theta}\,\xi(\omega_i;\theta^{k}).
$$
- Approximate the expectation gradient by a sample average.
- With $n=1$, this becomes stochastic gradient descent (SGD).

### Replication notes on AiO integration and SGD
- AiO integration operator: when the objective contains nested expectations
  (e.g., Euler residuals with multiple shocks), draw two independent shock
  vectors and use their product. This reduces integration to just two draws
  per state even with many shocks.
- SGD extreme: set $n=1$ in (19), which evaluates the gradient at one random
  point. The paper uses ADAM as a practical optimizer.

## 3.4 How DL differs from conventional projection
The standard projection approach (e.g., Judd, 1992) uses tensor-product
polynomials on a fixed grid and quadrature for expectations. This becomes
computationally expensive as:
- the grid size grows exponentially with the number of state variables,
- the quadrature nodes grow exponentially with the number of shocks,
- the number of coefficients increases with grid density,
- ill-conditioning worsens in high dimension.

The DL approach mitigates this by:
- simulating only along the ergodic set,
- using very few random integration nodes (AiO),
- solving all model equations in a single objective,
- relying on GPU-parallelized autodiff and SGD.

## 3.5 Neural network approximation (high level)
The decision rule (and value function, if needed) is represented by a neural
network with an input layer for state variables, one or more hidden layers,
and an output layer producing choices. Nonlinear activations (e.g., sigmoid
or tanh) allow flexible approximation beyond polynomials.

### Replication notes on network design
- Inputs: concatenate all state variables in a consistent order used by the
  model equations and transitions.
- Outputs: map to feasible choices (e.g., via softplus/logistic transforms
  or manual projection onto $X(m,s)$).
- Start with a small architecture (1–2 hidden layers) and increase width only
  if residuals plateau.
