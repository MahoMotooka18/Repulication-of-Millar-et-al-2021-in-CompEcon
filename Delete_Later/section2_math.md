---
title: "Section 2 Math Notes"
paper: "Deep learning for solving dynamic economic models (Maliar, Maliar, Winant, 2021, JME)"
scope: "Section 2 — Casting dynamic economic models into DL expectation functions"
language: "English"
---

# Section 2 Math Notes (All Equations)

Note: Some symbols are corrupted by PDF extraction, so I reconstruct them
in standard notation consistent with the paper’s definitions. When a symbol
is ambiguous in the screenshot, I keep the equation correct and state the
role of the object in words.

## 2.1 Model setup

### (1) Exogenous state transition
$$
m_{t+1} = M(m_t, \varepsilon_t).
$$
- $m_t \in \mathbb{R}^{n_m}$ is the exogenous state.
- $\varepsilon_t$ is an i.i.d. shock.
- $M(\cdot)$ is the Markov transition.

### (2) Endogenous state transition
$$
s_{t+1} = S(m_t, s_t, x_t, m_{t+1}).
$$
- $s_t \in \mathbb{R}^{n_s}$ is the endogenous state.
- $x_t \in \mathbb{R}^{n_x}$ is the choice vector.
- Next state depends on $(m_t, s_t, x_t, m_{t+1})$.

### (3) Feasible set
$$
x_t \in X(m_t, s_t).
$$
- State-dependent feasibility set.

### (4) Lifetime reward maximization
$$
\max_{\{x_t, s_{t+1}\}_{t\ge 0}}\; \mathbb{E}_0\left[\sum_{t=0}^{\infty} \beta^t\, r(m_t, s_t, x_t)\right].
$$
- $r(\cdot)$ is period reward and $\beta\in[0,1)$ is the discount factor.
- $\mathbb{E}_0$ is conditional on $(m_0,s_0)$.

## 2.2 Objective 1: Lifetime-reward maximization

### (5) Value function
$$
V(m_0,s_0) \equiv \max_{\{x_t,s_{t+1}\}_{t\ge 0}}\; \mathbb{E}_{(\varepsilon_1,\ldots)}\left[\sum_{t=0}^{\infty} \beta^t r(m_t,s_t,x_t)\right].
$$
- Maximum attainable expected lifetime reward from $(m_0,s_0)$.

### (6) Finite-horizon approximation
$$
V^T(m_0,s_0;\theta) \equiv \mathbb{E}_{(\varepsilon_1,\ldots,\varepsilon_T)}\left[\sum_{t=0}^{T} \beta^t r\big(m_t,s_t,\varphi(m_t,s_t;\theta)\big)\right].
$$
- Truncate the infinite horizon at $T<\infty$.
- $\varphi(\cdot;\theta)$ is a parametric decision rule.
Note: The paper uses $V^T$ (superscript) for the truncated-horizon value.

### (7) Randomized initial state objective
$$
J(\theta) \equiv \mathbb{E}_{(m_0,s_0)}\,\mathbb{E}_{(\varepsilon_1,\ldots,\varepsilon_T)}\left[\sum_{t=0}^{T} \beta^t r\big(m_t,s_t,\varphi(m_t,s_t;\theta)\big)\right].
$$
- Randomizing $(m_0,s_0)$ targets accuracy over a broader domain.
- This introduces nested expectations.
Note: The paper denotes this objective by a script/Greek symbol (not fully
legible in the screenshot). Here I label it $J(\theta)$.

### (8) AiO (All-in-One) expectation
$$
J(\theta) = \mathbb{E}_{\omega}[\xi(\omega;\theta)]
= \mathbb{E}_{(m_0,s_0,\varepsilon_1,\ldots,\varepsilon_T)}\left[\sum_{t=0}^{T} \beta^t r\big(m_t,s_t,\varphi(m_t,s_t;\theta)\big)\right].
$$
- Draw $\omega=(m_0,s_0,\varepsilon_1,\ldots,\varepsilon_T)$ jointly.
- This collapses nested expectations into a single one.

### Replication notes for Objective 1
- Choose a distribution for $(m_0,s_0)$ over the domain where accuracy is
  desired (e.g., stationary distribution or a broad uniform grid box).
- Pick a truncation horizon $T$ large enough that $\beta^T$ is negligible.
- For each parameter vector $\theta$, simulate $N$ Monte Carlo paths:
  1) Draw $(m_0,s_0)$, then $\varepsilon_{1:T}$,
  2) Generate $(m_{t+1},s_{t+1})$ using (1)-(2) and decisions
     $x_t=\varphi(m_t,s_t;\theta)$,
  3) Accumulate $\sum_{t=0}^T \beta^t r(m_t,s_t,x_t)$,
  4) Average across the $N$ paths to estimate $J(\theta)$.
- Optimize $\theta$ with SGD/Adam; gradients can be obtained by automatic
  differentiation through the simulation code if it is differentiable.

## 2.3 Objective 2: Euler-residual minimization

### (9) Euler equations (generic form)
$$
\mathbb{E}_{\varepsilon}\big[f_j(m,s,x,m',s',x')\big]=0,\quad j=1,\ldots,J.
$$
- $m' = M(m,\varepsilon)$, $s' = S(m,s,x,m')$, $x' = \varphi(m',s')$.
- $f_j$ encodes FOCs, equilibrium conditions, etc.

### (10) Weighted sum of squared Euler residuals
$$
\mathcal{L}(\theta) \equiv \mathbb{E}_{(m,s)}\left[\sum_{j=1}^{J} v_j\left(\mathbb{E}_{\varepsilon}\, f_j\big(m,s,\varphi(m,s;\theta),m',s',\varphi(m',s';\theta)\big)\right)^2\right].
$$
- $v_j$ are weights on optimality conditions.
- The inner expectation is squared, so it cannot be merged directly.

### (11) Product identity with independent shocks
$$
\mathbb{E}_{\varepsilon_1}[f(\varepsilon_1)]\,\mathbb{E}_{\varepsilon_2}[f(\varepsilon_2)]
= \mathbb{E}_{(\varepsilon_1,\varepsilon_2)}[f(\varepsilon_1)f(\varepsilon_2)].
$$
- Use independent draws to rewrite the squared expectation as a product.

### (12) AiO Euler-residual objective
$$
\mathcal{L}(\theta)=\mathbb{E}_{(m,s,\varepsilon_1,\varepsilon_2)}\Bigg[\Big(\sum_{j=1}^{J} v_j f_j(\cdot)\big|_{\varepsilon=\varepsilon_1}\Big)
\Big(\sum_{j=1}^{J} v_j f_j(\cdot)\big|_{\varepsilon=\varepsilon_2}\Big)\Bigg].
$$
- Two independent shocks produce the AiO product.

### Replication notes for Objective 2
- Sample $(m,s)$ from a distribution that covers the relevant state region.
- For each draw, generate two independent shocks $\varepsilon_1,\varepsilon_2$.
- Evaluate the Euler residuals using the same parametric policy
  $x=\varphi(m,s;\theta)$ and its implied next-period choices.
- Average the product in (12) across Monte Carlo draws.

## 2.4 Objective 3: Bellman-residual minimization

### (13) Bellman equation
$$
V(m,s)=\max_{x,s'}\left\{ r(m,s,x) + \beta\,\mathbb{E}_{\varepsilon}\big[V(m',s')\big] \right\},
$$
- $m'=M(m,\varepsilon)$, $s'=S(m,s,x,m')$, $x\in X(m,s)$.
- Standard dynamic programming fixed point.

#### Handling the maximization operator
Because the Bellman residual contains a $\max$, replace it with one of:
- FOC (first-order condition)
- Envelope condition
- Direct maximization

### (14) Bellman residual + FOC objective
$$
\mathcal{L}(\theta) \equiv \mathbb{E}_{(m,s)}\Big[ V(m,s;\theta_1) - r(m,s,x) - \beta\,\mathbb{E}_{\varepsilon}V(m',s';\theta_1) \Big]^2
+ v\,\mathbb{E}_{(m,s)}\Big[ r_x(m,s,x) + \beta\,\mathbb{E}_{\varepsilon}\big( V_s(m',s';\theta_1)\,\tfrac{\partial s'}{\partial x}\big) \Big]^2.
$$
- $x=\varphi(m,s;\theta_2)$, $\theta=(\theta_1,\theta_2)$.
- First term is Bellman residual, second term is FOC residual.
- $v>0$ is a weight.

### (15) AiO Bellman-residual objective
$$
\mathcal{L}(\theta)=\mathbb{E}_{(m,s,\varepsilon_1,\varepsilon_2)}\Big[
\big(R_{\varepsilon_1}\big)\big(R_{\varepsilon_2}\big)
+ v\,\big(G_{\varepsilon_1}\big)\big(G_{\varepsilon_2}\big)\Big],
$$
$$
R_{\varepsilon} \equiv V(m,s;\theta_1)-r(m,s,x)-\beta V(m',s';\theta_1)\big|_{\varepsilon},
\quad
G_{\varepsilon} \equiv r_x(m,s,x)+\beta V_s(m',s';\theta_1)\big|_{\varepsilon}\,\tfrac{\partial s'}{\partial x}.
$$
- Draw $\varepsilon_1,\varepsilon_2$ independently to form the AiO product.
- As in the Euler case, this replaces nested expectations with one.

### Replication notes for Objective 3
- Parameterize both the value function $V(m,s;\theta_1)$ and policy
  $x=\varphi(m,s;\theta_2)$.
- If using the FOC approach, ensure $r_x$ and $V_s$ are computed consistently
  with the model’s state ordering and the transition $s'=S(m,s,x,m')$.
- Use two independent shock draws to estimate the AiO objective (15).
- When models have constraints, enforce them in $\varphi$ (e.g., via
  transformations) so that gradients are well-defined.
