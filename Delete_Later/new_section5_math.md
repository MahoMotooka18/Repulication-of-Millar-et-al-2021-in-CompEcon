---
title: "Section 5 Math Notes"
paper: "Deep learning for solving dynamic economic models (Maliar, Maliar, Winant, 2021, JME)"
scope: "Section 5 — Krusell–Smith (1998) model"
language: "English"
---

# Section 5 Math Notes (All Equations)

Note: Equation numbers follow the paper’s Section 5 labels where available.
Symbols are reconstructed where the screenshot is ambiguous, with replication
notes to clarify implementation.

## 5.1 Krusell–Smith (1998) model

### (37) Household objective
$$
\max_{\{c^i_t,k^i_{t+1}\}_{t\ge 0}}\;\mathbb{E}_0\left[\sum_{t=0}^{\infty}\beta^t u(c^i_t)\right],
\qquad i=1,\ldots,\ell.
$$

### (38) Cash-on-hand transition
$$
w^i_{t+1}=R_{t+1}(w^i_t-c^i_t)+W_{t+1}\exp(y^i_{t+1}).
$$

### (39) Borrowing constraint
$$
c^i_t\le w^i_t.
$$

### (40) Idiosyncratic productivity
$$
y^i_{t+1}=\rho_y y^i_t+\sigma_y\varepsilon^i_{t+1},\qquad \varepsilon^i_t\sim\mathcal{N}(0,1).
$$

### (41) Aggregate productivity
$$
z_{t+1}=\rho_z z_t+\sigma_z\epsilon_{t+1},\qquad \epsilon_t\sim\mathcal{N}(0,1).
$$

### (42) Prices and aggregation
$$
R_t=1-d+\alpha z_t k_t^{\alpha-1}\Big[\frac{1}{\ell}\sum_{i=1}^{\ell}\exp(y^i_t)\Big],
\qquad
W_t=z_t(1-\alpha)k_t^{\alpha}\Big[\frac{1}{\ell}\sum_{i=1}^{\ell}\exp(y^i_t)\Big],
$$
with aggregate capital $k_t=\sum_{i=1}^{\ell}k^i_t$ and $k^i_{t+1}=w^i_t-c^i_t$.
The paper notes that (38) implies $w^i_t=R_t k^i_t+W_t\\exp(y^i_t)$.

Calibration reported in the text:
- $u(c)=(c^{1-\gamma}-1)/(1-\gamma)$ with $\gamma=1$,
- $\beta=0.96$, $\rho_z=0.95$, $\sigma_z=0.01$,
- $\rho_y=0.9$, $\sigma_y=0.2(1-\rho_y^2)^{1/2}$.

## 5.2 Deep learning solution algorithm

### State space
The full state includes all individual states and the aggregate shock:
$$
\{y^i_t,w^i_t\}_{i=1}^{\ell},\; z_t.
$$
Because agents are homogeneous in fundamentals, one decision rule is learned
and applied symmetrically to all agents.

### Parameterization
Consumption share, multiplier, and value are modeled as
$$
\frac{c^i_t}{w^i_t}=\sigma\big(\zeta_0+\eta(y^i_t,w^i_t,D_t,z_t;\vartheta)\big)\equiv\varphi(\cdot;\theta),
$$
$$
h^i_t=\exp\big(\zeta_0+\eta(y^i_t,w^i_t,D_t,z_t;\vartheta)\big)\equiv h(\cdot;\theta),
$$
$$
V^i_t=\zeta_0+\eta(y^i_t,w^i_t,D_t,z_t;\vartheta)\equiv V(\cdot;\theta),
$$
where $D_t=\{y^i_t,w^i_t\}_{i=1}^{\ell}$ denotes the distribution of individual
states and $\sigma(x)=1/(1+e^{-x})$.

Replication notes:
- Baseline NN: two hidden layers with 64 and 64 neurons and sigmoid activation.
- Initialize $\zeta_0=0$; weights/biases drawn from He/Glorot uniform
  distributions (as stated in the text).
- In this section the paper keeps explicit time subscripts (no recursive
  representation) because the model is solved on simulated data.

### Simulation steps (per training iteration)
Given current state $\{w^i_t,y^i_t\}_{i=1}^{\ell}, z_t$:
1) Compute $c^i_t/w^i_t=\varphi(y^i_t,w^i_t,D_t,z_t;\theta)$ and set
   $k^i_{t+1}=w^i_t-c^i_t$ for all $i$.
2) Draw $y^i_{t+1}$ and $z_{t+1}$ from (40) and (41).
3) Compute $R_{t+1}$ and $W_{t+1}$ from (42).
4) Compute next cash-on-hand $w^i_{t+1}=R_{t+1}k^i_{t+1}+W_{t+1}\exp(y^i_{t+1})$.
5) Evaluate the chosen objective (lifetime reward, Euler, or Bellman) and
   update $\theta$.

Training details reported:
- Simulate for $K=300{,}000$ periods; train every 10th period.
- Use 100 simulated points per iteration and ADAM with $\lambda_k=0.001$.

## 5.3 Objective 1: Lifetime reward

### (43) Lifetime reward objective
$$
\Xi(\theta)=\mathbb{E}_{\omega}\left[\sum_{t=0}^{T}\beta^t u(c^i_t)\right],
\quad \omega=(y_0,w_0,z_0,\Sigma,\varepsilon).
$$
- $\Sigma=(\varepsilon^1_1,\ldots,\varepsilon^{\ell}_1,\ldots,\varepsilon^1_T,\ldots,\varepsilon^{\ell}_T)$
  stacks idiosyncratic shocks; $\varepsilon=(\epsilon_1,\ldots,\epsilon_T)$ are
  aggregate shocks.
- The paper uses TensorFlow “muting” so each agent’s objective is differentiated
  only w.r.t. that agent’s variables (competitive equilibrium training).

Replication note:
- Because random variables are autocorrelated, the stochastic gradient is
  biased; the paper trains on cross-sections sufficiently separated in time
  to mitigate this bias.

## 5.4 Objective 2: Euler equation with Kuhn–Tucker conditions

### (44) Euler objective with AiO operator
$$
\Xi(\theta)=\mathbb{E}_{\omega}\Bigg[\Psi^{FB}\Big(1-\tfrac{c^i_t}{w^i_t},\;1-h^i_t\Big)^2
+\nu\Big(\tfrac{\beta R_{t+1}u'(c^i_{t+1})}{u'(c^i_t)}\big|_{\Sigma=\Sigma_1,\epsilon=\epsilon_1}-h^i_t\Big)
\Big(\tfrac{\beta R_{t+1}u'(c^i_{t+1})}{u'(c^i_t)}\big|_{\Sigma=\Sigma_2,\epsilon=\epsilon_2}-h^i_t\Big)\Bigg],
$$
where $\Psi^{FB}(a,h)=a+h-\sqrt{a^2+h^2}$ and
$\omega=(Y_t,W_t,z_t,\Sigma_1,\Sigma_2,\epsilon_1,\epsilon_2)$ with
$Y_t=(y^1_t,\ldots,y^{\ell}_t)$ and $W_t=(w^1_t,\ldots,w^{\ell}_t)$.

Replication notes:
- Use two independent draws $(\Sigma_1,\epsilon_1)$ and $(\Sigma_2,\epsilon_2)$
  to implement the AiO product.
- This objective parallels the consumption–saving case in Section 4.

## 5.5 Objective 3: Bellman equation

### (45) Bellman objective with AiO operator
$$
\Xi(\theta)=\mathbb{E}_{\omega}\Big[\big(R_{\Sigma_1,\epsilon_1}\big)\big(R_{\Sigma_2,\epsilon_2}\big)
+\nu\,\Psi^{FB}\Big(1-\tfrac{c^i_t}{w^i_t},\;1-h^i_t\Big)^2
+\nu_h\big(G_{\Sigma_1,\epsilon_1}\big)\big(G_{\Sigma_2,\epsilon_2}\big)\Big],
$$
where
$$
R_{\Sigma,\epsilon}=V(s^i_t;\theta)-u(c^i_t)-\beta V(s^i_{t+1};\theta)\big|_{\Sigma,\epsilon},
$$
$$
G_{\Sigma,\epsilon}=\frac{\beta\,\partial_{w^i_{t+1}}V(s^i_{t+1};\theta)}{u'(c^i_t)}\bigg|_{\Sigma,\epsilon}-h^i_t,
$$
and $s^i_t=(y^i_t,w^i_t,D_t,z_t)$.

Replication notes:
- The paper pre-trains the value function for 100,000 iterations with initial
  decision rules fixed, explaining the initial flat region in the loss plot.

## 5.6 Comparison of three methods
The paper compares the decision rule, individual wealth simulation, and
aggregate capital paths across lifetime reward, Euler, and Bellman methods.
Differences are small and mostly due to stochastic optimization noise.
Replication note:
- Decision rules are compared holding all other individual and aggregate
  variables at steady state; simulations use identical shock sequences.

## 5.7 Properties of the solution and KS regression

### (46) KS-style regression
$$
\ln(k_{t+1})=\xi_0+\xi_1\ln(k_t)+\xi_2\ln(z_t).
$$
- The reported $R^2$ is high (near 1), consistent with strong approximate
  aggregation; the paper notes differences from the original KS results due
  to using a continuum of aggregate states rather than two regimes.

## 5.8 Reduced state space (moments)
To reduce cost, replace $D_t$ with low-dimensional moments $m_t$.
If only the first moment is used, the individual state reduces to
$$
(y^i_t,w^i_t,m_t,z_t).
$$
The DL approach can incorporate more moments or other statistics without the
KS regression step; with few moments (e.g., 10–20), the paper can scale to
large panels (up to 10,000 agents) with similar results and figures.
