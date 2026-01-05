
---

```md
---
title: "Replication Requirements — Section 5 (Krusell–Smith 1998 Model)"
paper: "Deep learning for solving dynamic economic models (Maliar, Maliar, Winant, 2021, JME)"
scope: "Section 5 — Numerical analysis of Krusell and Smith’s (1998) model"
language: "English"
---

# 0. Purpose

Build a **fully reproducible replication package** for Section 5, including:

- heterogeneous-agent Krusell–Smith economy,
- deep-learning solution under **three objectives**,
- long-run simulation with intermittent training,
- approximate aggregation evaluation.

This document is a **requirements specification for an LLM**.

---

# 1. Model Specification (Krusell–Smith)

## 1.1 Economy

- Agents $i = 1,\dots,I$ with states $(y^i_t, w^i_t)$
- Aggregate shock $z_t$
- Control: consumption $c^i_t$
- Household problem (Eq. 37):
$$
\max_{\{c^i_t,k^i_{t+1}\}} \; \mathbb{E}_0\sum_{t=0}^\infty \beta^t u(c^i_t)
$$
- Budget and constraint (Eqs. 38–39):
$$
w^i_{t+1} = R_{t+1} (w^i_t - c^i_t) + W_{t+1}\exp(y^i_{t+1})
$$
- Feasibility: $c^i_t \le w^i_t$
- Idiosyncratic productivity (Eq. 40):
$$
y^i_{t+1}=\rho_y y^i_t + \sigma_y \varepsilon^i_{t+1},\quad \varepsilon^i_t\sim\mathcal{N}(0,1)
$$
- Aggregate productivity (Eq. 41):
$$
z_{t+1}=\rho_z z_t + \sigma_z \epsilon_{t+1},\quad \epsilon_t\sim\mathcal{N}(0,1)
$$
- Prices depend on aggregates (capital, distribution).
- Production and prices (Eq. 42):
$$
Y_t=z_t K_t^\alpha L_t^{1-\alpha},\quad
R_t=1-d+\alpha z_t K_t^{\alpha-1}L_t^{1-\alpha},\quad
W_t=z_t(1-\alpha)K_t^\alpha L_t^{-\alpha}.
$$

## 1.2 Preferences and parameters

- CRRA utility with:
  - $\gamma = 1$
  - $\beta = 0.96$
- Technology:
  - capital share $\alpha$
  - depreciation $d \in (0,1)$
- Aggregate shock:
  - persistence $\rho_z = 0.95$
  - volatility $\sigma_z = 0.01$
- Idiosyncratic shock:
  - persistence $\rho_y = 0.9$
  - volatility $\sigma_y = 0.2\sqrt{1-\rho_y^2}$

All remaining parameters must be configurable and documented.

---

# 2. State Representation

## 2.1 Network inputs

- own state $(y^i_t, w^i_t)$,
- aggregate shock $z_t$,
- distribution vector $D_t = (y^1_t,\dots,y^I_t,w^1_t,\dots,w^I_t)$.

Implementation note:
- The baseline implementation feeds the **full concatenated distribution** directly into the network (no explicit permutation-invariant encoder).
- The state lists individual variables both as an agent’s own state and as part of $D_t$, which induces **perfect multicollinearity**; this is acceptable and matches the paper.

---

# 3. Outputs and Constraints

## 3.1 Consumption parameterization

- Network outputs share $s^i_t \in (0,1)$,
- Consumption:
$$
c^i_t = s^i_t \cdot w^i_t
$$

## 3.2 Additional outputs

- Nonnegative multiplier $h_t = \exp(\tilde h_t)$,
- Value function output $V_t \in \mathbb{R}$.
- Use the same transforms as Section 4:
  - $c^i_t/w^i_t=\sigma(\zeta_0+\eta(\cdot))$,
  - $h^i_t=\exp(\zeta_0+\eta(\cdot))$,
  - $V^i_t=\zeta_0+\eta(\cdot)$.

---

# 4. Neural Network Baseline

- Activation: sigmoid
- Architecture: $64 \times 64$
- Initialization: He / Glorot
- Optimizer: ADAM, learning rate $0.001$

---

# 5. Objectives

Implement all three:

1. **Lifetime reward objective**
2. **Euler-equation residual objective**
3. **Bellman residual objective**

Selectable via config.

## 5.1 Lifetime reward objective (Eq. 43)

Use AiO over $\omega=(Y_0,W_0,z_0,\Sigma,\epsilon)$:
$$
\Xi(\theta)=E_\omega\left[\sum_{t=0}^T \beta^t u(c_t)\right].
$$
Training detail: when optimizing agent $i$'s objective, **mute gradients** with respect to other agents’ variables to preserve competitive equilibrium.

## 5.2 Euler objective with Kuhn–Tucker conditions (Eq. 44)

Use the Fischer–Burmeister function with two independent shock draws:
$$
E_{Y_t,W_t,z_t,\Sigma_1,\Sigma_2,\epsilon_1,\epsilon_2}\left[
\Psi^{FB}\!\left(1-\frac{c^i_t}{w^i_t},1-h^i_t\right)^2
+\nu\left(\frac{\beta R_{t+1}u'(c^i_{t+1})}{u'(c^i_t)}\Big|_{\Sigma_1,\epsilon_1}-h^i_t\right)
\left(\frac{\beta R_{t+1}u'(c^i_{t+1})}{u'(c^i_t)}\Big|_{\Sigma_2,\epsilon_2}-h^i_t\right)
\right].
$$

## 5.3 Bellman objective (Eq. 45)

Use the Bellman residual and FB constraints with two independent shock draws:
$$
E_{Y_t,W_t,z_t,\Sigma_1,\Sigma_2,\epsilon_1,\epsilon_2}\Big[
\left(V(s^i_t)-u(c^i_t)-\beta V(s^i_{t+1})\big|_{\Sigma_1,\epsilon_1}\right)
\left(V(s^i_t)-u(c^i_t)-\beta V(s^i_{t+1})\big|_{\Sigma_2,\epsilon_2}\right)
+\nu\,\Psi^{FB}\!\left(1-\frac{c^i_t}{w^i_t},1-h^i_t\right)^2
+\nu_h(\cdot)
\Big],
$$
with the envelope-based term $\nu_h(\cdot)$ matching Eq. (45) in the paper.

## 5.4 Section 2 (casting into DL expectation functions) implementation notes

- **Definition 2.2**: Decision rules are **time-invariant** and evaluated over a fixed horizon $T$ for lifetime reward.
- **Definitions 2.3–2.4**: Lifetime-reward objective uses a **composite draw** $\omega = (m_0, s_0, \varepsilon_1,\ldots,\varepsilon_T)$ to collapse nested expectations into a single expectation.
- **Definitions 2.6–2.10**: Euler/Bellman objectives with squared residuals use **AiO with two independent shock draws** per step:
  $$
  E_\varepsilon[f(\varepsilon)]^2 \Rightarrow E_{\varepsilon^{(1)},\varepsilon^{(2)}}[f(\varepsilon^{(1)})f(\varepsilon^{(2)})].
  $$
- **Definitions 2.1, 2.5**: State vectors $(m_t,s_t)$ are sampled from the **target distribution** (not a fixed grid), consistent with the expectation-function formulation.

---

# 6. Simulation and Training Protocol

## 6.1 Simulation

- Total periods: $K = 300{,}000$
- State updates use next-period aggregates:
  - $k^i_{t+1} = w^i_t - c^i_t$,
  - $K_{t+1} = \sum_i k^i_{t+1}$,
  - $L_{t+1} = \sum_i \exp(y^i_{t+1})$,
  - compute $R_{t+1}, W_{t+1}$ from $(K_{t+1}, L_{t+1}, z_{t+1})$,
  - then $w^i_{t+1} = R_{t+1} k^i_{t+1} + W_{t+1} \exp(y^i_{t+1})$.
- Normalize idiosyncratic productivity each period so that $\frac{1}{I}\sum_i \exp(y^i_t)=1$.

## 6.1.1 Section 3 (DL solution method) implementation notes

- **Algorithm 1, Step 2i**: Simulate the model **on the ergodic set**; avoid fixed grids outside the region where the solution lives.
- **Algorithm 1, Step 2i–2ii** and **Equation (17)**: Re-sample training data each iteration (no fixed dataset); empirical risk is a **sample average**.
- **Equation (11)**: Use **two independent shock draws** for AiO when residuals are squared; only two integration nodes per iteration.
- **Algorithm 1, Step 2ii–2iii** and **Equation (19)**: Training is stochastic gradient descent (Adam), with gradients computed by automatic differentiation on sampled objectives.

## 6.2 Training schedule

- Train every $10$ periods,
- Each update uses $100$ simulated points.
- For Euler/Bellman objectives, compute $c_{t+1}$ using **next-period distribution inputs** $D_{t+1}$ (not $D_t$).
- Use Adam with learning rate $\lambda=0.001$.
- Bellman objective: pre-train the value function for the first $100{,}000$ iterations with consumption and multiplier fixed.
- To reduce bias from autocorrelated shocks, train on **cross-sections sufficiently separated in time** rather than consecutive periods.

## 6.3 AiO expectation operator (independence requirement)

- For Euler/Bellman objectives, draw two **independent** shock realizations $(\varepsilon^{(1)}, \varepsilon^{(2)})$.
- Do **not** reuse the same shock tensor within a batch for the two terms in the AiO product.
- Independence applies to both idiosyncratic and aggregate shocks.

---

# 7. Evaluation

Evaluation alignment:
- Euler residuals use $R_{t+1}$ matched to $c_{t+1}$ from the same transition step.

## 7.1 Approximate aggregation (KS regression)

Estimate:
$$
\ln k_{t+1} = \xi_0 + \xi_1 \ln k_t + \xi_2 \ln z_t
$$

Report coefficients and $R^2$.
The regression is run on aggregate capital $K_t$ and aggregate productivity $z_t$ over the evaluation window (no state-contingent split by default).
Note: Krusell and Smith (1998) estimate **state-contingent** regressions for good/bad states; our baseline uses a single regression as in Eq. (46).

## 7.2 Summary statistics

Export:
- wealth moments (mean, variance, percentiles, Gini) computed on **capital/asset holdings** $k^i_t = w^i_t - c^i_t$,
- aggregate moments for **output** $Y_t$ and **aggregate consumption** $C_t$,
- correlation statistics: $\mathrm{corr}(Y_t, C_t)$,
- reported $\mathrm{std}(y)$ corresponds to $\mathrm{std}(Y_t)$ (aggregate output).
- include runtime (seconds) for the full training run.

Table 1 column names must match the paper:
- `l`, `std(y)`, `corr(y,c)`, `Gini(k)`, `Bottom 40%`, `Top 20%`, `Top 1%`, `Time, sec.`, `R2`.

## 7.3 Plots

Plotting conventions (match paper style):
- Use log-scale on the training-iteration axis.
- Apply moving-average smoothing to training curves (window configurable via config).
- Keep the same layout across objectives for comparability.
- Panel 2 (consumption rule): plot $c(w)$ under **7 productivity levels** spanning $[-2, +2]$ standard deviations of productivity, fixing the aggregate state and all other agents’ productivities at their **steady-state** levels.
- Panel 3 (wealth simulation): plot wealth paths for **5 randomly selected agents** from the simulated cross-section.

Paper-matching panels:
- Fig. 10 (lifetime reward): losses, consumption rule, wealth simulation.
- Fig. 11 (Euler): log losses, consumption rule, wealth simulation.
- Fig. 12 (Bellman): log losses, value function under 7 productivity levels, consumption rule, wealth simulation.
- Fig. 13: decision-rule comparison across methods (single-agent slice), individual capital simulation, and aggregate capital simulation.

## 7.4 Debug logging (validation run)

For at least one run, log the following diagnostics:
- $\min(c)$, $\max(c)$, $\min(w)$, $\max(w)$, $\min(w_{t+1})$.
- Constraint violation rates: share$\{c<0\}$, share$\{c>w\}$, share$\{w<0\}$.
- AiO independence checks: confirm $\varepsilon^{(1)} \neq \varepsilon^{(2)}$ for idiosyncratic and aggregate shocks.

Save under `outputs/section5/`:
- aggregate time series,
- individual wealth paths,
- policy function slices,
- regression diagnostics.

---

# 8. Acceptance Criteria

A run is successful if:

- simulation is stable (no NaNs),
- KS regression has high $R^2$ (comparable to paper),
- all required artifacts are produced.

---

# 9. Engineering Requirements

- Config-driven execution:
```bash
python Lab_Section5_Krusell_and_Smith_1998/train_ks_experiment.py --config configs/section5.yaml
```

Config parsing note:
- YAML does not evaluate arithmetic; if values like `0.2 * sqrt(1 - 0.9**2)` are used, they must be safely evaluated to floats during config load.
