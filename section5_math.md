
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
- Wealth evolution:
$$
w^i_{t+1} = r_t (w^i_t - c^i_t) + y^i_t
$$
- Prices depend on aggregates (capital, distribution).

## 1.2 Preferences and parameters

- CRRA utility with:
  - $\gamma = 1$
  - $\beta = 0.96$
- Aggregate shock:
  - persistence $\rho = 0.95$
  - volatility $\sigma = 0.01$

All remaining parameters must be configurable and documented.

---

# 2. State Representation

## 2.1 Network inputs

- own state $(y^i_t, w^i_t)$,
- aggregate shock $z_t$,
- distribution $D_t = \{(y^j_t, w^j_t)\}_{j=1}^I$.

## 2.2 Permutation invariance (required)

Implement a permutation-invariant encoder (e.g. DeepSets):
$$
\phi(D_t) = \text{Agg}\bigl(\psi(y^j_t, w^j_t)\bigr)
$$
where $\text{Agg}$ is mean or sum.

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

---

# 6. Simulation and Training Protocol

## 6.1 Simulation

- Total periods: $K = 300{,}000$

## 6.2 Training schedule

- Train every $10$ periods,
- Each update uses $100$ simulated points.

---

# 7. Evaluation

## 7.1 Approximate aggregation (KS regression)

Estimate:
$$
\ln k_{t+1} = \xi_0 + \xi_1 \ln k_t + \xi_2 \ln z_t
$$

Report coefficients and $R^2$.

## 7.2 Summary statistics

Export:
- wealth moments (mean, variance, percentiles, Gini),
- aggregate moments,
- correlation statistics.

## 7.3 Plots

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
python experiments/run_section5.py --config configs/section5.yaml