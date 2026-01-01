---
title: "Replication Requirements — Section 4 (Consumption–Saving Problem)"
paper: "Deep learning for solving dynamic economic models (Maliar, Maliar, Winant, 2021, JME)"
scope: "Section 4 — Numerical analysis of the consumption-saving problem"
language: "English"
---

# 0. Purpose

Build a **fully reproducible replication package** for Section 4 of the paper, including:

- exact model specification (consumption–saving with borrowing constraint),
- deep-learning solution under **three objectives** (lifetime reward, Euler residuals, Bellman residuals),
- training + evaluation pipeline that matches the paper’s numerical settings,
- plots/tables that match the paper’s reported diagnostics.

This document is a **requirements specification to hand to an LLM** so it can generate the repository and code.

---

# 1. Model Specification (Consumption–Saving)

## 1.1 Household problem

Maximize expected discounted utility:
$$
\max_{\{c_t\}_{t\ge 0}} \; \mathbb{E}\sum_{t=0}^\infty \beta^t u(c_t)
$$

subject to:

- feasibility / borrowing constraint: $0 \le c_t \le w_t$
- cash-on-hand evolution:
$$
w_{t+1} = r(w_t - c_t) + e y_t
$$

- income process:
  - baseline: AR(1) process (as in the paper),
  - alternative (for exposition): temporary shock $y_t = \sigma \varepsilon_t$.

## 1.2 Preferences and baseline parameters

- CRRA utility:
$$
u(c)=\frac{c^{1-\gamma}-1}{1-\gamma}
$$
(with special handling for $\gamma = 1$)

- Baseline parameter values:
  - $\gamma = 2$
  - $\beta = 0.9$
  - $r = 1.04$
  - temporary-shock option: $\sigma = 0.1$

All additional parameters must be configurable and documented.

---

# 2. Policy Representation (Neural Network)

## 2.1 Inputs / outputs

- State: $(y_t, w_t)$
- Output: consumption $c_t$ or consumption share $s_t = c_t / w_t$

## 2.2 Feasibility enforcement (non-negotiable)

Ensure $0 \le c_t \le w_t$ by construction:

- network outputs $s_t \in (0,1)$ via sigmoid,
- set $c_t = s_t \cdot w_t$.

---

# 3. Objectives to Replicate

Implement **all three** objectives:

1. **Lifetime reward objective**  
   Maximize expected discounted utility along simulated paths.

2. **Euler-equation residual objective**  
   Minimize Euler residuals with borrowing constraint handled via complementarity (e.g. Fischer–Burmeister).

3. **Bellman residual objective**  
   Approximate the value function and minimize Bellman equation residuals.

Each objective must be selectable via configuration.

---

# 4. Training & Evaluation Protocol

## 4.1 Training

- Epochs: $K = 50{,}000$
- Training samples per epoch: $64$ random points
- Wealth domain: $w \in [0.1, 4]$
- Income draws from stationary distribution or temporary-shock distribution.

## 4.2 Evaluation

- Evaluation set size: $8{,}192$
- Expectation computed using **10-node Gauss–Hermite quadrature**
- Report:
  - mean / median / max Euler residuals,
  - residual distribution plots,
  - feasibility violations (must be zero).

---

# 5. Neural Network Baseline

- Architecture: two hidden layers, $64 \times 64$
- Activation: configurable (paper default preserved)
- Optimizer: ADAM (configurable learning rate)

---

# 6. Required Outputs

Save all outputs to `outputs/section4/`:

- `metrics.csv`
- model checkpoint
- plots:
  - training loss vs epoch,
  - Euler residual distributions,
  - policy function slices ($c$ vs $w$ for fixed $y$).

---

# 7. Acceptance Criteria

A run is successful if:

- feasibility holds everywhere: $0 \le c_t \le w_t$,
- Euler residuals are of the same **order of magnitude** as in the paper,
- all artifacts are saved deterministically.

---

# 8. Engineering Requirements

- Config-driven execution:
```bash
python experiments/run_section4.py --config configs/section4.yaml
