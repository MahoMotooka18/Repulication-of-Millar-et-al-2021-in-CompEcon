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
w_{t+1} = r(w_t - c_t) + e^{y_t}
$$

- income process:
  - baseline (Eq. 23): $y_{t+1}=\rho y_t + \sigma \varepsilon_{t+1}$, $\varepsilon_t \sim \mathcal{N}(0,1)$,
  - alternative (for exposition): temporary shock $y_t = \sigma \varepsilon_t$.

Implementation detail (time indexing):
- The budget update uses current income, i.e., $w_{t+1} = r(w_t - c_t) + e^{y_t}$.
- When forming Euler residuals, $u'(c_{t+1})$ is evaluated under the shock realization for $y_{t+1}$, but the cash-on-hand transition uses $y_t$ as in the paper’s Eq. (21).

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
  - income persistence: $\rho \in (-1,1)$

All additional parameters must be configurable and documented.

---

# 2. Policy Representation (Neural Network)

## 2.1 Inputs / outputs

- State: $(y_t, w_t)$
- Outputs: consumption share $s_t = c_t / w_t$, Lagrange multiplier $h_t$, and value function $V_t$.

## 2.2 Feasibility enforcement (non-negotiable)

Ensure $0 \le c_t \le w_t$ by construction:

- network outputs $s_t \in (0,1)$ via sigmoid,
- set $c_t = s_t \cdot w_t$.

## 2.3 Paper-consistent parameterization (Section 4.2)

Use the following output transforms:
$$
\\frac{c_t}{w_t}=\sigma(\\zeta_0+\\eta(y_t,w_t;\\vartheta))\\equiv \\varphi(y_t,w_t;\\theta),\\quad
h_t=\\exp(\\zeta_0+\\eta(y_t,w_t;\\vartheta))\\equiv h(y_t,w_t;\\theta),\\quad
V_t=\\zeta_0+\\eta(y_t,w_t;\\vartheta)\\equiv V(y_t,w_t;\\theta).
$$

Initialization and architecture:
- Two hidden layers with leaky ReLU activations.
- Compare 4 sizes: $8\\times 8$, $16\\times 16$, $32\\times 32$, $64\\times 64$.
- Initialize $\\zeta_0=0$; initialize remaining weights with He/Glorot uniform.

Objective-specific usage:
- Lifetime reward uses only $\\varphi(y,w;\\theta)$.
- Euler method uses $\\varphi(y,w;\\theta)$ and $h(y,w;\\theta)$.
- Bellman method uses all three $\\varphi$, $h$, and $V$.

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

## 3.1 Lifetime reward objective (Eq. 27)

Use the AiO expectation over $\\omega=(y_0,w_0,\\varepsilon_1,\\ldots,\\varepsilon_T)$:
$$
\\Xi(\\theta)=E_\\omega\\left[\\sum_{t=0}^{T}\\beta^t u(c_t)\\right],\\quad
\\frac{c_t}{w_t}=\\varphi(y_t,w_t;\\theta).
$$

## 3.2 Euler objective with borrowing constraint (Eqs. 28–30)

Use Fischer–Burmeister (FB) to enforce the Kuhn–Tucker conditions:
$$
\\Psi^{FB}(a,h)=a+h-\\sqrt{a^2+h^2}=0,\\quad
a=1-\\frac{c}{w},\\quad
h=1-\\frac{\\beta r E_\\varepsilon[u'(c')]}{u'(c)}.
$$

The composite objective uses an AiO product with independent shocks:
$$
E_{y,w,\\varepsilon_1,\\varepsilon_2}\\left[
\\Psi^{FB}(1-\\tfrac{c}{w},1-h)^2
+\\nu_h\\left(\\frac{\\beta r u'(c')}{u'(c)}\\Big|_{\\varepsilon_1}-h\\right)
\\left(\\frac{\\beta r u'(c')}{u'(c)}\\Big|_{\\varepsilon_2}-h\\right)
\\right].
$$

## 3.3 Bellman residual objective (Eqs. 31–32)

Bellman equation:
$$
V(y,w)=\\max_{c,w'}\\{u(c)+\\beta E_\\varepsilon[V(y',w')]\\}.
$$

Use FB with $h$ defined from the derivative of $V$ and apply AiO with two independent shocks. The objective combines:
- squared Bellman residuals, and
- FB residuals for the max operator,
with weights $\\nu$ and $\\nu_h$ chosen so residual magnitudes are comparable (as in Section 4.5).

## 3.4 AiO expectation operator (independence requirement)

- Euler/Bellman objectives use two **independent** shocks $(\varepsilon^{(1)}, \varepsilon^{(2)})$.
- Do **not** reuse the same shock tensor on both terms of the AiO product within a batch.

## 3.5 Section 2 (casting into DL expectation functions) implementation notes

- **Definition 2.2**: Decision rule $\varphi(\cdot;\theta)$ must be **time-invariant** and applied over a **fixed horizon** $T$ for the lifetime-reward objective.
- **Definition 2.4**: Lifetime reward uses a **composite draw** $\omega = (m_0, s_0, \varepsilon_1,\ldots,\varepsilon_T)$ so the nested expectation is written as a single expectation (AiO for reward).
- **Definitions 2.6–2.7**: Euler/Bellman objectives with squared residuals require **two independent shock draws** to form AiO products, i.e.,
  $$
  E_\varepsilon[f(\varepsilon)]^2 \Rightarrow E_{\varepsilon^{(1)},\varepsilon^{(2)}}[f(\varepsilon^{(1)})f(\varepsilon^{(2)})].
  $$
- **Definitions 2.1, 2.5**: State sampling is **random from a specified distribution** (not a fixed grid), matching Section 2’s expectation-function formulation.

---

# 4. Training & Evaluation Protocol

## 4.1 Training

- Epochs: $K = 50{,}000$
- Training samples per epoch: $64$ random points
- Wealth domain: $w \in [0.1, 4]$
- Income draws from stationary distribution or temporary-shock distribution.

## 4.1.1 Section 3 (DL solution method) implementation notes

- **Algorithm 1, Step 2i**: Simulated training data must be generated **on the ergodic set** of the model; do not use fixed hypercube grids outside the ergodic region.
- **Algorithm 1, Step 2i–2ii**: Training uses **fresh random draws each iteration** (data is re-sampled, not held fixed).
- **Equation (17)**: The empirical risk is a **sample average** of the integrand $\xi(\omega;\theta)$ over random draws.
- **Algorithm 1, Step 2ii–2iii**: Gradient is computed by **automatic differentiation** of the sampled objective and updated with **SGD/Adam**.
- **Equation (11)**: AiO integration uses **two independent shock draws** per state when squared residuals appear; use only **two integration nodes** per iteration.
 - **Equation (19)**: Learning rate is applied per-iteration; use Adam with overall step size $\\lambda=0.001$.

## 4.2 Evaluation

- Evaluation set size: $8{,}192$
- Expectation computed using **10-node Gauss–Hermite quadrature**
- Euler residuals use quadrature to approximate $E_{\varepsilon}[u'(c_{t+1})]$.
- Report:
  - mean / median / p90 / max Euler residuals,
  - residual distribution plots,
  - feasibility violations (must be zero).

## 4.3 Plotting conventions (match paper style)

- X-axis uses log scale for epochs (e.g., $\\log_{10}$ of iterations).
- The paper reports $K=50{,}000$ as $4\\cdot \\log_{10} 5$ on the horizontal axis.
- Training curves are plotted with a moving average (window configurable via config).
- Raw curves may be retained but should be visually distinct (lighter alpha).

Paper-matching panels:
- Fig. 3 (lifetime reward): objective over epochs, test Euler residuals, test lifetime reward.
- Fig. 4 (Euler): objective Euler residuals, test Euler residuals, test lifetime reward.
- Fig. 5 (Bellman): objective Bellman residuals, test Euler residuals, test lifetime reward.
- Fig. 6: comparison of decision rules ($c$ vs $w$) and a 100-period wealth simulation.

Output scope for this repo:
- Only the three training-curve figures are exported:
  - `training_curves_lifetime_reward.png`
  - `training_curves_euler.png`
  - `training_curves_bellman.png`
- Do not emit per-network single-metric plots, residual distributions, or policy-slice plots.

## 4.4 Debug logging (validation run)

For at least one run, log:
- $\min(c)$, $\max(c)$, $\min(w)$, $\max(w)$, $\min(w_{t+1})$.
- Constraint violation rates: share$\{c<0\}$, share$\{c>w\}$, share$\{w<0\}$.
- AiO independence checks: confirm $\varepsilon^{(1)} \neq \varepsilon^{(2)}$.

---

# 5. Neural Network Baseline

- Architecture: two hidden layers, compare $8\\times 8$, $16\\times 16$, $32\\times 32$, $64\\times 64$
- Activation: leaky ReLU in hidden layers
- Optimizer: ADAM with learning rate $\\lambda=0.001$

---

# 6. Required Outputs

Save all outputs to `outputs/section4/`:

- `metrics.csv` (combined across objectives and network sizes)
- `metrics_<objective>.csv` (optional per-objective breakdown)
- model checkpoint
- plots:
  - `training_curves_lifetime_reward.png`
  - `training_curves_euler.png`
  - `training_curves_bellman.png`

## 6.1 Evaluation pipeline (shared across objectives)

- All objectives (`lifetime_reward`, `euler`, `bellman`) must use the same `evaluate_metrics()` function for evaluation.
- Loss and evaluation metrics must be computed once and reused for display and CSV logging (no separate "display" vs "save" paths).
- Metric failures must not create empty cells in CSV. Use NaN/0/empty-string per schema and log failures to debug artifacts.

## 6.2 Numerical stability & metric robustness

Common causes of NaN/Inf include: `c<=0`, division by zero, log/sqrt domain violations, overflow.

Rules:
- Guard: explicitly check finite values and domain constraints before aggregation.
- Collect: record `violation_count` and sample violations for debugging.
- Save: record `euler_fb_finite_ratio` (ratio of finite elements used in the metric).
- Exceptions: if metric computation fails, fill metrics with NaN (or 0/empty string per schema) and log the exception to debug artifacts.

## 6.3 Metric schema (no blank cells)

All metrics CSVs (`metrics.csv` and `metrics_<objective>.csv`) must share the same schema and forbid blank cells.

Required columns:
- `run_id`
- `timestamp`
- `git_hash` (if available; otherwise empty string)
- `objective` (`lifetime_reward`, `euler`, `bellman`)
- `network_size` (e.g., `8x8`)
- `epoch`
- `loss`
- `euler_fb_mean` (NaN if not computable)
- `euler_fb_finite_ratio` (0..1, 0 if not computable)
- `violation_count` (integer; use 0 if not computable)
- `warning_count` (integer runtime warning count)
- `exception_flag` (0/1)
- `exception_type` (empty string if none)
- `exception_message` (empty string if none)
- `lifetime_reward_mean` (NaN if not computable)

No-blank rule:
- Numeric metrics: NaN or 0 as specified; never leave empty cells.
- String fields: empty string if not available.
- Integer fields: 0 if not available.

## 6.4 Debug artifacts & failure reporting

Save debug artifacts under `outputs/section4/<run_id>/debug/`:
- `metrics_failures.jsonl` (one JSON per metric failure/exception).
- `violations_samples.jsonl` (sample violations like `c<=0`, up to K samples).
- `warnings.log` (captured runtime warnings).
- `config_snapshot.json` (experiment configuration snapshot).

Plotting rule:
- Plot generation must not fail when NaN values are present; NaNs are treated as missing data, and a debug note is recorded.

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
python Lab_Section4_ConsumptionSaving/run_section4_experiment.py --config configs/section4.yaml
```

Config parsing note:
- YAML does not evaluate arithmetic; if expressions are used in the config, they must be safely evaluated to floats during load.
