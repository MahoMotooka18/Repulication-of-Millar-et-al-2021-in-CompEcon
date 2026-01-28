# Replication Differences Report (Updated)

This note compares the current implementation against the updated replication instructions in:
- `Delete_Later/new_section4_math.md`
- `Delete_Later/new_section5_math.md`

It lists mismatches and concrete fix proposals.

---

## Section 4 (Consumption–Saving)

1) **Cash-on-hand transition in Euler/Bellman residuals uses `y_t`, not `y_{t+1}`**
- **Where in code:**
  - `Lab_Section4_ConsumptionSaving/model_consumption_saving.py:79-98` (transition uses `exp(y_t)`)
  - `Lab_Section4_ConsumptionSaving/objectives.py:110-118` (Euler: `state_transition(..., y_batch)`)
  - `Lab_Section4_ConsumptionSaving/objectives.py:200-207` (Bellman: `state_transition(..., y_batch)`)
  - `Lab_Section4_ConsumptionSaving/evaluator.py:120-128` (evaluation uses `y_t`)
- **Why it conflicts:** `new_section4_math.md` (Section 4.6) says “Use `y'` in `w' = r(w-c)+e^{y'}` when evaluating Bellman/Euler residuals.”
- **Fix proposal:**
  - Add a residual-specific transition: `state_transition_next_y(w_t, c_t, y_next)`.
  - In Euler/Bellman objectives and evaluator, compute `y_next` first and pass it into the transition for `w_next`.
  - Keep the original transition for simulation if you want to preserve the baseline timing.

2) **Bellman objective uses FOC-based `λ` in FB term and `a·λ` in the multiplier term**
- **Where in code:**
  - `Lab_Section4_ConsumptionSaving/objectives.py:224-233` (FB uses `λ` from FOC, then `fb_1 * fb_2`)
  - `Lab_Section4_ConsumptionSaving/objectives.py:235-239` (multiplier term uses `a * λ`)
- **Why it conflicts:** `new_section4_math.md` (Eq. 32) specifies
  - FB term as `PsiFB(1 - c/w, 1 - h)^2` (no shock-wise product), and
  - multiplier term as `[(βr dV/dw'/u'(c) - h)_1 * (βr dV/dw'/u'(c) - h)_2]`.
- **Fix proposal:**
  - Compute `h = policy.forward_h(...)` at current state.
  - Use `fb = PsiFB(1 - c/w, 1 - h)` and add `fb**2` (not `fb_1 * fb_2`).
  - Replace `a * λ` with `(βr dV/dw'/u'(c) - h)` AiO product.
  - Keep the Bellman residual AiO product as is.

---

## Section 5 (Krusell–Smith)

1) **Prices/aggregation use total labor and `exp(z_t)`; updated notes use average labor and `z_t` directly**
- **Where in code:**
  - `Lab_Section5_Krusell_and_Smith_1998/model_ks1998.py:160-215` (`exp(z_t)` and `total_labor = sum_i exp(y_i)`) 
- **Why it conflicts:** `new_section5_math.md` (Eq. 42) defines
  - `R_t, W_t` with `z_t` (no `exp`) and **average** labor `(1/ℓ)∑ exp(y_i)`.
- **Fix proposal:**
  - If following the updated notes strictly, set labor input to `mean(exp(y))` and use `z_t` directly in prices.
  - Alternatively, adjust the notes to `exp(z_t)` if `z_t` is meant to be log TFP (current code path). Decide and document.

2) **Policy parameterization differs (frozen intercept + steady-state logit shift)**
- **Where in code:**
  - `Lab_Section5_Krusell_and_Smith_1998/nn_policy_ks.py:62-75` (`phi_intercept` is non-trainable; `phi_logit_shift` added)
- **Why it conflicts:** `new_section5_math.md` specifies a shared `zeta_0 + eta(...)` for `c/w`, `h`, and `V`, with `zeta_0` initialized to 0 (not frozen) and no steady-state shift.
- **Fix proposal:**
  - Make `phi_intercept` trainable or share one trainable intercept across all heads.
  - Make `phi_logit_shift` optional (default off) for strict replication.

3) **Euler objective clamps `1 - h` to be nonnegative**
- **Where in code:**
  - `Lab_Section5_Krusell_and_Smith_1998/objectives_ks.py:160-165`
- **Why it conflicts:** `new_section5_math.md` (Eq. 44) uses `PsiFB(1 - c/w, 1 - h)` directly; clamping removes penalties when `h > 1`.
- **Fix proposal:**
  - Remove the clamp on `1 - h` (keep only numerical guard on `w`).

4) **Bellman objective uses FOC-based `λ` in FB term instead of `h`**
- **Where in code:**
  - `Lab_Section5_Krusell_and_Smith_1998/objectives_ks.py:287-296`
- **Why it conflicts:** `new_section5_math.md` (Eq. 45) uses `PsiFB(1 - c/w, 1 - h)` and keeps FOC consistency in the separate `G` term.
- **Fix proposal:**
  - Replace the FB term to use `1 - h_t` instead of `λ`.
  - Keep `(β R dV/dw'/u'(c) - h)` AiO product as is.

5) **Idiosyncratic productivity normalization and mean-shifted shocks are extra modeling choices**
- **Where in code:**
  - `Lab_Section5_Krusell_and_Smith_1998/model_ks1998.py:274-291` (normalize productivity)
  - `Lab_Section5_Krusell_and_Smith_1998/model_ks1998.py:153-158` (mean shift)
  - `Lab_Section5_Krusell_and_Smith_1998/main_section5.py:392-470` (normalization applied in training loop)
- **Why it conflicts:** The updated notes do not mention per-period normalization or mean-shift corrections.
- **Fix proposal:**
  - For strict replication, disable `use_log_shock_shift` and `enforce_bounds`, and remove `normalize_productivity` calls.
  - Otherwise document these as intentional deviations.

6) **Input scaling is not mentioned in the updated notes**
- **Where in code:**
  - `Lab_Section5_Krusell_and_Smith_1998/policy_utils_ks.py`
  - `Lab_Section5_Krusell_and_Smith_1998/main_section5.py:176-238`
- **Why it conflicts:** `new_section5_math.md` describes raw state inputs.
- **Fix proposal:**
  - Set `input_scaling.enabled: false` for strict replication, or document it as a training-stability deviation.

