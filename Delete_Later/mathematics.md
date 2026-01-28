# Mathematical Roadmap: Maliar, Maliar, Winant (2021)

This note is a **standalone** explanation of the paper’s mathematical argument.
You should be able to follow the logic without opening any other file. All
symbols and equations are introduced as they appear in the narrative.

---

## 1. Generic dynamic model (Section 2)

The paper starts from a **generic dynamic economic model** with exogenous and
endogenous states, controls, and a reward function.

### (1) Exogenous state (Markov) transition
$$
m_{t+1} = M(m_t, \varepsilon_t).
$$
- $m_t \in \mathbb{R}^{n_m}$ is the exogenous state.
- $\varepsilon_t$ is an i.i.d. shock; $M(\cdot)$ is the Markov transition.

### (2) Endogenous state transition
$$
s_{t+1} = S(m_t, s_t, x_t, m_{t+1}).
$$
- $s_t \in \mathbb{R}^{n_s}$ is the endogenous state.
- $x_t \in \mathbb{R}^{n_x}$ is the decision variable.

### (3) Feasibility/constraints
$$
x_t \in X(m_t, s_t).
$$

### (4) Lifetime reward maximization
$$
\max_{\{x_t, s_{t+1}\}_{t\ge 0}}\; \mathbb{E}_0\left[\sum_{t=0}^{\infty} \beta^t\, r(m_t, s_t, x_t)\right].
$$
- $r(\cdot)$ is the period reward; $\beta \in [0,1)$ is the discount factor.

The goal is to approximate the **optimal decision rule** $x_t=\varphi(m_t,s_t)$
by a parametric rule $\varphi(\cdot;\theta)$.

---

## 2. Three equivalent ways to characterize the solution (Section 2)

The paper shows that the same solution can be obtained by minimizing one of
three objective functions.

### 2.1 Lifetime-reward objective

#### (5) Value function
$$
V(m_0,s_0) \equiv \max_{\{x_t,s_{t+1}\}_{t\ge 0}}\;\mathbb{E}\left[\sum_{t=0}^{\infty}\beta^t r(m_t,s_t,x_t)\right].
$$

#### (6) Finite-horizon approximation under a fixed policy
$$
V_T(m_0,s_0;\theta) \equiv \mathbb{E}\left[\sum_{t=0}^{T}\beta^t
r\big(m_t,s_t,\varphi(m_t,s_t;\theta)\big)\right].
$$

#### (7) Randomized initial conditions (nested expectation)
$$
\mathcal{L}(\theta) \equiv
\mathbb{E}_{(m_0,s_0)}\,\mathbb{E}_{(\varepsilon_1,\ldots,\varepsilon_T)}
\left[\sum_{t=0}^{T}\beta^t r\big(m_t,s_t,\varphi(m_t,s_t;\theta)\big)\right].
$$

#### (8) AiO (All-in-One) expectation
$$
\mathcal{L}(\theta) = \mathbb{E}_{\omega}[\xi(\omega;\theta)],
\quad \omega=(m_0,s_0,\varepsilon_1,\ldots,\varepsilon_T).
$$
- AiO collapses nested expectations into a single draw of $\omega$.

### 2.2 Euler-residual objective

#### (9) Euler equations (generic form)
$$
\mathbb{E}_{\varepsilon}\big[f_j(m,s,x,m',s',x')\big]=0,\quad j=1,\ldots,J.
$$
- These include FOCs, market clearing, and other equilibrium conditions.

#### (10) Squared residuals over the state distribution
$$
\mathcal{L}(\theta)=\mathbb{E}_{(m,s)}\left[
\sum_{j=1}^J v_j\left(\mathbb{E}_{\varepsilon}
 f_j(\cdot)\right)^2\right].
$$

#### (11) Independence trick
$$
\mathbb{E}_{\varepsilon_1}[f(\varepsilon_1)]\,\mathbb{E}_{\varepsilon_2}[f(\varepsilon_2)]
= \mathbb{E}_{(\varepsilon_1,\varepsilon_2)}[f(\varepsilon_1)f(\varepsilon_2)].
$$

#### (12) AiO Euler objective
$$
\mathcal{L}(\theta)=\mathbb{E}_{(m,s,\varepsilon_1,\varepsilon_2)}\Big[
F(\varepsilon_1)\,F(\varepsilon_2)\Big],
\quad F(\varepsilon)=\sum_{j=1}^J v_j f_j(\cdot)\big|_{\varepsilon}.
$$

### 2.3 Bellman-residual objective

#### (13) Bellman equation
$$
V(m,s)=\max_{x,s'}\left\{ r(m,s,x) + \beta\,\mathbb{E}_{\varepsilon}
\big[V(m',s')\big] \right\}.
$$

#### (14) Bellman residual + FOC penalty (generic form)
$$
\mathcal{L}(\theta)=\mathbb{E}_{(m,s)}\Big[\text{Bellman residual}\Big]^2
+ v\,\mathbb{E}_{(m,s)}\Big[\text{FOC residual}\Big]^2.
$$

#### (15) AiO Bellman objective
$$
\mathcal{L}(\theta)=\mathbb{E}_{(m,s,\varepsilon_1,\varepsilon_2)}\Big[
R(\varepsilon_1)\,R(\varepsilon_2)
+ v\,G(\varepsilon_1)\,G(\varepsilon_2)\Big],
$$
where $R$ is the Bellman residual and $G$ is the FOC residual.

---

## 3. Learning view (Section 3)

The three objectives above are written in the same statistical-learning form.

### (16) Unified expected loss
$$
\min_{\theta\in\Theta}\; \mathbb{E}_{\omega}\big[\xi(\omega;\theta)\big].
$$

### (17) Empirical risk (simulation)
$$
\min_{\theta\in\Theta}\; \frac{1}{n}\sum_{i=1}^n \xi(\omega_i;\theta).
$$

### (18) Curse of dimensionality (volume ratio)
$$
V_d =
\begin{cases}
\dfrac{(\pi/2)^{(d-1)/2}}{1\cdot3\cdot5\cdots d}, & d=1,3,5,\ldots \\
\dfrac{(\pi/2)^{d/2}}{2\cdot4\cdot6\cdots d}, & d=2,4,6,\ldots
\end{cases}
$$
- $V_d$ shrinks quickly with $d$, motivating simulation-based AiO methods.

### (19) Stochastic/mini-batch gradient descent
$$
\theta^{k+1} \leftarrow \theta^{k} - \lambda_k\,\frac{1}{n}\sum_{i=1}^{n}
\nabla_{\theta}\,\xi(\omega_i;\theta^{k}).
$$

---

## 4. Section 4: Consumption–Saving model (concrete instantiation)

### 4.1 Model equations

#### (21) Cash-on-hand evolution
$$
w_{t+1} = r\,(w_t - c_t) + e^{y_t}.
$$

#### (22) Borrowing/feasibility constraint
$$
0 \le c_t \le w_t.
$$

#### (23) Income process (AR(1))
$$
y_{t+1} = \rho y_t + \sigma\varepsilon_{t+1},\quad \varepsilon_t\sim\mathcal{N}(0,1).
$$

#### (24) Temporary-shock variant
$$
y_t = \sigma\varepsilon_t.
$$

#### (25) Utility maximization
$$
\max_{\{c_t\}} \; \mathbb{E}\sum_{t=0}^{\infty} \beta^t u(c_t).
$$

#### (26) CRRA utility
$$
u(c)=\frac{c^{1-\gamma}-1}{1-\gamma}.
$$

### 4.2 Policy parameterization

#### (26a) Consumption share
$$
\frac{c_t}{w_t}=\sigma\big(\zeta_0+\eta(y_t,w_t;\vartheta)\big)
\equiv \varphi(y_t,w_t;\theta).
$$

#### (26b) Multiplier and value outputs
$$
h_t=\exp\big(\zeta_0+\eta(y_t,w_t;\vartheta)\big),
\qquad
V_t=\zeta_0+\eta(y_t,w_t;\vartheta).
$$

### 4.3 Objective 1: Lifetime reward

#### (27) AiO lifetime objective
$$
\Xi(\theta)=\mathbb{E}_{\omega}\left[\sum_{t=0}^{T}\beta^t u(c_t)\right],
\quad \omega=(y_0,w_0,\varepsilon_1,\ldots,\varepsilon_T).
$$

### 4.4 Objective 2: Euler residuals with borrowing constraint

#### (28) Fischer–Burmeister complementarity
$$
\Psi^{FB}(a,h)=a+h-\sqrt{a^2+h^2}=0.
$$

#### (29) Slack and multiplier definitions
$$
a=1-\frac{c}{w},\qquad h=1-\frac{\beta r\,\mathbb{E}_\varepsilon[u'(c')]}{u'(c)}.
$$

#### (30) AiO Euler objective
$$
\mathbb{E}_{y,w,\varepsilon_1,\varepsilon_2}\left[
\Psi^{FB}\left(1-\tfrac{c}{w},1-h\right)^2
+\nu_h\left(\frac{\beta r u'(c')}{u'(c)}\Big|_{\varepsilon_1}-h\right)
\left(\frac{\beta r u'(c')}{u'(c)}\Big|_{\varepsilon_2}-h\right)
\right].
$$

### 4.5 Objective 3: Bellman residuals

#### (31) Bellman equation
$$
V(y,w)=\max_{c,w'}\{u(c)+\beta\,\mathbb{E}_\varepsilon[V(y',w')]\}.
$$

#### (32) AiO Bellman objective
$$
\mathbb{E}_{y,w,\varepsilon^{(1)},\varepsilon^{(2)}}\left[
R(\varepsilon^{(1)})\,R(\varepsilon^{(2)})\right],
\quad R(\varepsilon)=V(y,w)-u(c)-\beta V(y',w')\big|_{\varepsilon}.
$$

#### (33) Correct next-state construction
$$
w' = r(w-c) + e^{y'},\quad y' = \rho y + \sigma\varepsilon.
$$
- This is the concrete instance of the generic transition (1)–(2).

---

## 5. Section 5: Krusell–Smith (1998) model (concrete instantiation)

### 5.1 Household problem

#### (37) Utility maximization
$$
\max_{\{c^i_t,k^i_{t+1}\}} \; \mathbb{E}_0\sum_{t=0}^{\infty}\beta^t u(c^i_t).
$$

#### (38) Budget constraint
$$
w^i_{t+1} = R_{t+1}(w^i_t - c^i_t) + W_{t+1}\exp(y^i_{t+1}).
$$

#### (39) Feasibility
$$
c^i_t \le w^i_t.
$$

#### (40) Idiosyncratic productivity
$$
y^i_{t+1}=\rho_y y^i_t + \sigma_y\varepsilon^i_{t+1},\quad \varepsilon^i_t\sim\mathcal{N}(0,1).
$$

#### (41) Aggregate productivity
$$
z_{t+1}=\rho_z z_t + \sigma_z\epsilon_{t+1},\quad \epsilon_t\sim\mathcal{N}(0,1).
$$

#### (42) Production and factor prices
$$
Y_t=z_t K_t^{\alpha} L_t^{1-\alpha},\quad
R_t=1-d+\alpha z_t K_t^{\alpha-1}L_t^{1-\alpha},\quad
W_t=z_t(1-\alpha)K_t^{\alpha}L_t^{-\alpha}.
$$
- $K_t=\sum_i k^i_t$, $L_t=\sum_i \exp(y^i_t)$.

### 5.2 Policy parameterization

#### (43) Consumption share
$$
\frac{c^i_t}{w^i_t}=\sigma\big(\zeta^{(c)}_0+\eta^{(c)}(y^i_t,w^i_t,z_t,D_t)\big).
$$

#### (44) Multiplier and value outputs
$$
h^i_t=\exp\big(\zeta^{(h)}_0+\eta^{(h)}(\cdot)\big),
\qquad
V^i_t=\zeta^{(V)}_0+\eta^{(V)}(\cdot).
$$

### 5.3 Objectives

#### (45) Lifetime reward (AiO)
$$
\Xi(\theta)=\mathbb{E}_{\omega}\left[\sum_{t=0}^{T}\beta^t u(c^i_t)\right].
$$

#### (46) Euler residual (AiO product)
$$
\mathbb{E}_{\text{state},\varepsilon_1,\varepsilon_2}\left[
F(\varepsilon_1)\,F(\varepsilon_2)\right].
$$

#### (47) Bellman residual (AiO product)
$$
\mathbb{E}_{\text{state},\varepsilon_1,\varepsilon_2}\left[
R(\varepsilon_1)\,R(\varepsilon_2)\right].
$$

### 5.4 Approximate aggregation

#### (48) KS law of motion (example form)
$$
\log K' = a_0 + a_1 \log K + a_2 z + a_3 z\log K.
$$

---

## 6. One-sentence takeaway

All models in the paper are solved by **minimizing a single expected loss**
(Eq. (16)) constructed from either **lifetime reward**, **Euler residuals**, or
**Bellman residuals**, and Sections 4–5 are concrete instantiations of the
same generic structure defined in Section 2.

