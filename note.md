# Technical Note: Maliar, Maliar, Winant (2021) — Deep Learning for Solving Dynamic Economic Models

## 1. Research questions
- Can dynamic economic models be solved by casting them as **nonlinear regression problems** and training with **stochastic gradient descent**?
- Can a **single deep-learning framework** handle value-function, Bellman-equation, and Euler-equation characterizations?
- How can **high-dimensional expectations** be approximated efficiently in large-scale models (e.g., Krusell–Smith)?

## 2. Generic model structure
The paper starts from a standard dynamic model with exogenous state \(m_t\), endogenous state \(s_t\), control \(x_t\), and reward \(r\):

- Exogenous transition: \(m_{t+1}=M(m_t,\varepsilon_t)\)
- Endogenous transition: \(s_{t+1}=S(m_t,s_t,x_t,m_{t+1})\)
- Feasibility: \(x_t\in X(m_t,s_t)\)
- Objective: \(\max \mathbb{E}_0\sum_{t\ge0}\beta^t r(m_t,s_t,x_t)\)

The **decision rule** is approximated by a parametric function \(x_t=\varphi(m_t,s_t;\theta)\), typically a neural network.

## 3. Three equivalent objective functions
The same solution can be recovered by minimizing one of three **expected-loss** functions over \(\theta\):

### (a) Lifetime-reward objective
Simulate a trajectory under \(\varphi(\cdot;\theta)\) and maximize discounted utility:
\[
\mathcal{L}(\theta)=\mathbb{E}_{\omega}\Big[\sum_{t=0}^{T}\beta^t r(m_t,s_t,\varphi(m_t,s_t;\theta))\Big].
\]

### (b) Euler-residual objective
Stack equilibrium conditions (Euler equations, constraints) and minimize squared residuals:
\[
\mathcal{L}(\theta)=\mathbb{E}_{(m,s)}\Big[\sum_j v_j(\mathbb{E}_\varepsilon f_j(\cdot))^2\Big].
\]
An **independence trick** allows the expectation to be moved outside the square by using two independent shock draws.

### (c) Bellman-residual objective
Use the Bellman equation and (if needed) complementarity/FOC residuals in a weighted sum:
\[
\mathcal{L}(\theta)=\mathbb{E}[R(\varepsilon_1)R(\varepsilon_2)+v\,G(\varepsilon_1)G(\varepsilon_2)].
\]

## 4. All-in-One (AiO) integration operator
To avoid nested expectations, the paper merges the distribution over **state draws** and **future shocks** into a single composite draw \(\omega\). This gives a **single Monte Carlo expectation** and makes high-dimensional problems computationally feasible.

## 5. Learning and optimization setup
- The expected-loss form is treated as a **statistical learning** problem.
- Training uses **stochastic/mini-batch gradient descent** on simulated draws.
- Neural networks are used for decision rules, multipliers, and value functions because they scale well with dimension and can handle nonlinearity and kinks.

## 6. Application 1: Consumption–saving with borrowing constraint
**State:** income \(y_t\), cash-on-hand \(w_t\).  
**Law of motion:** \(w_{t+1}=r(w_t-c_t)+e^{y_t}\), \(y_{t+1}=\rho y_t+\sigma\varepsilon_{t+1}\).  
**Constraint:** \(0\le c_t\le w_t\).  
**Utility:** CRRA \(u(c)=(c^{1-\gamma}-1)/(1-\gamma)\).

The policy is parameterized as a **consumption share** \(c_t/w_t=\sigma(\zeta_0+\eta(y_t,w_t;\vartheta))\). The paper implements **three versions** of the DL method (lifetime reward, Euler, Bellman) and shows close agreement across them.

## 7. Application 2: Krusell–Smith (1998) heterogeneous-agent model
**Households:** choose \(c^i_t\), \(k^i_{t+1}\) subject to
\(w^i_{t+1}=R_{t+1}(w^i_t-c^i_t)+W_{t+1}e^{y^i_{t+1}}\).

**Idiosyncratic shock:** \(y^i_{t+1}=\rho_y y^i_t+\sigma_y\varepsilon^i_{t+1}\).  
**Aggregate shock:** \(z_{t+1}=\rho_z z_t+\sigma_z\epsilon_{t+1}\).  
**Prices:** from Cobb–Douglas production with capital and labor aggregates.

The policy uses a neural network that depends on individual and aggregate states, and approximate aggregation is captured by a law of motion such as:
\(\log K' = a_0+a_1\log K+a_2 z+a_3 z\log K\).

## 8. Data and computational setup
- **No empirical data** are required; all training data are **simulated**.
- Training samples are drawn from the state space (or from simulated ergodic distributions in large models).
- Expectations over shocks are approximated via **Monte Carlo** using AiO.
- Optimization is carried out with **stochastic gradient descent** (mini-batches).

## 9. Main technical takeaways
- Dynamic models can be solved by **minimizing a single expected loss** derived from lifetime rewards, Euler residuals, or Bellman residuals.
- The **AiO operator** collapses nested expectations and makes large-scale models tractable.
- Neural networks offer a scalable, flexible approximation family for high-dimensional decision rules.
