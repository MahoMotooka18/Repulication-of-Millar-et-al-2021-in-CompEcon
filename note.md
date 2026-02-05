# Technical Note: Maliar, Maliar, Winant (2021) — Deep Learning for Solving Dynamic Economic Models

## 1. Research questions
The core research question is whether deep learning can provide a unified way to decide **what to optimize**, **how to train**, and **how to compute expectations** when solving dynamic economic models. The section centers on three points:

- **Unified objective formulation**  
  Can the three fundamental objects in economic dynamics—**lifetime reward functions, Bellman equations, and Euler equations**—all be cast as **objective functions for Monte Carlo simulation**? If so, can these distinct theoretical characterizations (value-function, residual-minimization, and optimality-condition forms) be handled within a **single learning framework**?

- **Adapting stochastic gradient descent**  
  Given the three constructed objectives, how can **stochastic gradient descent (SGD)** be adapted to their training? Specifically, can random mini-batches drawn from the state space replace fixed grids, enabling stable training in both small and large problems?

- **Efficient expectation via the AiO operator**  
  Can the **All-in-One (AiO) expectation operator** merge the two nested expectations in Monte Carlo integration (over state draws and future shocks) into a single composite draw, thereby **reducing the cost of high-dimensional integration** while preserving accuracy in large-scale models?

## 2. Key equations and models  
### 2.1  All-in-One (AiO) integration operator
The AiO operator is critical when models have **many stochastic shocks**. 

For Euler-residual objectives with a squared conditional expectation,

$$
E_{(m,s)}\Big(E_{\varepsilon}[f(m,s,\varepsilon)]\Big)^2,
$$

Standard quadrature would require a **tensor-product grid** over shocks, so the number of integration nodes explodes with the number of shocks. 

AiO avoids this by using **two independent random shock draws** $\varepsilon'$ and $\varepsilon''$ and rewriting the square as an expectation of a product:

$$
E_{(m,s,\varepsilon',\varepsilon'')}\Big[f(m,s,\varepsilon')\,f(m,s,\varepsilon'')\Big].
$$

**Only two integration nodes (two random draws)** are needed per state, regardless of the number of shocks.
 
The approximation is crude in any single iteration but **unbiased**, and it converges as iterations accumulate.

This same logic allows the expectation operator and gradient operator to commute, so the gradient of the expected loss can be approximated by a **sample average of gradients**:

$$
\nabla E(\theta)=\nabla E_{\omega}[\xi(\omega;\theta)] \approx \frac{1}{n}\sum_{i=1}^n \nabla_{\theta}\xi(\omega_i;\theta).
$$

where $k$ is iteration, $\nabla E(\theta)$ is a gradient of $E(\theta)$.

By using sample average of gradients, we can approximate the gradient as follows:

$$
\theta_{k+1} \leftarrow \theta_k - \lambda_k [ \frac{1}{n}\sum_{i=1}^{n} \nabla_{\theta}\,\xi(\omega_i;\theta_k) ]
$$

where $\lambda_k$ is a learning rate.

In the limit $n=1$, the method above becomes **stochastic gradient descent**, where each update uses only **one randomly selected state draw** and **two shock draws**.

### 2.2 DL solution algorithm
Algorithm 1 (as described in the paper) can be summarized as follows.

Step 1. Initialize the algorithm:
1. Construct theoretical risk $E(\theta)=E_{\omega}[\xi(\omega;\theta)]$ (lifetime reward, Euler, or Bellman equations).
2. Define empirical risk $\hat{E}(\theta)=\frac{1}{n}\sum_{i=1}^n \xi(\omega_i;\theta)$.
3. Define a topology of the neural network $\varphi(\cdot;\theta)$.
4. Fix initial vector of coefficients $\theta$.

Step 2. Train the machine, i.e., find $\theta$ that minimizes empirical risk $\hat{E}(\theta)$:
1. Simulate the model to produce data ${\omega_i}_{i=1}^n$ by using the decision rule $\varphi(\cdot;\theta)$.
2. Construct the gradient $\nabla \hat{E}(\theta)=\frac{1}{n}\sum_{i=1}^n \nabla_{\theta}\xi(\omega_i;\theta)$.
3. Update coefficients $\theta \leftarrow \theta - \lambda \nabla \hat{E}(\theta)$ and return to Step 2.1.

End Step 2 if the convergence criterion $|\hat{\theta}-\theta|\ll \varepsilon$ is satisfied.

Step 3. Assess the accuracy of the constructed approximation $\varphi(\cdot;\theta)$ on a new sample.

The algorithm has multiple hyperparameters: network topology, learning rate, number of simulation points and integration nodes, training method, and possibly regularization (e.g., Tikhonov or Lasso) to address overfitting and ill-conditioning. The objective function also has hyperparameters such as **relative weights of the different model equations**, and these are selected by standard validation procedures (accuracy and speed).

### 2.3 Application 1: Consumption–saving with borrowing constraint
#### 2.3.1 Model
The consumption–saving problem is

$$
\max_{(c_t,w_{t+1})_{t = 0}^{\infty}}\;\mathbb{E}_0\left[\sum_{t=0}^{\infty}\beta^t u(c_t)\right]
$$

subject to

$$
w_{t+1}=r\,(w_t-c_t)+e^{y_t},
$$

$$
0 \le c_t \le w_t,
$$

$$
y_{t+1}=\rho y_t+\sigma\varepsilon_{t+1},\qquad \varepsilon_t\sim\mathcal{N}(0,1).
$$

**Assumptions:**
- Utility is CRRA: $u(c)=(c^{1-\gamma}-1)/(1-\gamma)$, $\gamma>0$.
- Discount factor $\beta\in(0,1)$.
- Exogenous income shock follows AR(1) with persistence $\rho$ and volatility $\sigma$; shocks are standard normal.
- Borrowing is not allowed: $c_t\le w_t$.
- Temporary income shock: $y_t=\sigma\varepsilon_t$, so the only state variable is $w_t$.
- $y_t$ is drawn from its ergodic (Normal) distribution.
- $w_t$ is drawn from a uniform distribution on $[w_1,w_2]$.
- This setup abstracts from convergence of simulated $w_t$ and focuses on convergence of coefficients $\theta$.

**Variables:**
- $c_t$: consumption at time $t$.
- $w_t$: cash-on-hand at time $t$.
- $w_{t+1}$: next-period cash-on-hand.
- $y_t$: Exogenous income shock (log income shock) at time $t$.
- $\varepsilon_t$: standard normal shock.
- $u(c_t)$: per-period utility from consumption.

**Parameters:**
- $\beta=0.9$
- $r=1.04$
- $\rho=0$
- $\gamma = 2$ (risk-aversion)
- $\sigma=0.1$ 
- $w_1, w_2$ (bounds of the uniform draw for $w_t$)

The policy is parameterized as a **consumption share** $c_t/w_t=\sigma(\zeta_0+\eta(y_t,w_t;\vartheta))$. 

The paper implements **three versions** of the DL method (lifetime reward, Euler, Bellman) and shows close agreement across them.

#### 2.3.2 Euler equation as the solution
The solution to the model can be characterized by **Kuhn–Tucker conditions**:

$$
A\ge 0,\quad H\ge 0,\quad AH=0,
$$

where $A=w-c$ and $H=u'(c)-\beta r \mathbb{E}_\varepsilon[u'(c')]$ is the Lagrange multiplier.  
To build a differentiable DL objective, the inequality system is rewritten as the **Fischer–Burmeister (FB) function**:

$$
\Psi^{FB}(a,h)=a+h-\sqrt{a^2+h^2}=0,
$$

with unit-free terms $a=1-\frac{c}{w}$ and $h=1-\frac{\beta r \mathbb{E}_\varepsilon[u'(c')]}{u'(c)}$.  

If needed, a weight $\nu>0$ can scale the second term, i.e., $\Psi^{FB}(a,\nu h)$, to balance the relative importance of the two objectives.

#### 2.3.3 Bellman equation as the solution
The same model can be characterized by the **Bellman equation**:

$$
V(y,w)=\max_{c,w'} \lbrace u(c)+\beta\,\mathbb{E}_\varepsilon[V(y',w')] \rbrace,
$$

subject to the transition and constraints in the model.  

The maximization operator can again be expressed through Kuhn–Tucker conditions, but now the multiplier is defined by the derivative of the value function,

$$
H=u'(c)-\beta r\,\mathbb{E}_\varepsilon[V_w(y',w')].
$$

The inequality constraints are transformed using the **the same FB function**, with

$$
a=1-\frac{c}{w},\qquad h=1-\frac{\beta r \mathbb{E}_\varepsilon[V_w(y',w')]}{u'(c)}.
$$

This provides a differentiable way to incorporate the Bellman maximization conditions into the DL objective.

#### 2.3.4 Deep learning solution method
To implement the lifetime-reward, Euler, and Bellman methods, the paper uses **Algorithm 1**; the only difference across methods is how the model is simulated in Step 2.1 and which network outputs are used (policy only, policy + multiplier, or policy + multiplier + value).

The decision rule, multiplier, and value function share a common neural network representation:

$$
\frac{c_t}{w_t}=\sigma\big(\zeta_0+\eta(y_t,w_t;\vartheta)\big)\equiv\varphi(y_t,w_t;\theta),
$$

$$
h_t=\exp\big(\zeta_0+\eta(y_t,w_t;\vartheta)\big)\equiv h(y_t,w_t;\theta),
$$

$$
V_t=\zeta_0+\eta(y_t,w_t;\vartheta)\equiv V(y_t,w_t;\theta),
$$

where $\eta(\cdot)$ is the neural network, $\theta=(\zeta_0,\vartheta)$, and $\sigma(x)=1/(1+e^{-x})$. 

The lifetime reward method will use only $\varphi(y_t,w_t;\theta)$.

The Euler method will use both $\varphi(y_t,w_t;\theta)$ and h(y_t,w_t;\theta).

The Bellman method will use all three of them.

The sigmoid bounds consumption shares in $[0,1]$, the exponential keeps $h_t\ge 0$, and the linear activation leaves $V_t$ unrestricted.

Training uses stochastic optimization with re-sampled simulation points each epoch, and accuracy is evaluated on a fresh sample with numerical integration.

**Parameters:**
- Network architecture: two identical hidden layers with (leaky) ReLU; widths compared are 8x8, 16x16, 32x32, and 64x64.
- Output heads: sigmoid for $c_t/w_t$, exponential for $h_t$, linear for $V_t$.
- Initialization: $\zeta_0=0$; remaining parameters $\vartheta$ are randomized (He-uniform for biases, Glorot-uniform for weights).
- Optimizer: Adam; learning rate $\lambda=0.001$.
- Training length: $K=50000$ epochs.
- Per-epoch training sample: 64 random draws of $w_t$ from $[w_1,w_2]=[0.1,4]$.
- Accuracy evaluation: 8192 random draws and a 10-node Gauss–Hermite rule for integrals.
- Implementation: Python with TensorFlow 1.14.0; Intel i7-7500U (2.70GHz), RAM 16GB, 4 physical (8 virtual) cores.

#### 2.3.5 Objective 1: Lifetime reward
This objective follows directly from the original problem: fix a decision rule $c_t/w_t=\varphi(y_t,w_t;\theta)$, simulate the model forward for a finite horizon $T$, and evaluate the discounted utility along the simulated path. 

Using the AiO operater, the expected loss becomes a single Monte Carlo expectation:

$$
\Xi(\theta)=E_{\omega}[\xi(\omega;\theta)] = E_{(y_0,w_0,\epsilon_1,\ldots,\epsilon_T)}\left[\sum_{t=0}^{T}\beta^t u(c_t)\right]
$$

**Random Draw:**
- $\omega=(y_0,w_0,\varepsilon_1,\ldots,\varepsilon_T)$, with $y_0$ drawn from its ergodic distribution, $w_0\sim U[w_1,w_2]$, and $\varepsilon_t\sim\mathcal{N}(0,1)$.

**Decision Rules:**
- $c_t/w_t=\varphi(y_t,w_t;\theta)$, with transitions of the model used to generate $(y_t,w_t)$ forward.

Algorithm 1 minimizes this by repeatedly simulating paths under $\varphi$ and updating $\theta$ with stochastic gradients.

#### 2.3.6 Objective 2: Euler equation (Kuhn–Tucker + AiO)
Start from the Kuhn–Tucker conditions for the borrowing constraint and rewrite them with the differentiable Fischer–Burmeister (FB) function. Because the Euler term contains a conditional expectation inside a nonlinear transformation, the paper introduces a separate approximation for the multiplier $h$, and then applies the AiO operator with two **independent** shocks to avoid nested expectations under the square. The resulting Euler objective is

$$
\Xi(\theta)=E_{(y,w,\varepsilon_1,\varepsilon_2)}\Bigg[\Psi^{FB} \left(1-\tfrac{c}{w},1-h\right)^2
+\nu_h\Big(\tfrac{\beta r\,u'(c')\big|_{\varepsilon_1}}{u'(c)}-h\Big)
\cdot\Big(\tfrac{\beta r\,u'(c')\big|_{\varepsilon_2}}{u'(c)}-h\Big)\Bigg].
$$

**Random Draw:**
- $\omega=(y,w,\varepsilon_1,\varepsilon_2)$, with $y$ drawn from its ergodic distribution, $w\sim U[w_1,w_2]$, and $\varepsilon_1,\varepsilon_2$ independent draws from $\mathcal{N}(0,1)$.

**Decision Rules:**
- $c/w=\varphi(y,w;\theta)$.
- $h=h(y,w;\theta)$.

#### 2.3.7 Objective 3: Bellman equation (residual + FB + AiO)
The Bellman method combines the Bellman residual with the maximization conditions, again enforced via the FB function. The multiplier now depends on the value-function derivative $V_w$, so the method approximates $V$, $\varphi$, and $h$. Applying the two-shock AiO construction to the squared residuals yields:

$$
\Xi(\theta)
=E_{\omega}[\xi(\omega;\theta)]
=E_{(y,w,\varepsilon_1,\varepsilon_2)}\Big[
\big(V(y,w;\theta)-u(c)-\beta V(y',w';\theta)\big)_{\varepsilon=\varepsilon_1}
\big(V(y,w;\theta)-u(c)-\beta V(y',w';\theta)\big)_{\varepsilon=\varepsilon_2}
+\nu\big(\Psi^{FB}(1-\tfrac{c}{w},1-h)\big)^2
+\nu_h\Big(\tfrac{\beta\,\partial_{w'}V(y',w';\theta)_{\varepsilon=\varepsilon_1}}{u'(c)}-h\Big)
\Big(\tfrac{\beta\,\partial_{w'}V(y',w';\theta)_{\varepsilon=\varepsilon_2}}{u'(c)}-h\Big)
\Big].
$$

**Random Draw:**
- $\omega=(y,w,\varepsilon_1,\varepsilon_2)$, with $y$ drawn from its ergodic distribution, $w\sim U[w_1,w_2]$, 

and $\varepsilon_1,\varepsilon_2$ independent draws from $\mathcal{N}(0,1)$.

**Decision Rules:**
- $V=V(y,w;\theta)$.
- $c/w=\varphi(y,w;\theta)$.
- $h=h(y,w;\theta)$.

## 3. Application 2: Krusell–Smith (1998) heterogeneous-agent model
### 3.1 Krusell and Smith (1998) model
Each heterogeneous agent $i=1,\ldots,\ell$ solves

$$
\max_{(c_t^i,k_{t+1}^i\)^{\infty}_{t\ge 0}} \; \mathbb{E}_0\left[\sum_{t=0}^{\infty}\beta^t u(c_t^i)\right],
$$

subject to

$$
w_{t+1}^i = R_{t+1}(w_t^i-c_t^i) + W_{t+1}\exp(y_{t+1}^i),
$$

$$
c_t^i \le w_t^i,
$$

$$
y_{t+1}^i = \rho_y y_t^i + \sigma_y \varepsilon_{t+1}^i,\qquad \varepsilon_t^i\sim\mathcal{N}(0,1),
$$

and aggregate production is Cobb–Douglas,

$$
Y_t = z_t k_t^{\alpha}[\sum_{i=1}^{\ell}\exp(y_t^i)\Big],
\qquad
z_{t+1}=\rho_z z_t+\sigma_z \varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,1).
$$

Equilibrium prices are

$$
R_t = 1-d + z_t \alpha k_t^{\alpha-1}\Big[\sum_{i=1}^{\ell}\exp(y_t^i)\Big],\qquad
W_t = z_t (1-\alpha) k_t^{\alpha}\Big[\sum_{i=1}^{\ell}\exp(y_t^i)\Big],
$$

where aggregate capital is $k_t=\sum_{i=1}^{\ell}k_t^i$ and $k_{t+1}^i=w_t^i-c_t^i$. 

Initial conditions $(y_0^i,w_0^i)$ and $z_0$ are given.

**Assumptions:**
- Agents are identical in fundamentals but differ in productivity and capital.
- Individual productivity follows an AR(1) process; aggregate productivity follows an AR(1) process.
- Competitive equilibrium prices $R_t,W_t$ are determined from the Cobb–Douglas technology.

**Variables:**
- $c_t^i$: consumption of agent $i$ at time $t$.
- $k_{t+1}^i = w_t^i - c_t^i$: next-period capital of agent $i$.
- $w_t^i$: cash-on-hand of agent $i$.
- $y_t^i$: idiosyncratic productivity of agent $i$.
- $\varepsilon_t^i$: idiosyncratic shock.
- $z_t$: aggregate productivity.
- $\varepsilon_t$: aggregate shock.
- $R_t$: gross interest rate.
- $W_t$: wage.
- $Y_t$: aggregate output.
- $k_t=\sum_{i=1}^{\ell}k_t^i$: aggregate capital.
- $\ell$: number of agents.
- $u(c_t^i)$: per-period utility of agent $i$.

**Parameters:**
- Preferences: $u(c)=\frac{c^{1-\gamma}-1}{1-\gamma}$ with $\gamma=1$.
- Discount factor: $\beta=0.96$.
- Capital share: $\alpha\in(0,1)$ (In the replication, we set $\alpha = 0.36$).
- Depreciation: $d\in(0,1]$ (In the replication, we set $d = 0.08$).
- Idiosyncratic shock: $\rho_y=0.9$, $\sigma_y=0.2(1-\rho_y^2)^{1/2}$.
- Aggregate shock: $\rho_z=0.95$, $\sigma_z=0.01$.

The policy uses a neural network that depends on individual and aggregate states.

Also, approximate aggregation is captured by a law of motion such as:
$\log K' = a_0+a_1\log K+a_2 z+a_3 z\log K$.

### 3.2 Deep learning solution algorithm
The Krusell–Smith implementation mirrors the consumption–saving case: the model is solved using the **lifetime-reward, Euler, and Bellman objectives** with Algorithm 1. The key difference is the **high-dimensional state** that includes all agents’ states plus the aggregate shock.

**State space:**
- Individual states $\{y_t^i,w_t^i\}_{i=1}^{\ell}$ and the aggregate shock $z_t$.
- Because agents are homogeneous in fundamentals, the decision and value functions are shared across agents; if agents differed, separate functions would be needed.

**Parameterization:**
Consumption share, multiplier, and value function are parameterized by a common neural network.

$$
\frac{c_t^i}{w_t^i}=\sigma \big(\zeta_0+\eta(y_t^i,w_t^i,D_t,z_t;\vartheta)\big)\equiv \varphi(\cdot;\theta),
$$

$$
h_t^i=\exp \big(\zeta_0+\eta(y_t^i,w_t^i,D_t,z_t;\vartheta)\big)\equiv h(\cdot;\theta),
$$

$$
V_t^i=\zeta_0+\eta(y_t^i,w_t^i,D_t,z_t;\vartheta)\equiv V(\cdot;\theta),
$$

where $D_t=\{y_t^i,w_t^i\}_{i=1}^{\ell}$, $\theta=(\zeta_0,\vartheta)$, and $\sigma(x)=1/(1+e^{-x})$.
- Sigmoid ensures $c_t^i/w_t^i\in[0,1]$; exponential keeps $h_t^i\ge 0$; value output is unrestricted.
- Baseline network uses two hidden layers with 64×64 neurons and sigmoid activation at the output.

**Simulation (Step 2.i in Algorithm 1):**
1. Given state ${w_t^i,y_t^i}_{i=1}^{\ell}$, $z_t$ and parameters $\theta$, 

compute $\frac{c_t^i}{w_t^i}=\varphi(y_t^i,w_t^i,D_t,z_t;\vartheta;\theta)$ and $k_{t+1}^i=w_t^i-c_t^i$.

2. Draw $y_{t+1}^i$ for all $i(1,...\ell)$ and $z_{t+1}$ using $y_{t+1}^i$  and $z_{t+1}$.
3. Compute prices $R_{t+1},W_{t+1}$ given $k_{t+1}=\sum_i k_{t+1}^i$.
4. Update cash-on-hand $w_{t+1}^i=R_{t+1}k_{t+1}^i+W_{t+1}\exp(y_{t+1}^i)$.
5. Compute $c_{t+1}^i/w_{t+1}^i=\varphi(y_{t+1}^i,w_{t+1}^i,D_{t+1},z_{t+1};\theta)$ for all $i$.
6. Evaluate the objective (lifetime reward, Euler, or Bellman), update $\theta$, and repeat.

**Training setup:**
- Simulate for $K=300000$ periods; update the network every 10th period.
- Each update uses 100 simulated points.
- Optimizer: Adam with learning rate $\lambda=0.001$.

**Notes on high dimension:**
- The state includes repeated variables (agent state and its appearance in the distribution $D_t$), creating perfect multicollinearity; neural networks handle this without explicit inversion.
- Model reduction occurs implicitly: feeding many state variables into the network compresses information into low-dimensional hidden layers (e.g., 64 neurons), analogous to principal-component compression.

### 3.3 Lifetime reward
For the Krusell–Smith model, the lifetime-reward objective follows directly from the general exposition. Using a composite draw that includes initial conditions and sequences of shocks, the objective is

$$
\Xi(\theta)=E_{\omega}\big[\xi(\omega;\theta)\big]\equiv
E_{(Y_0,W_0,z_0,\Sigma,\varepsilon)}\left[\sum_{t=0}^{T}\beta^t u(c_t^i)\right].
$$

Here $Y_0=(y_0^1,\ldots,y_0^{\ell})$, $W_0=(w_0^1,\ldots,w_0^{\ell})$, and $\Sigma$ collects idiosyncratic shocks for all agents over $t=1,\ldots,T$; $\varepsilon=(\varepsilon_1,\ldots,\varepsilon_T)$ are aggregate shock innovations.

Because the competitive equilibrium requires each agent’s utility to be maximized with respect to their own variables (not others’), the implementation **mutes** cross-agent gradients in TensorFlow. Since shocks are autocorrelated, training uses cross-sections sufficiently separated in time to reduce gradient bias.

**Random Draw:**
- $\omega=(Y_0,W_0,z_0,\Sigma,\varepsilon)$, where $Y_0=(y_0^1,\ldots,y_0^{\ell})$, $W_0=(w_0^1,\ldots,w_0^{\ell})$, $\Sigma=(\varepsilon_t^1,\ldots,\varepsilon_t^{\ell})_{t=1}^T$.

**Decision Rules:**
- $c_t^i/w_t^i=\varphi(y_t^i,w_t^i,D_t,z_t;\theta)$, with transitions of the model used to generate the simulated path.

### 3.4 Euler-equation method with Kuhn–Tucker conditions
The Euler objective is parallel to that of 2.2.6 in the consumption–saving problem. 

Applying the AiO operator with two uncorrelated shock draws yields:

$$
\Xi(\theta)=E_{Y_t,W_t,z_t,\Sigma_1,\Sigma_2,\varepsilon_1,\varepsilon_2}\Bigg[
\Big[\Psi^{FB} \left(1-\tfrac{c_t^i}{w_t^i},1-h_t^i\right)\Big]^2
+\nu\Big(\tfrac{\beta R_{t+1}u'(c_{t+1}^i)\big|_{\Sigma_1,\varepsilon_1}}{u'(c_t^i)}-h_t^i\Big)
\cdot\Big(\tfrac{\beta R_{t+1}u'(c_{t+1}^i)\big|_{\Sigma_2,\varepsilon_2}}{u'(c_t^i)}-h_t^i\Big)
\Bigg].
$$

Here $Y_t=(y_t^1,\ldots,y_t^{\ell})$, $W_t=(w_t^1,\ldots,w_t^{\ell})$, and $\Sigma_1,\Sigma_2$ are two independent draws of idiosyncratic shocks across agents; $\varepsilon_1,\varepsilon_2$ are two independent draws of aggregate shocks. Transitions follow the assumptions of the model.

**Random Draw:**
- $\omega=(Y_t,W_t,z_t,\Sigma_1,\Sigma_2,\varepsilon_1,\varepsilon_2)$, with $\Sigma_1,\Sigma_2$ independent draws of idiosyncratic shocks and $\varepsilon_1,\varepsilon_2$ independent draws of aggregate shocks.

**Decision Rules:**
- $c_t^i/w_t^i=\varphi(y_t^i,w_t^i,D_t,z_t;\theta)$.
- $h_t^i=h(y_t^i,w_t^i,D_t,z_t;\theta)$.

### 3.5 Objective 3: Bellman equation
The Bellman objective parallels that of 2.2.7 and combines the Bellman residual with the FB term and the multiplier condition

Using two independent shock draws:

$$
\Xi(\theta)
= E_{\omega}[\xi(\omega;\theta)]
= E_{(Y_t,W_t,z_t,\Sigma_1,\Sigma_2,\varepsilon_1,\varepsilon_2)}\Big[
\big(V(s_t^i;\theta)-u(c_t^i)-\beta V(s_{t+1}^i;\theta)\big)_{\Sigma_1,\varepsilon_1}
\big(V(s_t^i;\theta)-u(c_t^i)-\beta V(s_{t+1}^i;\theta)\big)_{\Sigma_2,\varepsilon_2}
+\nu\big(\Psi^{FB}(1-\tfrac{c_t^i}{w_t^i},1-h_t^i)\big)^2
+\nu_h\Big(\tfrac{\beta\,\partial_{w_{t+1}^i}V(s_{t+1}^i;\theta)_{\Sigma_1,\varepsilon_1}}{u'(c_t^i)}-h_t^i\Big)
\Big(\tfrac{\beta\,\partial_{w_{t+1}^i}V(s_{t+1}^i;\theta)_{\Sigma_2,\varepsilon_2}}{u'(c_t^i)}-h_t^i\Big)
\Big].
$$


Here $s_t^i$ denotes the vector of state variables for agent $i$, and all other notation follows (44). In the reported implementation, the value function is pre-trained for an initial block of iterations while holding the consumption and multiplier rules fixed.

**Random Draw:**
- $\omega=(Y_t,W_t,z_t,\Sigma_1,\Sigma_2,\varepsilon_1,\varepsilon_2)$, with $\Sigma_1,\Sigma_2$ independent idiosyncratic-shock draws and $\varepsilon_1,\varepsilon_2$ independent aggregate-shock draws.

**Decision Rules:**
- $V(s_t^i;\theta)$.
- $c_t^i/w_t^i=\varphi(y_t^i,w_t^i,D_t,z_t;\theta)$.
- $h_t^i=h(y_t^i,w_t^i,D_t,z_t;\theta)$.

## 4. Data and computational setup
- **No empirical data** are required; all training data are **simulated**.
- Training samples are drawn from the state space (or from simulated ergodic distributions in large models).
- Expectations over shocks are approximated via **Monte Carlo** using AiO.
- Optimization is carried out with **stochastic gradient descent** (mini-batches).
- AiO uses **two independent shock draws** per state when squared conditional expectations appear, avoiding nested integration.
- Training draws are **re-sampled during optimization** rather than held fixed, aligning the method with SGD.
- Reported consumption–saving implementation details: TensorFlow 1.14.0; Intel i7-7500U (2.70GHz), RAM 16GB; Adam with $\lambda=0.001$; $K=50000$ epochs; per-epoch sample of 64 draws; accuracy check via 8,192 random draws and a 10-node Gauss–Hermite rule.
- Reported Krusell–Smith implementation details: simulation length $K=300000$ periods, parameter updates every 10th period, 100 simulated points per update.

## 5. Main technical takeaways
- Dynamic models can be solved by **minimizing a single expected loss** derived from lifetime rewards, Euler residuals, or Bellman residuals.
- The **AiO operator** collapses nested expectations and makes large-scale models tractable.
- Neural networks offer a scalable, flexible approximation family for high-dimensional decision rules.
- **Unified loss construction** allows stacking constraints, FOCs, and equilibrium conditions in one objective, with weights to balance residuals.
- **Stochastic gradients are unbiased** under AiO; the expectation and gradient operators commute, enabling SGD with one (or a few) random draws.
- **Simulation-based training on the ergodic set** reduces the curse of dimensionality relative to fixed grids.
- **Two-shock AiO** handles squared conditional expectations by converting products of expectations into a single expectation over paired draws.
- **High-dimensional state compression** occurs implicitly through network hidden layers, helping with multicollinearity and repeated state variables.
