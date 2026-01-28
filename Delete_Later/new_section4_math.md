---
title: "Section 4 Math Notes"
paper: "Deep learning for solving dynamic economic models (Maliar, Maliar, Winant, 2021, JME)"
scope: "Section 4 — Consumption–Saving problem"
language: "English"
---

# Section 4 Math Notes (All Equations)

Note: Equation numbers follow the paper’s Section 4 labels where available.
Symbols are reconstructed where the screenshot is ambiguous, with short
replication notes to clarify implementation choices.

## 4.1 Consumption–saving model

### (20) Problem
$$
\max_{\{c_t,w_{t+1}\}_{t\ge 0}}\;\mathbb{E}_0\left[\sum_{t=0}^{\infty}\beta^t u(c_t)\right]
$$
subject to
$$
w_{t+1}=r\,(w_t-c_t)+e^{y_t},
$$
$$
c_t\le w_t,
$$
$$
y_{t+1}=\rho y_t+\sigma\varepsilon_{t+1},\qquad \varepsilon_t\sim\mathcal{N}(0,1).
$$
- $w_t$ is cash-on-hand, $c_t$ consumption, $r$ the gross interest rate.
- The borrowing limit is set to zero, so $c_t\le w_t$.
- Utility is CRRA: $u(c)=(c^{1-\gamma}-1)/(1-\gamma)$, $\gamma>0$.

### Expository simplifications used in the paper
- Temporary shock: set $\rho=0$ so $y_t=\sigma\varepsilon_t$ with $\sigma=0.1$.
- Parameters used in the reported experiments: $\beta=0.9$, $r=1.04$.
- For training, $w$ is drawn from a broad interval (e.g., $[w_1,w_2]=[0.1,4]$)
  to cover the relevant ergodic set.

## 4.2 Deep learning parameterization

### (NN outputs)
The network inputs are $(y_t,w_t)$, and a shared hidden representation
$\eta(y_t,w_t;\vartheta)$ feeds three outputs:
$$
\frac{c_t}{w_t}=\sigma\big(\zeta_0+\eta(y_t,w_t;\vartheta)\big)\equiv\varphi(y_t,w_t;\theta),
$$
$$
h_t=\exp\big(\zeta_0+\eta(y_t,w_t;\vartheta)\big)\equiv h(y_t,w_t;\theta),
$$
$$
V_t=\zeta_0+\eta(y_t,w_t;\vartheta)\equiv V(y_t,w_t;\theta).
$$
- Sigmoid keeps $c_t/w_t\in(0,1)$ and hence $c_t\in(0,w_t)$.
- Exponential keeps $h_t\ge 0$.
- Two hidden layers with (leaky) ReLU are used; widths compared: 8x8, 16x16,
  32x32, 64x64.

Replication notes:
- Hidden layers use (leaky) ReLU; output heads use sigmoid/exp/linear as above.
- Lifetime reward uses only $\varphi$; Euler uses $\varphi$ and $h$; Bellman uses
  $\varphi$, $h$, and $V$.
- Initialize $\zeta_0=0$.
- Initialize remaining weights with He/Glorot uniform (as stated in the text).
- Use ADAM with learning rate $\lambda_k=0.001$ and $K=50{,}000$ epochs.
- Each epoch: draw 64 random points from $w\in[0.1,4]$; accuracy is assessed
  on 8,192 test points and 10-node Gauss–Hermite integration.
- Implementation details reported: TensorFlow 1.14.0 on a laptop CPU (i7-7500U,
  16GB RAM, 4 physical / 8 virtual cores).

## 4.3 Objective 1: Lifetime reward

### (27) Lifetime reward objective
$$
\Xi(\theta)=\mathbb{E}_{\omega}\left[\sum_{t=0}^{T}\beta^t u(c_t)\right],
\quad \omega=(y_0,w_0,\varepsilon_1,\ldots,\varepsilon_T).
$$
- Use $c_t/w_t=\varphi(y_t,w_t;\theta)$ and transitions from (20).
- Monte Carlo: simulate forward with the policy, then average the discounted
  utility across paths.

## 4.4 Objective 2: Euler equation with borrowing constraint

### (24) Kuhn–Tucker conditions
$$
A\ge 0,\quad H\ge 0,\quad AH=0,
$$
where
$$
A=w-c,\qquad H=u'(c)-\beta r\,\mathbb{E}_\varepsilon[u'(c')].
$$
- $H$ is the Lagrange multiplier on the borrowing constraint.

### (25) Fischer–Burmeister (FB) function
Define unit-free terms
$$
a=1-\frac{c}{w},\qquad h=1-\frac{\beta r\,\mathbb{E}_\varepsilon[u'(c')]}{u'(c)},
$$
and
$$
\Psi^{FB}(a,h)=a+h-\sqrt{a^2+h^2}=0.
$$
- Optionally use a weight $\nu>0$: $\Psi^{FB}(a,\nu h)$.

### (28) Euler objective (squared FB residual)
$$
\mathbb{E}_{y,w}\left[\Psi^{FB}\left(1-\tfrac{c}{w},\;1-\tfrac{\beta r\,\mathbb{E}_\varepsilon[u'(c')]}{u'(c)}\right)^2\right].
$$

### (29) Composite objective with explicit multiplier
Introduce a separate approximation $h(y,w;\theta)$ and use
$$
\mathbb{E}_{y,w}\left[\Psi^{FB}\left(1-\tfrac{c}{w},1-h\right)^2\right]
+\nu_h\left[\frac{\beta r\,\mathbb{E}_\varepsilon[u'(c')]}{u'(c)}-h\right]^2.
$$

### (30) AiO objective with two independent shocks
$$
\Xi(\theta)=\mathbb{E}_{\omega}\Bigg[\Psi^{FB}\left(1-\tfrac{c}{w},1-h\right)^2
+\nu_h\Big(\tfrac{\beta r\,u'(c')}{u'(c)}\big|_{\varepsilon_1}-h\Big)
\Big(\tfrac{\beta r\,u'(c')}{u'(c)}\big|_{\varepsilon_2}-h\Big)\Bigg],
$$
with $\omega=(y,w,\varepsilon_1,\varepsilon_2)$.

Replication notes:
- Use two independent draws $\varepsilon_1,\varepsilon_2$ for the AiO product.
- Train with $c/w=\varphi(y,w;\theta)$ and $h=h(y,w;\theta)$.

## 4.5 Objective 3: Bellman equation

### (26) Bellman equation
$$
V(y,w)=\max_{c,w'}\left\{u(c)+\beta\,\mathbb{E}_\varepsilon[V(y',w')]\right\},
$$
subject to the constraints and transition in (20).

### (31) Bellman objective (squared residual + FB)
$$
\mathbb{E}_{y,w}\Big[V(y,w;\theta_1)-u(c)-\beta\mathbb{E}_\varepsilon V(y',w';\theta_1)\Big]^2
+\nu\,\mathbb{E}_{y,w}\left[\Psi^{FB}\left(1-\tfrac{c}{w},\;1-\frac{\beta\,\mathbb{E}_\varepsilon[\partial_{w'}V(y',w';\theta_1)]}{u'(c)}\right)^2\right].
$$

### (32) Bellman AiO objective with uncorrelated shocks
Introduce $h(y,w;\theta)$ and define
$$
\Xi(\theta)=\mathbb{E}_{\omega}\Big[\big(R_{\varepsilon_1}\big)\big(R_{\varepsilon_2}\big)
+\Psi^{FB}\left(1-\tfrac{c}{w},1-h\right)^2
+\nu_h\big(G_{\varepsilon_1}\big)\big(G_{\varepsilon_2}\big)\Big],
$$
where
$$
R_{\varepsilon}=V(y,w;\theta_1)-u(c)-\beta V(y',w';\theta_1)\big|_{\varepsilon},
$$
$$
G_{\varepsilon}=\frac{\beta\,\partial_{w'}V(y',w';\theta_1)}{u'(c)}\bigg|_{\varepsilon}-h.
$$
- Two independent shocks avoid nested expectations in the squared terms.

## 4.6 Implementation details that affect replication
- Use $y'$ in $w'=r(w-c)+e^{y'}$ when evaluating Bellman/Euler residuals.
- The paper reports sensitivity to weights ($\nu,\nu_h$); tune them so the
  residual magnitudes are comparable across terms.
