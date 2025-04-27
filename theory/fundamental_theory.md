# Beta-Neutral Portfolio Theory

## 1. Introduction
A **beta-neutral** (or market-neutral) portfolio aims to isolate alpha (idiosyncratic returns) by eliminating systematic market risk. This is achieved by constructing long and short positions whose combined sensitivity to market movements ($\beta$) is zero.

---

## 2. Key Definitions

- **Beta ($\beta$)**: Measures an asset’s sensitivity to market movements. For asset $i$:

  $$
  \beta_i = \frac{\mathrm{Cov}(r_i, r_m)}{\mathrm{Var}(r_m)}
  $$

  where $r_i$ is the return of asset $i$, and $r_m$ is the market return.

- **Portfolio Beta ($\beta_p$)**: For $N$ assets with weights $w_i$:

  $$
  \beta_p = \sum_{i=1}^N w_i \beta_i
  $$

- **Beta-Neutral Condition**:

  $$
  \beta_p = 0 \quad\Longrightarrow\quad \sum_{i=1}^N w_i \beta_i = 0
  $$

- **Alpha ($\alpha$)**: The intercept from the regression $r_{i,t} = \alpha + \beta_i r_{m,t} + \epsilon_t$, representing the asset’s return not explained by market movements.

  $$
  \alpha_i = E[r_i] - \beta_i E[r_m]
  $$

- **Volatility ($\sigma$)**: The standard deviation of an asset’s returns, capturing total risk.

  $$
  \sigma_i = \sqrt{\mathrm{Var}(r_i)}
  $$

- **Sharpe Ratio**: A measure of risk-adjusted return:

  $$
  \text{Sharpe} = \frac{E[r_p] - r_f}{\sigma_p}
  $$

  where $r_f$ is the risk-free rate.

- **Systematic Risk**: The portion of an asset’s risk explained by market movements (i.e., determined by $\beta$).

- **Unsystematic (Idiosyncratic) Risk**: The portion of an asset’s risk unique to the asset, not explained by market factors (the residual $\epsilon$).

---

## 3. Constructing a Beta-Neutral Portfolio

1. **Estimate Betas**
   - Use time-series regression: $r_{i,t} = \alpha + \beta_i r_{m,t} + \epsilon_t$.

2. **Determine Weights**
   - For a two-asset hedge (long asset 1, short asset 2):

     $$
     w_1 \beta_1 + w_2 \beta_2 = 0 \quad\Longrightarrow\quad w_2 = -w_1 \frac{\beta_1}{\beta_2}
     $$

   - For multiple assets, solve:

     $$
     \min_w \; \frac{1}{2} w^T \Sigma w - \lambda \mu^T w \quad \text{s.t.} \quad \beta^T w = 0,\; \mathbf{1}^T w = 1
     $$

     where $\Sigma$ is the covariance matrix and $\mu$ is the expected return vector.

---

## 4. Risk & Return Metrics

- **Portfolio Variance**:

  $$
  \sigma_p^2 = w^T \Sigma w
  $$

- **Value-at-Risk (VaR)** and **Conditional Value-at-Risk (CVaR)**:  
  Evaluate risk on the P&L distribution after removing market exposure.

---

## 5. Practical Considerations

- **Estimation Error**: Beta estimates are noisy — use rolling windows/shrinkage.
- **Leverage & Transaction Costs**: Beta-neutral portfolios often need leverage.
- **Rebalancing**: Maintain neutrality with frequent rebalancing, but monitor turnover.

---

*References*:
- Bodie, Kane & Marcus (Investments)
- Korn & Korn (Option Pricing and Portfolio Optimization)
