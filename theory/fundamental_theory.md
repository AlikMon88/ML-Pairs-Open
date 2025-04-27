# Beta-Neutral Portfolio Theory

## 1. Introduction
A **beta-neutral** (or market-neutral) portfolio aims to isolate alpha (idiosyncratic returns) by eliminating systematic market risk. This is achieved by constructing long and short positions whose combined sensitivity to market movements ($\beta$) is zero.

---

## 2. Key Definitions

- **Beta ($\beta$)**: Measures an asset’s sensitivity to market movements. For asset $i$:

  `βᵢ = Cov(rᵢ, rₘ) / Var(rₘ)`

  where $rᵢ$ is the return of asset $i$, and $rₘ$ is the market return.

- **Portfolio Beta ($\beta_p$)**: For $N$ assets with weights $wᵢ$:

  `βₚ = Σ(wᵢ * βᵢ)`

- **Beta-Neutral Condition**:

  `βₚ = 0  →  Σ(wᵢ * βᵢ) = 0`

- **Alpha ($\alpha$)**: The intercept from the regression `rᵢ,t = α + βᵢ * rₘ,t + εₜ`, representing the asset’s return not explained by market movements.

  `αᵢ = E[rᵢ] - βᵢ * E[rₘ]`

- **Volatility ($σ$)**: The standard deviation of an asset’s returns, capturing total risk.

  `σᵢ = sqrt(Var(rᵢ))`

- **Sharpe Ratio**: A measure of risk-adjusted return:

  `Sharpe = (E[rₚ] - r_f) / σₚ`

  where $r_f$ is the risk-free rate.

- **Systematic Risk**: The portion of an asset’s risk explained by market movements (i.e., determined by $\beta$).

- **Unsystematic (Idiosyncratic) Risk**: The portion of an asset’s risk unique to the asset, not explained by market factors (the residual $ε$).

---

## 3. Constructing a Beta-Neutral Portfolio

1. **Estimate Betas**
   - Use time-series regression: `rᵢ,t = α + βᵢ * rₘ,t + εₜ`.

2. **Determine Weights**
   - For a two-asset hedge (long asset 1, short asset 2):

     `w₁ * β₁ + w₂ * β₂ = 0  →  w₂ = -w₁ * (β₁ / β₂)`

   - For multiple assets, solve the optimization problem:

     `min_w  ½ * wᵀ * Σ * w - λ * μᵀ * w  s.t.  βᵀ * w = 0,  1ᵀ * w = 1`

     where `Σ` is the covariance matrix of asset returns, and `μ` is the expected returns vector.

---

## 4. Risk & Return Metrics

- **Portfolio Variance**:

  `σₚ² = wᵀ * Σ * w`

- **Value-at-Risk (VaR)** and **Conditional Value-at-Risk (CVaR)**:
  - Evaluate risk on the portfolio’s P&L distribution after removing market exposure.

---

## 5. Practical Considerations

- **Estimation Error**: Beta estimates are noisy — use rolling windows or shrinkage techniques.
- **Leverage & Transaction Costs**: Beta-neutral portfolios often need leverage. Transaction costs must be considered during rebalancing.
- **Rebalancing**: Frequent rebalancing is needed to maintain neutrality, but this can increase turnover and transaction costs.

---

## References:
- Bodie, Kane & Marcus — *Investments*
- Korn & Korn — *Option Pricing and Portfolio Optimization*
