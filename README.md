An institutional-grade Quantitative Finance framework designed to execute dynamic solvency risk stress tests. The pipeline calibrates a continuous-time **Vasicek Short-Rate Model** against live 10-Year Treasury Yield data (`^TNX`) via Yahoo Finance, simulating macroeconomic trajectories while simultaneously evaluating capital erosion against systemic actuarial liability drains.

## 🔗 Live Deployment & Framework Links
* **Interactive Web Application**: [Explore the Live Streamlit Dashboard](https://streamlit.app) *(Update with your specific ALM app link once deployed)*
* **Core Notebook Workspace**: [View Interactive Jupyter Notebook](./ALM_Solvency_Engine.ipynb)

---

## 📐 Core Architecture & Mathematical Specification

### 1. Macroeconomic Engine (Vasicek Short-Rate Model)
The instantaneous interest rate \(r_t\) is modeled using the mean-reverting Vasicek Stochastic Differential Equation (SDE):

$$dr_t = k(\theta - r_t)dt + \sigma dW_t$$

Where:
* **$k$**: Speed of mean reversion
* **$\theta$**: Long-term mean yield level
* **$\sigma$**: Instantaneous volatility parameter
* **$dW_t$**: Standard Brownian motion increment


The framework performs an Ordinary Least Squares (OLS) regression on historic daily interest shifts to extract calibrated, annualized parameters (\(k, \theta, \sigma\)) cleanly without manual tuning.

### 2. Exact Zero-Coupon Bond Pricing Matrix
To eliminate numerical discretization drift common in Euler-Maruyama approximations, asset pricing transforms along generated paths via analytical transition densities:

$$P(t, T) = A(t, T)e^{-B(t, T)r_t}$$

Where $A(t, T)$ and $B(t, T)$ are deterministic functions derived explicitly from the calibrated framework matrices:

$$B(t, T) = \frac{1 - e^{-k(T-t)}}{k}$$

$$A(t, T) = \exp \left( \left(\theta - \frac{\sigma^2}{2k^2}\right)(B(t, T) - (T-t)) - \frac{\sigma^2 B(t, T)^2}{4k} \right)$$


### 3. Actuarial Claims Drain Pipeline
Future capital outfluxes are modeled as continuous weekly structural liability drains mapped from Social Security Administration (SSA) mortality records:

$$\text{Expected Claims}_t = \text{Active Policies}_t \times \left( \frac{q_x}{52} \right) \times \text{Sum Assured}$$


---

## 📂 Repository Layout

```text
├── .gitignore                  # Institutional Python & Jupyter cache exclusions
├── ALM-vasicek-solvency-engine.ipynb   # Complete research development code notebook workflow
├── app.py                      # Production-ready Streamlit user interface engine
└── requirements.txt            # Explicit dependency version pinning matrix
```

---

## 📊 Key Operational Metrics Tracked
* **Mean Terminal Economic Surplus**: The expected capital cushion left across 10,000 parallel Monte Carlo runs after fulfilling multi-year continuous insurance payouts.
* **Aggregate Claims Liquidated**: The true capital scale distributed back to policyholders over the operational cycle.
* **Capital Shortfall Risk (Value at Risk)**: The mathematically derived percentage of simulated trajectories where asset liquidation boundaries breach zero, resulting in structural insolvency ($\text{Surplus} < \$0$).

---

## 💻 Local Installation & Initialization

1. **Clone the Repository**:
   ```bash
   git clone https://github.com
   cd YOUR_REPOSITORY_NAME
   ```

2. **Deploy the Environment Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the App Engine Locally**:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Technological Infrastructure
* **Core Language**: Python 3.12+
* **Data Sourcing API**: `yfinance` (Automated Federal Reserve 10-Year Yield extraction)
* **Mathematical Vectorization**: `numpy` (Dense execution math), `pandas`
* **Visualization Layer**: `matplotlib` (Distribution histograms & path charts)
* **Cloud Interface**: `streamlit` (Highly-scalable cloud dashboard application deployment)
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✓ Fixed README.md has been generated with native GitHub markdown ($ and $$) math delimiters!")
