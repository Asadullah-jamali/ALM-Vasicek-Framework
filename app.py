
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ==============================================================================
# 1. MARKET DATA FETCHING & ANNUALIZED REGRESSION CALIBRATION
# ==============================================================================
ticker = "^TNX"

try:
    raw_data = yf.download(ticker, period="max", interval="1d")
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            df_clean = raw_data['Close'][ticker].to_frame()
        else:
            df_clean = raw_data[['Close']].copy()

        df_clean.columns = ['DGS10']
        df_clean.index.name = 'DATE'
        df = df_clean.dropna().copy()

        # ^TNX is quoted 10x basis points (45.00 = 4.5%). Divided by 1000 to fix scaling.
        df['Rate'] = df['DGS10'] / 100.0
        print("✓ Live market data fetched and parsed cleanly via Yahoo Finance!")
    else:
        raise ValueError("Data frame returned empty.")
except Exception as e:
    print(f"⚠️ Network restriction or data issue: {e}. Activating synthetic fallback matrix...")
    idx = pd.date_range(start='2010-01-01', end='2025-12-31', freq='B')
    df = pd.DataFrame({'DGS10': np.random.normal(35.0, 4.0, len(idx))}, index=idx)
    df.index.name = 'DATE'
    df['Rate'] = df['DGS10'] / 1000.0

df_calib = df[['Rate']].copy()
df_calib['r_t'] = df_calib['Rate']
df_calib['r_next'] = df_calib['Rate'].shift(-1)
df_calib = df_calib.dropna()

dt_daily = 1 / 252
X = df_calib['r_t'].values
y = df_calib['r_next'].values
beta, alpha = np.polyfit(X, y, 1)

k_calibrated = -np.log(beta) / dt_daily
theta_calibrated = alpha / (1 - beta)
residuals = y - (alpha + beta * X)
sigma_calibrated = np.std(residuals, ddof=2) * np.sqrt(252) * np.sqrt((2 * k_calibrated * dt_daily) / (1 - np.exp(-2 * k_calibrated * dt_daily)))
r0_real = df_calib['Rate'].iloc[-1]

print(f"Calibrated Parameters -> Speed (k): {k_calibrated:.4f} | Mean (theta): {theta_calibrated*100:.2f}% | Vol (sigma): {sigma_calibrated*100:.2f}%")

real_ssa_mortality = pd.Series({
    45: 0.0031, 46: 0.0034, 47: 0.0037, 48: 0.0041, 49: 0.0045,
    50: 0.0049, 51: 0.0054, 52: 0.0059, 53: 0.0064, 54: 0.0070,
    55: 0.0077, 56: 0.0084, 57: 0.0091, 58: 0.0099, 59: 0.0108
})

extended_mortality = real_ssa_mortality.to_dict()

# ==============================================================================
# 2. EXACT CONTINUOUS INTEREST RATE SCENARIO ENGINE
# ==============================================================================
class VasicekModel:
    def __init__(self, r0, k, theta, sigma, T, steps, simulations, seed=42):
        self.r0 = r0
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.simulations = simulations
        self.dt = T / steps
        self.rng = np.random.default_rng(seed)

    def generate_paths(self):
        rates = np.zeros((self.steps + 1, self.simulations))
        rates[0, :] = self.r0
        exp_k = np.exp(-self.k * self.dt)
        exact_vol = self.sigma * np.sqrt((1 - np.exp(-2 * self.k * self.dt)) / (2 * self.k))

        for t in range(1, self.steps + 1):
            Z = self.rng.normal(0, 1, self.simulations)
            rates[t, :] = rates[t-1, :] * exp_k + self.theta * (1 - exp_k) + exact_vol * Z
        return rates

    def price_zcb(self, r_t, tau):
        tau = np.maximum(tau, 0.0)

        if np.isscalar(tau):
            if tau < 1e-9:
                return np.ones_like(r_t)
            B = (1 - np.exp(-self.k * tau)) / self.k
            A = np.exp((self.theta - (self.sigma**2) / (2 * self.k**2)) * (B - tau) - (self.sigma**2 * (B**2)) / (4 * self.k))
        else:
            mask = tau > 1e-9
            # FIX: Ensure array initialization maps to r_t shape to prevent broadcast shape errors
            A = np.ones_like(r_t)
            B = np.zeros_like(r_t)
            B_mask = (1 - np.exp(-self.k * tau[mask])) / self.k
            A_mask = np.exp((self.theta - (self.sigma**2) / (2 * self.k**2)) * (B_mask - tau[mask]) - (self.sigma**2 * (B_mask**2)) / (4 * self.k))
            A[mask] = A_mask
            B[mask] = B_mask

        return A * np.exp(-B * r_t)

# ==============================================================================
# 3. DYNAMIC ASSET PORTFOLIO WITH WEEKLY LIABILITY DEDUCTIONS (ALM INTEGRATED)
# ==============================================================================
class DynamicBondPortfolio:
    def __init__(self, face_value, maturity, model, tx_fee_rate=0.0015):
        self.face_value = face_value
        self.initial_maturity = maturity
        self.model = model
        self.tx_fee_rate = tx_fee_rate

    def simulate_managed_portfolio(self, rate_paths, expected_payouts_weekly, target_cash_buffer=0.05):
        steps_plus_1, simulations = rate_paths.shape
        steps = steps_plus_1 - 1
        dt = self.model.dt

        portfolio_wealth = np.zeros((steps + 1, simulations))
        portfolio_wealth[0, :] = self.face_value

        cash_balance = portfolio_wealth[0, :] * target_cash_buffer
        bond_allocation = portfolio_wealth[0, :] * (1.0 - target_cash_buffer)

        for t in range(1, steps + 1):
            tau_prev = self.initial_maturity - (t - 1) * dt
            tau_curr = self.initial_maturity - t * dt

            p_prev = self.model.price_zcb(rate_paths[t-1, :], tau_prev)
            p_curr = self.model.price_zcb(rate_paths[t, :], tau_curr)
            bond_returns = p_curr / p_prev

            cash_balance *= np.exp(rate_paths[t-1, :] * dt)
            bond_allocation *= bond_returns

            # CORE ALM FIX: Deduct liability payouts directly from cash balance each week
            # expected_payouts_weekly indices align 1:1 with time step indices
            payout_today = expected_payouts_weekly[t-1]
            cash_balance -= payout_today

            gross_wealth = cash_balance + bond_allocation

            # Rebalance asset allocations to handle cash buffer constraints
            target_bonds_pre_cost = gross_wealth * (1.0 - target_cash_buffer)
            bond_trade_volume = np.abs(target_bonds_pre_cost - bond_allocation)
            transaction_costs = bond_trade_volume * self.tx_fee_rate

            net_wealth = gross_wealth - transaction_costs
            cash_balance = net_wealth * target_cash_buffer
            bond_allocation = net_wealth * (1.0 - target_cash_buffer)

            portfolio_wealth[t, :] = net_wealth

        return portfolio_wealth


# ==============================================================================
# 4. LIABILITIES MANAGEMENT CLASS
# ==============================================================================
class InsuranceLiabilities:
    def __init__(self, initial_policies, sum_assured, mortality_table):
        self.initial_policies = initial_policies
        self.sum_assured = sum_assured
        self.mortality_table = mortality_table

    def project_cash_flows_weekly(self, start_age, projection_years, steps_per_year=52):
        total_steps = projection_years * steps_per_year
        expected_payouts = []
        active_policies = self.initial_policies
        dt_step = 1 / steps_per_year

        for step in range(total_steps):
            current_age = int(start_age + (step * dt_step))
            qx_annual = self.mortality_table.get(current_age, 0.01)
            qx_step = qx_annual / steps_per_year 

            deaths = active_policies * qx_step
            payout = deaths * self.sum_assured
            expected_payouts.append(payout)
            active_policies -= deaths

        return np.array(expected_payouts)

# ==============================================================================
# 5. HIGH-CAPACITY SIMULATION PIPELINE & ANALYSIS RUN
# ==============================================================================
simulations = 10000
years = 10
steps = years * 52  

rate_engine = VasicekModel(
    r0=r0_real, 
    k=k_calibrated, 
    theta=theta_calibrated, 
    sigma=sigma_calibrated, 
    T=years, 
    steps=steps, 
    simulations=simulations
)

simulated_rates = rate_engine.generate_paths()

bonds = DynamicBondPortfolio(face_value=50_000_000, maturity=years, model=rate_engine, tx_fee_rate=0.0015)
liabilities = InsuranceLiabilities(initial_policies=10000, sum_assured=120000, mortality_table=extended_mortality)

expected_payouts_weekly = liabilities.project_cash_flows_weekly(start_age=45, projection_years=years, steps_per_year=52)

# Portfolio simulation now factors in dynamic liability cash outflows automatically
asset_wealth_paths = bonds.simulate_managed_portfolio(simulated_rates, expected_payouts_weekly, target_cash_buffer=0.05)
final_asset_vals = asset_wealth_paths[-1, :]

# Because claims are completely met during the 10-year term, terminal active liabilities are 0.
final_liab_vals = np.zeros(simulations)
surplus = final_asset_vals - final_liab_vals

# ==============================================================================
# 6. RISK METRICS REPORTING & VISUALIZATION PIPELINE
# ==============================================================================
print(f"\n================ SOLVENCY ANALYSIS COMPLETE ================")
print(f"Total Simulation Paths   : {simulations:,}")
print(f"Mean Final Asset Value   : ${final_asset_vals.mean():,.2f}")
print(f"Total Liability Paid out : ${np.sum(expected_payouts_weekly):,.2f}")
print(f"Mean Net Economic Surplus: ${surplus.mean():,.2f}")
print(f"Capital Shortfall Prob   : {(surplus < 0).mean() * 100:.2f}%")
print(f"============================================================")

plt.figure(figsize=(11, 5.5))
plt.hist(surplus, bins=50, color='teal', edgecolor='black', alpha=0.7, density=False)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Solvency Threshold ($0)')
plt.title("Net Solvency Surplus Distribution at Terminal Horizon (Year 10)", fontsize=12, fontweight='bold')
plt.xlabel("Surplus Value ($)", fontsize=10)
plt.ylabel("Path Frequency", fontsize=10)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
