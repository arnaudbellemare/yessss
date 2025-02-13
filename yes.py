import pandas as pd
import numpy as np
from scipy.stats import t as studentt
import matplotlib.pyplot as plt
import ccxt
import streamlit as st
import matplotlib.dates as mdates

# ---------------------------
# SECTION 1: Real Data Analysis using Kraken via ccxt
# ---------------------------

# Define lookback options (in minutes)
lookback_options = {
    "1 Day": 1440,
    "3 Days": 4320,
    "1 Week": 10080,
    "2 Weeks": 20160,
    "1 Month": 43200
}

# Fetch real OHLCV data from Kraken
def fetch_data(symbol, timeframe="1m", lookback_minutes=1440):
    exchange = ccxt.kraken({'enableRateLimit': True})
    since = exchange.milliseconds() - lookback_minutes * 60 * 1000
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {e}")
    if not ohlcv:
        raise ValueError(f"No data returned for {symbol}.")
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    print("Fetched data columns:", df.columns)
    print("First rows of fetched data:", df.head())
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# Define the HawkesBVC class (custom metric estimation)
class HawkesBVC:
    def __init__(self, window: int, kappa: float = None, dof=0.25, decays=None):
        """
        :param window: Lookback window for volatility calculation.
        :param kappa: Decay factor (if None, will be learned from data).
        :param dof: Degrees-of-freedom for Student-t distribution.
        :param decays: List of decay rates for fallback estimation.
        """
        self._window = window
        self._kappa = kappa
        self._dof = dof
        self.decays = decays if decays else [0.1, 0.5, 1.0]
        self.metrics = None

    def eval(self, df: pd.DataFrame, scale=1e4):
        times = df['stamp']
        prices = df['close']
        cumr = np.log(prices / prices.iloc[0])
        r = cumr.diff().fillna(0.0)
        volume = df['volume']
        sigma = r.rolling(self._window).std().fillna(0.0)
        labels = np.array([self._label(r.iloc[i], sigma.iloc[i]) for i in range(len(r))])
        if self._kappa is None:
            self._kappa = self._learn_kappa(times, labels, volume)
        alpha_exp = np.exp(-self._kappa)
        bvc = np.zeros(len(volume), dtype=float)
        current_bvc = 0.0
        for i in range(len(volume)):
            current_bvc = current_bvc * alpha_exp + volume.iloc[i] * labels[i]
            bvc[i] = current_bvc
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc / np.max(np.abs(bvc)) * scale
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': bvc})
        return self.metrics

    def _label(self, r: float, sigma: float):
        if sigma > 0.0:
            cum = studentt.cdf(r / sigma, df=self._dof)
            return 2 * cum - 1.0
        else:
            return 0.0

    def _learn_kappa(self, times, labels, volume):
        mask = (labels > 0.5) | (labels < -0.5)
        if mask.sum() < 2:
            estimated_kappa = np.mean(self.decays)
        else:
            selected_times = pd.to_datetime(times[mask])
            selected_seconds = selected_times.astype(np.int64) // 10**9
            selected_seconds = np.sort(selected_seconds)
            time_diffs = np.diff(selected_seconds)
            if len(time_diffs) == 0 or np.mean(time_diffs) == 0:
                estimated_kappa = np.mean(self.decays)
            else:
                avg_diff = np.mean(time_diffs)
                estimated_kappa = 1.0 / avg_diff
        print(f"Estimated kappa: {estimated_kappa}")
        return estimated_kappa

    def plot(self):
        import plotnine as p9
        return (
            p9.ggplot(self.metrics, p9.aes(x='stamp', y='bvc'))
            + p9.geom_line(color='blue', size=0.5)
            + p9.labs(title="B/S Imbalance", x="Time", y="BVC")
            + p9.theme_minimal()
            + p9.theme(figure_size=(11, 5))
        )

def ema(data, window):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

# Streamlit App Layout for Section 1
st.header("Section 1: Real Data Analysis")

symbol_bsi1 = st.sidebar.text_input(
    "Enter Ticker Symbol (Kraken Format, e.g. XBT/USD)",
    value="BTC/USD",
    key="symbol_bsi1"
)

lookback_label_bsi1 = st.sidebar.selectbox(
    "Select Lookback Period",
    list(lookback_options.keys()),
    key="lookback_label_bsi1"
)
limit_bsi1 = lookback_options[lookback_label_bsi1]

st.write(f"Fetching data for **{symbol_bsi1}** with a lookback of **{limit_bsi1}** minutes.")

try:
    prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe="1m", lookback_minutes=limit_bsi1)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

prices_bsi.dropna(subset=['close', 'volume'], inplace=True)
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])
prices_bsi.set_index('stamp', inplace=True)

# Data transformations
prices_bsi['ScaledPrice'] = np.log(prices_bsi['close'] / prices_bsi['close'].iloc[0]) * 1e4
prices_bsi['ScaledPrice_EMA'] = ema(prices_bsi['ScaledPrice'].values, window=10)
prices_bsi['cum_vol'] = prices_bsi['volume'].cumsum()
prices_bsi['cum_pv'] = (prices_bsi['close'] * prices_bsi['volume']).cumsum()
prices_bsi['vwap'] = prices_bsi['cum_pv'] / prices_bsi['cum_vol']
prices_bsi['vwap_transformed'] = np.log(prices_bsi['vwap'] / prices_bsi['vwap'].iloc[0]) * 1e4

if 'buyvolume' not in prices_bsi.columns or 'sellvolume' not in prices_bsi.columns:
    prices_bsi['buyvolume'] = prices_bsi['volume'] * 0.5
    prices_bsi['sellvolume'] = prices_bsi['volume'] - prices_bsi['buyvolume']

st.write("## Skewness and BVC Analysis")
df_skew = prices_bsi.copy()
df_skew['hlc3'] = (df_skew['high'] + df_skew['low'] + df_skew['close']) / 3.0
SkewLength = 14
alpha_val = 2.0 / (1.0 + SkewLength)
df_skew['TrueRange'] = (np.abs(df_skew['hlc3'] - df_skew['hlc3'].shift(1, fill_value=df_skew['hlc3'].iloc[0]))
                         / df_skew['hlc3'].shift(1, fill_value=df_skew['hlc3'].iloc[0]))
dev_max_series, dev_min_series = [], []
dev_max_prev, dev_min_prev = 1.618, 1.618
for i in range(len(df_skew)):
    if i == 0:
        dev_max_series.append(dev_max_prev)
        dev_min_series.append(dev_min_prev)
    else:
        current_tr = df_skew['TrueRange'].iloc[i]
        prior_hlc3 = df_skew['hlc3'].iloc[i - 1]
        current_hlc3 = df_skew['hlc3'].iloc[i]
        if current_hlc3 > prior_hlc3:
            dev_max_prev = alpha_val * current_tr + (1 - alpha_val) * dev_max_prev
        else:
            dev_max_prev = (1 - alpha_val) * dev_max_prev
        if current_hlc3 < prior_hlc3:
            dev_min_prev = alpha_val * current_tr + (1 - alpha_val) * dev_min_prev
        else:
            dev_min_prev = (1 - alpha_val) * dev_min_prev
        dev_max_series.append(dev_max_prev)
        dev_min_series.append(dev_min_prev)
df_skew['deviation_max'] = dev_max_series
df_skew['deviation_min'] = dev_min_series
df_skew['normalized_skew'] = (df_skew['deviation_max'] / df_skew['deviation_min'] - 1) * 3
df_skew['normalized_z'] = (df_skew['normalized_skew'] + 3) / 6
df_skew['normalized_z'] = df_skew['normalized_z'].ffill().bfill()
df_skew['ScaledPrice'] = np.log(df_skew['close'] / df_skew['close'].iloc[0]) * 1e4
ema_window = 10
df_skew['ScaledPrice_EMA'] = ema(df_skew['ScaledPrice'].values, ema_window)

hawkes_bvc = HawkesBVC(window=20, kappa=0.1)
bvc_metrics = hawkes_bvc.eval(prices_bsi.reset_index())
df_skew = df_skew.merge(bvc_metrics, on='stamp', how='left')
global_min = df_skew['ScaledPrice'].min()
global_max = df_skew['ScaledPrice'].max()

fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=120)
norm_bvc = plt.Normalize(df_skew['bvc'].min(), df_skew['bvc'].max())

# Plot the price line with BVC color coding
for i in range(len(df_skew['stamp']) - 1):
    xvals = df_skew['stamp'].iloc[i:i+2]
    yvals = df_skew['ScaledPrice'].iloc[i:i+2]
    bvc_val = df_skew['bvc'].iloc[i]
    cmap_bvc = plt.cm.Blues if bvc_val >= 0 else plt.cm.Reds
    color = cmap_bvc(norm_bvc(bvc_val))
    ax1.plot(xvals, yvals, color=color, linewidth=1)

# Plot EMA of ScaledPrice
ax1.plot(df_skew['stamp'], df_skew['ScaledPrice_EMA'], color='gray', linewidth=0.7, label=f"EMA({ema_window})")

# Overlay the VWAP with conditional coloring
# (blue if price is above VWAP, red if below)
for i in range(len(df_skew['stamp']) - 1):
    xvals = df_skew['stamp'].iloc[i:i+2]
    # Compare the price and the transformed VWAP at the start of the segment
    if df_skew['ScaledPrice'].iloc[i] >= df_skew['vwap_transformed'].iloc[i]:
        color = 'blue'
    else:
        color = 'red'
    # Add label only for the first segment to avoid duplicate legend entries
    ax1.plot(xvals, df_skew['vwap_transformed'].iloc[i:i+2], color=color, linewidth=0.7, 
             label="VWAP" if i == 0 else None)

ax1.set_xlabel("Time", fontsize=8)
ax1.set_ylabel("ScaledPrice", fontsize=8)
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax1.get_yticklabels(), fontsize=7)
ax1.legend(fontsize=7)
ax1.text(0.5, 0.5, symbol_bsi1, transform=ax1.transAxes, fontsize=24, color='lightgrey', alpha=0.3, ha='center', va='center')
price_range = global_max - global_min
margin = price_range * 0.05
ax1.set_ylim(global_min - margin, global_max + margin)
plt.tight_layout()
st.pyplot(fig1)



fig2, ax2 = plt.subplots(figsize=(10, 3), dpi=120)
ax2.plot(bvc_metrics['stamp'], bvc_metrics['bvc'], color="blue", linewidth=0.8, label="BVC")
ax2.set_xlabel("Time", fontsize=8)
ax2.set_ylabel("BVC", fontsize=8)
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax2.get_yticklabels(), fontsize=7)
ax2.legend(fontsize=7)
ax2.text(0.5, 0.5, symbol_bsi1, transform=ax2.transAxes, fontsize=24, color='lightgrey', alpha=0.3, ha='center', va='center')
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------
# SECTION 2: Hawkes Process Simulation (Alternative to tick)
# ---------------------------
st.header("Section 2: Hawkes Process Simulation (Alternative)")

import numpy as np
from numba import njit
from numba.typed import List
import matplotlib.pyplot as plt

# Optimized simulation of a univariate Hawkes process using Ogata's thinning algorithm with Numba
@njit
def simulate_hawkes(mu, alpha, beta, T):
    # Use numba's typed List for dynamic appending
    events = List()
    t = 0.0
    while t < T:
        # Compute current intensity lambda(t)
        lambda_t = mu
        for ti in events:
            if t > ti:
                lambda_t += alpha * np.exp(-beta * (t - ti))
        M = lambda_t if lambda_t > 0 else mu
        
        # Generate waiting time using exponential distribution
        u = np.random.random()
        w = -np.log(u) / M
        t_candidate = t + w
        if t_candidate > T:
            break
        
        # Compute intensity at candidate time
        lambda_candidate = mu
        for ti in events:
            if t_candidate > ti:
                lambda_candidate += alpha * np.exp(-beta * (t_candidate - ti))
                
        d = np.random.random()
        if d <= lambda_candidate / M:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)

# Parameters for the Hawkes process simulation
mu_sim = 0.1
alpha_sim = 0.5
beta_sim = 0.2  # slower decay leads to more events
T_sim = 1000  # simulation end time

# Simulate the Hawkes process using the optimized function
simulated_events = simulate_hawkes(mu_sim, alpha_sim, beta_sim, T_sim)

# Compute intensity over a time grid for plotting
time_grid = np.linspace(0, T_sim, 1000)
intensity = np.empty_like(time_grid)
for idx, t in enumerate(time_grid):
    lam = mu_sim
    for ti in simulated_events:
        if t > ti:
            lam += alpha_sim * np.exp(-beta_sim * (t - ti))
    intensity[idx] = lam

# Plot the simulated intensity and events
fig_opt, ax_opt = plt.subplots(figsize=(10, 6))
ax_opt.plot(time_grid, intensity, label="Intensity", color="green")
ax_opt.vlines(simulated_events, ymin=0, ymax=intensity.max()*0.8, color="red", alpha=0.5, label="Events")
ax_opt.set_xlabel("Time")
ax_opt.set_ylabel("Intensity")
ax_opt.legend()
plt.tight_layout()
plt.show()
import itertools
import numpy as np
# ---------------------------
# SECTION 3: Parameter Tuning for Enhanced Predictive Power
# ---------------------------
st.header("Section 3: Parameter Tuning for Enhanced Predictive Power")

# Ensure prices_bsi has a 'stamp' column (reset index if needed)
if 'stamp' not in prices_bsi.columns:
    prices_bsi.reset_index(inplace=True)

# Create a target column for evaluation.
# In this example, we use the next-period change in ScaledPrice as a proxy for future return.
# You can adjust this to your preferred target metric.
prices_bsi['target'] = prices_bsi['ScaledPrice'].diff().shift(-1).fillna(0)

def evaluate_model(bvc_metrics, actual_data):
    """
    Evaluation function: computes the mean squared error (MSE) between the predicted BVC 
    and the target (here, the next-period ScaledPrice change).
    """
    # Merge actual data with BVC metrics based on the timestamp
    merged = actual_data.merge(bvc_metrics, on='stamp', how='inner')
    mse = np.mean((merged['target'] - merged['bvc'])**2)
    return mse

# Define a grid of candidate kappa values (decay factor)
kappa_vals = [0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.50, 0.55]

best_score = float('inf')
best_params = {}

# Loop through each candidate kappa value and evaluate model performance
for kappa in kappa_vals:
    # Instantiate HawkesBVC with the candidate kappa
    hawkes_bvc = HawkesBVC(window=20, kappa=kappa)
    bvc_metrics = hawkes_bvc.eval(prices_bsi)
    
    # Evaluate performance using the defined metric (MSE in this case)
    score = evaluate_model(bvc_metrics, prices_bsi)
    st.write(f"Tested kappa: {kappa}, Score (MSE): {score}")
    
    # Keep track of the best-performing parameter
    if score < best_score:
        best_score = score
        best_params = {'kappa': kappa}

st.write("### Best Parameters Found:")
st.write(best_params)
st.write("Best Score (MSE):", best_score)

