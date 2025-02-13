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
    exchange = ccxt.kraken({
        'enableRateLimit': True,  # Respect Kraken's rate limits
    })
    since = exchange.milliseconds() - lookback_minutes * 60 * 1000
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {e}")
    
    if not ohlcv:
        raise ValueError(f"No data returned for {symbol}.")
    
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    print("Fetched data columns:", df.columns)  # Debugging
    print("First rows of fetched data:", df.head())  # Debugging
    
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# Define the HawkesBVC class (custom estimation without tick)
class HawkesBVC:
    def __init__(self, window: int, kappa: float = None, dof=0.25, decays=None):
        """
        :param window: Lookback window for volatility calculation.
        :param kappa: Decay factor (if None, will be learned from data).
        :param dof: Degrees-of-freedom for Student-t distribution (default 0.25).
        :param decays: List of decay rates for fallback estimation (if None, defaults to [0.1, 0.5, 1.0]).
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
        
        # Compute labels using Student's t CDF
        labels = np.array([self._label(r.iloc[i], sigma.iloc[i]) for i in range(len(r))])
        
        # Learn kappa if not provided
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
        """
        Estimate kappa using a simple heuristic:
        1. Select timestamps where the absolute label > 0.5.
        2. Convert these timestamps to seconds.
        3. Compute the average time difference between successive events.
        4. Set kappa as the reciprocal of the average time difference.
        
        If there are too few events, fallback to the average of the provided decay rates.
        """
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

# Exponential Moving Average function
def ema(data, window):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

# Streamlit layout for Section 1
st.header("Section 1: Real Data Analysis")

# Input widget: symbol (use "XBT/USD" for Kraken)
symbol_bsi1 = st.sidebar.text_input(
    "Enter Ticker Symbol for (Section 2)",
    value="BTC/USD",
    key="symbol_bsi1"
)

# Input widget: lookback period
lookback_label_bsi1 = st.sidebar.selectbox(
    "Select Lookback Period (Section 2)",
    list(lookback_options.keys()),
    key="lookback_label_bsi1"
)
limit_bsi1 = lookback_options[lookback_label_bsi1]

st.write(f"Fetching data for (Section 2): **{symbol_bsi1}** with a lookback of **{limit_bsi1}** minutes.")

# Fetch OHLCV data using ccxt (real, recent data)
try:
    prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe="1m", lookback_minutes=limit_bsi1)
except Exception as e:
    st.error(f"Error fetching data (Section 2): {e}")
    st.stop()

# Ensure required columns and set index
prices_bsi.dropna(subset=['close','volume'], inplace=True)
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])
prices_bsi.set_index('stamp', inplace=True)

# Data Transformations:
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

df_skew['TrueRange'] = (
    np.abs(df_skew['hlc3'] - df_skew['hlc3'].shift(1, fill_value=df_skew['hlc3'].iloc[0]))
    / df_skew['hlc3'].shift(1, fill_value=df_skew['hlc3'].iloc[0])
)

dev_max_series = []
dev_min_series = []
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

# Compute Hawkes BVC metrics using our custom class and merge into df_skew
hawkes_bvc = HawkesBVC(window=20, kappa=0.1)
bvc_metrics = hawkes_bvc.eval(prices_bsi.reset_index())
df_skew = df_skew.merge(bvc_metrics, on='stamp', how='left')

global_min = df_skew['ScaledPrice'].min()
global_max = df_skew['ScaledPrice'].max()

# Plot skewness with EMA overlay and BVC-based segment coloring
fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=120)
norm_bvc = plt.Normalize(df_skew['bvc'].min(), df_skew['bvc'].max())

for i in range(len(df_skew['stamp']) - 1):
    xvals = df_skew['stamp'].iloc[i:i+2]
    yvals = df_skew['ScaledPrice'].iloc[i:i+2]
    bvc_val = df_skew['bvc'].iloc[i]
    
    cmap_bvc = plt.cm.Blues if bvc_val >= 0 else plt.cm.Reds
    color = cmap_bvc(norm_bvc(bvc_val))
    ax1.plot(xvals, yvals, color=color, linewidth=1)

ax1.plot(df_skew['stamp'], df_skew['ScaledPrice_EMA'], color='gray', linewidth=0.7, label=f"EMA({ema_window})")
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

# Plot the Hawkes BVC metric separately
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
# SECTION 2: Hawkes Process Simulation using tick
# ---------------------------
st.header("Section 2: Hawkes Process Simulation")

# Import tick simulation and plotting functions
from tick.hawkes import SimuHawkesSumExpKernels, HawkesSumExpKern
from tick.plot import plot_point_process

# Simulation parameters
end_time = 1000  # simulation end time
decays = [0.1, 0.5, 1.0]
baseline = [0.12, 0.07]
# For a 2-dimensional process; adjust adjacency as needed
adjacency = [[[0, 0.1, 0.4], [0.2, 0, 0.2]], [[0, 0, 0], [0.6, 0.3, 0]]]

# Simulate the Hawkes process
hawkes_exp_kernels = SimuHawkesSumExpKernels(
    adjacency=adjacency, decays=decays, baseline=baseline, end_time=end_time,
    verbose=False, seed=1039
)
hawkes_exp_kernels.track_intensity(0.1)
hawkes_exp_kernels.simulate()

# Fit a Hawkes process model using tick's learner
learner = HawkesSumExpKern(decays, penalty='elasticnet', elastic_net_ratio=0.8)
learner.fit(hawkes_exp_kernels.timestamps)

# Define time window for plotting
t_min = 100
t_max = 200

# Create a figure with two subplots
fig_sim, ax_list = plt.subplots(2, 1, figsize=(10, 6))
learner.plot_estimated_intensity(hawkes_exp_kernels.timestamps, t_min=t_min, t_max=t_max, ax=ax_list)
plot_point_process(hawkes_exp_kernels, plot_intensity=True, t_min=t_min, t_max=t_max, ax=ax_list)

# Customize the plots
for ax in ax_list:
    if len(ax.lines) >= 2:
        ax.lines[0].set_label('Estimated intensity')
        ax.lines[1].set_label('Original intensity')
        ax.lines[1].set_linestyle('--')
        ax.lines[1].set_alpha(0.8)
    if len(ax.collections) >= 2:
        ax.collections[1].set_alpha(0)
    ax.legend()

fig_sim.tight_layout()
st.pyplot(fig_sim)
