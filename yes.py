import pandas as pd
import numpy as np
from scipy.stats import t as studentt
import matplotlib.pyplot as plt

class HawkesBVC:
    def __init__(self, window: int, kappa: float = None, dof=0.25, decays=None):
        """
        :param window: Lookback window for volatility calculation
        :param kappa: Decay factor (if None, will be learned from data)
        :param dof: Degrees-of-freedom for Student-t distribution (default 0.25)
        :param decays: List of decay rates for estimation (if None, defaults to [0.1, 0.5, 1.0])
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
        # Without tick, we use a simple fallback:
        # Return the average of the provided decay rates.
        estimated_kappa = np.mean(self.decays)
        print(f'Fallback kappa: {estimated_kappa}')
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


###############################
# SECTION 2: Momentum, Skewness & Hawkes BVC Analysis
###############################
import streamlit as st
import matplotlib.dates as mdates

# Placeholder definitions for dependencies in your code.
# Replace these with your actual implementations.
lookback_options = {'1h': 60, '6h': 360, '24h': 1440}

def fetch_data(symbol, timeframe, lookback_minutes):
    # Dummy data for example purposes.
    date_rng = pd.date_range(start='2021-01-01', periods=lookback_minutes, freq='T')
    data = pd.DataFrame(date_rng, columns=['stamp'])
    data['close'] = np.linspace(100, 200, lookback_minutes)
    data['volume'] = np.random.randint(1, 100, size=lookback_minutes)
    data['high'] = data['close'] * 1.01
    data['low'] = data['close'] * 0.99
    return data

def ema(data, window):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values

st.header("Section 1:")

symbol_bsi1 = st.sidebar.text_input(
    "Enter Ticker Symbol for (Section 2)",
    value="BTC/USD",
    key="symbol_bsi1"
)
lookback_label_bsi1 = st.sidebar.selectbox(
    "Select Lookback Period (Section 2)",
    list(lookback_options.keys()),
    key="lookback_label_bsi1"
)
limit_bsi1 = lookback_options[lookback_label_bsi1]

st.write(f"Fetching data for (Section 2): **{symbol_bsi1}** with a lookback of **{limit_bsi1}** minutes.")

try:
    prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe="1m", lookback_minutes=limit_bsi1)
except Exception as e:
    st.error(f"Error fetching data (Section 2): {e}")
    st.stop()
prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe="1m", lookback_minutes=limit_bsi1)
prices_bsi.dropna(subset=['close','volume'], inplace=True)

# Convert "stamp" to datetime
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])

# Set the index to the "stamp" column
prices_bsi.set_index('stamp', inplace=True)

# Transformations:
prices_bsi['ScaledPrice'] = np.log(prices_bsi['close'] / prices_bsi['close'].iloc[0]) * 1e4
prices_bsi['ScaledPrice_EMA'] = ema(prices_bsi['ScaledPrice'].values, window=10)
prices_bsi = prices_bsi.dropna(subset=['close', 'volume'])
prices_bsi['cum_vol'] = prices_bsi['volume'].cumsum()
prices_bsi['cum_pv'] = (prices_bsi['close'] * prices_bsi['volume']).cumsum()
prices_bsi['vwap'] = prices_bsi['cum_pv'] / prices_bsi['cum_vol']
prices_bsi['vwap_transformed'] = np.log(prices_bsi['vwap'] / prices_bsi['vwap'].iloc[0]) * 1e4

if 'buyvolume' not in prices_bsi.columns or 'sellvolume' not in prices_bsi.columns:
    prices_bsi['buyvolume'] = prices_bsi['volume'] * 0.5
    prices_bsi['sellvolume'] = prices_bsi['volume'] - prices_bsi['buyvolume']

# Skewness calculations
st.write("## Skewness")
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

# Compute EMA over ScaledPrice
ema_window = 10
df_skew['ScaledPrice_EMA'] = ema(df_skew['ScaledPrice'].values, ema_window)

# Merge Hawkes BVC metrics into df_skew
hawkes_bvc = HawkesBVC(window=20, kappa=0.1)
bvc_metrics = hawkes_bvc.eval(prices_bsi.reset_index())
df_skew = df_skew.merge(bvc_metrics, on='stamp', how='left')

global_min = df_skew['ScaledPrice'].min()
global_max = df_skew['ScaledPrice'].max()

# Plot Skewness + EMA overlay with BVC-based colors
fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

norm_bvc = plt.Normalize(df_skew['bvc'].min(), df_skew['bvc'].max())

for i in range(len(df_skew['stamp']) - 1):
    xvals = df_skew['stamp'].iloc[i:i+2]
    yvals = df_skew['ScaledPrice'].iloc[i:i+2]
    bvc_val = df_skew['bvc'].iloc[i]
    
    cmap_bvc = plt.cm.Blues if bvc_val >= 0 else plt.cm.Reds
    color = cmap_bvc(norm_bvc(bvc_val))
    
    ax.plot(xvals, yvals, color=color, linewidth=1)

ax.plot(
    df_skew['stamp'],
    df_skew['ScaledPrice_EMA'],
    color='gray',
    linewidth=0.7,
    label=f"EMA({ema_window})"
)

ax.set_xlabel("Time", fontsize=8)
ax.set_ylabel("ScaledPrice", fontsize=8)
ax.set_title(" ", fontsize=10)
ax.legend(fontsize=7)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)
ax.text(
    0.5, 0.5, symbol_bsi1, transform=ax.transAxes,  
    fontsize=24, color='lightgrey', alpha=0.3,
    ha='center', va='center'
)

price_range = global_max - global_min
margin = price_range * 0.05
ax.set_ylim(global_min - margin, global_max + margin)

plt.tight_layout()
st.pyplot(fig)

# Plot Hawkes BVC using the same prices_bsi data
fig_bvc, ax_bvc = plt.subplots(figsize=(10, 3), dpi=120)
ax_bvc.plot(
    bvc_metrics['stamp'], bvc_metrics['bvc'],
    color="blue", linewidth=0.8,
    label="BVC"
)
ax_bvc.set_xlabel("Time", fontsize=8)
ax_bvc.set_ylabel("BVC", fontsize=8)
ax_bvc.legend(fontsize=7)
ax_bvc.set_title("BVC", fontsize=10)
ax_bvc.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_bvc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax_bvc.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax_bvc.get_yticklabels(), fontsize=7)
ax_bvc.text(
    0.5, 0.5, symbol_bsi1, transform=ax_bvc.transAxes,  
    fontsize=24, color='lightgrey', alpha=0.3,
    ha='center', va='center'
)
plt.tight_layout()
st.pyplot(fig_bvc)
