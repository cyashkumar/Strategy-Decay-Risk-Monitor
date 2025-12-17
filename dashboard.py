import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Decay Monitor", layout="wide")

data = yf.download("^GSPC", start="2005-01-01")
data["Return"] = data["Close"].pct_change()

data["MA_50"] = data["Close"].rolling(50).mean()
data["MA_200"] = data["Close"].rolling(200).mean()

data["Signal"] = np.where(data["MA_50"] > data["MA_200"], 1, -1)
data["Strategy_Return"] = data["Signal"].shift(1) * data["Return"]
data = data.dropna()

window = 60

data["Sharpe"] = data["Strategy_Return"].rolling(window).mean() / data["Strategy_Return"].rolling(window).std()
data["WinRate"] = data["Strategy_Return"].rolling(window).apply(lambda x: (x > 0).mean())
data["Volatility"] = data["Return"].rolling(window).std()
data["Drawdown"] = data["Strategy_Return"].cumsum() - data["Strategy_Return"].cumsum().cummax()

future_window = 30
future_sharpe = data["Strategy_Return"].rolling(future_window).mean() / data["Strategy_Return"].rolling(future_window).std()
data["Decay"] = np.where(future_sharpe.shift(-future_window) < 0, 1, 0)

features = data[["Sharpe", "WinRate", "Volatility", "Drawdown"]].dropna()
labels = data.loc[features.index, "Decay"]

split = int(len(features) * 0.7)

X_train = features.iloc[:split]
X_test = features.iloc[split:]
y_train = labels.iloc[:split]

model = RandomForestClassifier(n_estimators=200, max_depth=5)
model.fit(X_train, y_train)

data.loc[X_test.index, "Decay_Probability"] = model.predict_proba(X_test)[:,1]

latest_prob = data["Decay_Probability"].dropna().iloc[-1]

st.title("Strategy Decay & Risk Monitoring System")

col1, col2, col3 = st.columns(3)

col1.metric("Current Sharpe", round(data["Sharpe"].iloc[-1], 2))
col2.metric("Win Rate", f"{round(data['WinRate'].iloc[-1]*100, 1)}%")
col3.metric("Decay Probability", f"{round(latest_prob*100, 1)}%")

if latest_prob < 0.3:
    st.success("Strategy Status: Healthy")
elif latest_prob < 0.6:
    st.warning("Strategy Status: Risk Increasing")
else:
    st.error("Strategy Status: Stop Trading Recommended")

st.subheader("Strategy Equity Curve")
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(data["Strategy_Return"].cumsum())
st.pyplot(fig1)

st.subheader("Rolling Sharpe Ratio")
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(data["Sharpe"])
st.pyplot(fig2)

st.subheader("Strategy Decay Probability")
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(data["Decay_Probability"])
st.pyplot(fig3)
