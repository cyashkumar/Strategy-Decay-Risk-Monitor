# Strategy Decay Risk Monitor
This project is a Streamlit-based dashboard that monitors the health of a trading strategy. It analyzes performance using metrics like Sharpe ratio, win rate, and decay probability. The system helps identify increasing risk and possible strategy degradation using clear visualizations.

## Features
- Rolling Sharpe Ratio monitoring
- Win rate calculation
- Strategy decay probability estimation
- Equity curve visualization
- Risk status alerts

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- yfinance
- scikit-learn

You can view the live app here: [Open App](https://share.streamlit.io/cyashkumar/Strategy-Decay-Risk-Monitor/main/dashboard.py)

## How to Run
```bash
pip install -r requirements.txt
streamlit run dashboard.py

