import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import time
from scipy.optimize import minimize
import matplotlib.dates as mdates

# Config variables
API_KEY = '1W8FTU6UYGR8POPY'  # My Alpha Vantage API key
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%d')  # 2 years back data
REBALANCE_FREQ = 'ME'  # Monthly rebalancing
NUM_STOCKS = 10  # selecting top 10 stocks
COLOR_PALETTE = plt.cm.tab20.colors
TRANSACTION_COST = 0.001  # 0.1% cost per trade

# Function for assessing Beta, PE Ratio, Market Cap from Alpha Vantage
def assessing_stock(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    if 'Note' in data:
        print(f"API rate limit hit for {ticker}")
        return None
    if 'Error Message' in data:
        print(f"Invalid API response for {ticker}: {data['Error Message']}")
        return None
    return {'Ticker': ticker}

# Calculating momentum = average price change over taken period of 6 months (126 trading days)
def calculating_momentum(data, period=126):
    momentum = data.diff(period).mean()
    return momentum

# Calculating volatility = std deviation of daily returns over the last 126 trading days
def calculating_volatility(price_series, window=126):
    returns = price_series.pct_change()
    volatility = returns.rolling(window=window).std().iloc[-1]
    return volatility

# Downloading historical price data from Yahoo Finance
def downloading_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    return data.dropna(axis=1)

# Selecting top stocks using scoring based on momentum and volatility
def selecting_top_stocks(api_key, tickers, start_date, end_date, num_stocks=10):
    stock_assessments = []
    for ticker in tickers:
        assessment = assessing_stock(ticker, api_key)
        if assessment:
            stock_assessments.append(assessment)
        time.sleep(1) # To avoid hitting the API rate limit

    stock_assessments = [a for a in stock_assessments if a is not None]
    if not stock_assessments:
        print("No valid stock data found")
        return []

    df = pd.DataFrame(stock_assessments)
    stock_data = downloading_stock_data(tickers, start_date, end_date)

    momentum_values = {}
    volatility_values = {}

    for ticker in tickers:
        if ticker in stock_data.columns:
            momentum_values[ticker] = calculating_momentum(stock_data[ticker])
            volatility_values[ticker] = calculating_volatility(stock_data[ticker])
        else:
            momentum_values[ticker] = 0
            volatility_values[ticker] = np.nan

    df['Momentum'] = [momentum_values[t] for t in df['Ticker']]
    df['Volatility'] = [volatility_values[t] for t in df['Ticker']]
    df = df.dropna(subset=['Momentum', 'Volatility'])

    df['Momentum_Norm'] = (df['Momentum'] - df['Momentum'].mean()) / df['Momentum'].std()
    df['Volatility_Norm'] = (df['Volatility'] - df['Volatility'].mean()) / df['Volatility'].std()

    df['Score'] = df['Momentum_Norm'] - df['Volatility_Norm']   #score = normalied momentum - normalized volatility (the higher the better)
    # Ranking stocks based on score
    df_sorted = df.sort_values(by='Score', ascending=False)
    selected_tickers = df_sorted['Ticker'].head(num_stocks).tolist()

    return selected_tickers

# Calculating portfolio variance = w * Covariance * w.T(transpose of w)
def calculating_portfolio_variance(w, V):
    w = np.matrix(w)
    return (w * V * w.T)[0, 0]

# Calculating risk contribution = (Marginal contribution * weight) / portfolio standard deviation (volatility)
def calculating_risk_contribution(w, V):
    w = np.matrix(w)
    sd = np.sqrt(calculating_portfolio_variance(w, V))  # portfolio standard deviation, sd = root(sd square), sd square = variance, sd = standard deviation
    MRC = V * w.T  # Marginal Risk Contribution
    RC = np.multiply(MRC, w.T) / sd  # Actual Risk Contribution per asset
    return RC


# Objective function: goal to match each asset’s risk contribution to its target (by minimizing squared error)
# J = summation((actual_RC - target_RC)^2) for each asset
def risk_budget_objective(x, pars):
    V, x_t = pars
    sd = np.sqrt(calculating_portfolio_variance(x, V))  # total portfolio std
    risk_target = np.asmatrix(sd * np.array(x_t))  # target risk contributions
    asset_RC = calculating_risk_contribution(x, V)  # actual risk contributions
    J = sum(np.square(asset_RC - risk_target.T))[0, 0]  # objective = squared error
    return J

# Constraint: total weight of all assets must equal 1. summation(w_i) = 1
def total_weight_constraint(x):
    return np.sum(x) - 1.0

# Performing Risk Parity Optimization using SLSQP
def performing_risk_parity_optimization(data):
    V = np.cov(data.T)  # Covariance matrix of returns i.e., sigma
    num_assets = len(data.columns)
    w0 = np.array([1.0 / num_assets] * num_assets) #initialy individual weights are equal (1/N)
    x_t = [1.0 / num_assets] * num_assets # target risk contribution (equal risk contribution for all assets)
    cons = ({'type': 'eq', 'fun': total_weight_constraint}) # constraint applied to the optimization problem
    bounds = tuple((0, 1) for _ in range(num_assets)) # weights must be between 0 and 1
    res = minimize(risk_budget_objective, w0, args=[V, x_t], method='SLSQP', constraints=cons, bounds=bounds) # objective function minimized using scipy.optimize.minimize function risk_budget_objective
    if not res.success:
        raise ValueError('Risk parity optimization failed')
    return dict(zip(data.columns, res.x))

# Backtesting: performing monthly rebalancing and storing weights
def backtesting_and_storing_weights(tickers, start_date, end_date, rebalance_frequency):
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')  #strings to datetime conversion
    rebalance_dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq=rebalance_frequency) # creates a list of rebalance detes for monthly rebalancing
    weights_over_time = pd.DataFrame(index=rebalance_dates, columns=tickers) # DataFrame to store weights on rebalance dates
    previous_weights = None

    for rebalance_date in rebalance_dates:
        current_end_date = rebalance_date.strftime('%Y-%m-%d')
        lookback_start_date = (rebalance_date - timedelta(days=252)).strftime('%Y-%m-%d') # 1 yeaar is the lookback period
        historical_data = downloading_stock_data(tickers, lookback_start_date, current_end_date)

        if len(historical_data) < 126:
            print(f"Skipping {rebalance_date}: Not enough data")
            continue

        try:
            optimal_weights = performing_risk_parity_optimization(historical_data)  # optimizaiton method for weights calculation

            if previous_weights is not None:
                for ticker in tickers:
                    w_prev = previous_weights.get(ticker, 0)
                    w_new = optimal_weights.get(ticker, 0)
                    cost = TRANSACTION_COST * abs(w_new - w_prev) # transaction cost given after rebalancing
                    optimal_weights[ticker] = max(0, w_new - cost) # deducting transaction cost from new weights
                total_weight = sum(optimal_weights.values())
                for ticker in optimal_weights:
                    optimal_weights[ticker] /= total_weight # normalizing weights to satisfy the constraints

            for ticker, weight in optimal_weights.items():
                weights_over_time.loc[rebalance_date, ticker] = weight # storing weights
            print(f"Rebalance Date: {rebalance_date.strftime('%Y-%m-%d')}") # peinting weights here
            for ticker, weight in optimal_weights.items():
                print(f"  {ticker}: {weight:.4f}")
            print("-" * 30)
            previous_weights = optimal_weights # updating previous weights for next rebalance date

        except Exception as e:
            print(f"Optimization failed for {rebalance_date}: {e}")
            continue

    weights_over_time = weights_over_time.dropna(how='all').fillna(0) #in case of no data available for a stock, weight is set to 0 and row is not used
    return weights_over_time

# Plotting portfolio weights over time (stackplot)
def plotting_weights_over_time(weights_data, title="Portfolio Weights Over Time - Risk Parity"):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    weights_data = weights_data.copy()
    weights_data.index = pd.to_datetime(weights_data.index)
    tickers = weights_data.columns.tolist()
    ax.stackplot(
        weights_data.index,
        weights_data.T.values,
        labels=tickers,
        colors=COLOR_PALETTE[:len(tickers)],
        alpha=0.95,
        linewidth=0.5
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Tickers", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()

def simulating_portfolio_returns(weights_over_time, tickers, start_date, end_date):
    price_data = downloading_stock_data(tickers, start_date, end_date)
    daily_returns = price_data.pct_change().dropna() # Daily return: r_i(t) = [P_i(t) / P_i(t-1)] - 1, P_i(t) = price of stock i at time t

    portfolio_returns = pd.Series(index=daily_returns.index)

    for date in weights_over_time.index:
        if date not in daily_returns.index:
            continue
        next_date_idx = daily_returns.index.get_loc(date) + 1
        if next_date_idx >= len(daily_returns):
            break
        next_date = daily_returns.index[next_date_idx]
        weights = weights_over_time.loc[date].fillna(0).values
        returns = daily_returns.loc[next_date].values
        portfolio_returns[next_date] = np.dot(weights, returns) # Portfolio return: r_p(t) = summation( w_i(t) * r_i(t) ) for i=1 to N), where w_i(t) = weight of stock i at time t, r_i(t) = return of stock i at time t

    portfolio_returns = portfolio_returns.dropna()
    # Cumulative return: C_t = product( 1 + r_p(t) ) for t = 1 to T, T = last date
    portfolio_cum_returns = (1 + portfolio_returns).cumprod()
    return portfolio_returns, portfolio_cum_returns

def calculating_sharpe_ratio(portfolio_returns, risk_free_rate=0.03):
    excess_returns = portfolio_returns - (risk_free_rate / 12) # Excess return: r_excess(t) = r_p(t) - r_f/12, where r_f = risk-free rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) # daily Ssharpe ratio = mean(excess return) / standard deviation of excess return(volatility)
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(12) # annual sharpe ratio = daily sharpe ratio * root(252)
    return sharpe_ratio_annualized

def calculating_max_drawdown(cumulative_returns):
    rolling_max = cumulative_returns.cummax() # rolling maximum: max(C_t) for t = 1 to T
    drawdown = (cumulative_returns - rolling_max) / rolling_max # drawdown = (C_t - max(C_t)) / max(C_t), where C_t = cumulative return at time t
    max_drawdown = drawdown.min() # maxi drawdown = min(drawdown at time t)
    return max_drawdown

# Calculating total return over the entire backtest period
def calculating_total_return(cumulative_returns):
    """# Total return = C_T - C_0 or rather (final/initial portfolio value -1), where C_T = cumulative return at last date, C_0 = cumulative return at first date considered as 1"""
    return cumulative_returns.iloc[-1] - 1

# Main driver function
def main():
    all_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'ORCL', 'IBM', 'TSLA', 'NVDA', 'INTC', 'JPM', 'V', 'PG',
                   'CSCO', 'UNH', 'HD', 'BAC', 'WMT', 'MA', 'CRM', 'ABBV', 'XOM', 'AVGO', 'CVX', 'LLY', 'GILD',
                   'NEE', 'TMO', 'VZ', 'KO', 'MRK', 'PEP', 'DHR', 'MCD', 'ACN', 'LIN', 'CMCSA', 'NKE', 'TXN', 'RTX',
                   'AMT', 'HON', 'UPS', 'LOW', 'ADP', 'CAT', 'INTU', 'MS', 'AMD']

    top_tickers = selecting_top_stocks(API_KEY, all_tickers, START_DATE, END_DATE, NUM_STOCKS)
    print("Selected top tickers:", top_tickers)

    if not top_tickers:
        print("No tickers selected. Exiting.")
        return

    data = downloading_stock_data(top_tickers, START_DATE, END_DATE)
    print("Downloaded data for:", list(data.columns))


    weights_over_time = backtesting_and_storing_weights(top_tickers, START_DATE, END_DATE, REBALANCE_FREQ)
    plotting_weights_over_time(weights_over_time)

    # Simulate returns from the weights
    portfolio_returns, cumulative_returns = simulating_portfolio_returns(
        weights_over_time, top_tickers, START_DATE, END_DATE
    )

    # performance metrics
    sharpe = calculating_sharpe_ratio(portfolio_returns)
    drawdown = calculating_max_drawdown(cumulative_returns)
    total_return = calculating_total_return(cumulative_returns)

    # Printing rewsults
    print(f"\nPerformance Metrics:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {drawdown:.2%}")
    print(f"Total Return: {total_return:.2%}")

if __name__ == "__main__":
    main()