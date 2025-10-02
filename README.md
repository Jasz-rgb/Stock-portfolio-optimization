📊 Stock Portfolio Optimization

This project implements a quantitative portfolio optimization and backtesting strategy using:
  1️⃣ Momentum + Volatility for stock selection
  2️⃣ Risk Parity Optimization for allocation
  3️⃣ Monthly rebalancing with realistic transaction costs
  It dynamically selects stocks from a universe of blue-chip & tech stocks, allocates weights through risk parity, and evaluates performance using backtesting.

🚀 Features
  1️⃣ Data fetching → Alpha Vantage API (with yfinance fallback)
  2️⃣ Stock selection →
        Momentum (6-month returns)
        Volatility (6-month rolling standard deviation)
      Score = normalized momentum – normalized volatility
  3️⃣ Top 10 stocks selected each rebalance
  4️⃣ Risk Parity Optimization (equal risk contribution across assets)
  5️⃣ Monthly rebalancing with transaction cost of 0.1% per trade
  6️⃣ Performance metrics → Sharpe Ratio, Max Drawdown, Total Return
  7️⃣ Weight plots → Stacked area chart of portfolio allocation over time

📂 Project Structure
  stock-portfolio-optimization/
  │── main.py          # Main script (data fetching, stock selection, optimization, backtesting)
  │── requirements.txt     # Python dependencies
  │── README.md            # Project documentation

⚙️ Installation & Setup
  1️⃣ Clone the repo
    git clone https://github.com/your-username/stock-portfolio-optimization.git
    cd stock-portfolio-optimization
  
  2️⃣ Install dependencies
    pip install -r requirements.txt
  
  3️⃣ Set your Alpha Vantage API Key
    Open quant/main.py and replace:
      API_KEY = "your_api_key_here"
    with your personal API key from [Alpha Vantage.]([url](https://www.alphavantage.co/support/#api-key))

🧮 Usage
  Run the main script:
    python main.py

  This will:
    1️⃣ Select top 10 stocks based on momentum–volatility score
    2️⃣ Perform risk parity optimization for allocations
    3️⃣ Backtest with monthly rebalancing
    4️⃣ Plot allocation weights over time
    5️⃣ Print key metrics (Sharpe Ratio, Max Drawdown, Total Return)

📊 Example Output
  Performance Metrics (sample run):
    Sharpe Ratio: 8.78
    Max Drawdown: -1.03%
    Total Return: 7.11%

  📈 The script also generates a stacked area chart of portfolio weights over time:
  <img width="820" height="454" alt="image" src="https://github.com/user-attachments/assets/5b3c3475-9dd7-4172-adac-0575574dba75" />

🔮 Improvements (Future Work)
  1️⃣ Add more factors → Liquidity, Fundamentals (P/E, ROE), Macro indicators
  2️⃣ Try alternative optimization methods → Minimum Variance, Hierarchical Risk Parity (HRP)
  3️⃣ Extend backtesting window for longer-term analysis

🛠️ Tech Stack
  1️⃣ Python → pandas, numpy, scipy, matplotlib, requests
  2️⃣ Finance APIs → Alpha Vantage, yfinance
  3️⃣ Optimization → Risk Parity (via scipy.optimize.minimize)

📜 License
  MIT License – free to use and modify.

✨ Built for Quant Finance, Portfolio Optimization & Systematic Trading Research. ✨
