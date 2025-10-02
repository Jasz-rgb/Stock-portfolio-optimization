# 📊 Stock Portfolio Optimization

This project implements a quantitative portfolio optimization and backtesting strategy using:  
> Momentum + Volatility for stock selection  
> Risk Parity Optimization for allocation  
> Monthly rebalancing with realistic transaction costs  

It dynamically selects stocks from a universe of blue-chip & tech stocks, allocates weights through risk parity, and evaluates performance using backtesting.

---

## 🚀 Features
> Data fetching → Alpha Vantage API (with yfinance fallback)  
> Stock selection →  
> • Momentum (6-month returns)  
> • Volatility (6-month rolling standard deviation)  
> • Score = normalized momentum – normalized volatility  
> Top 10 stocks selected each rebalance  
> Risk Parity Optimization (equal risk contribution across assets)  
> Monthly rebalancing with transaction cost of 0.1% per trade  
> Performance metrics → Sharpe Ratio, Max Drawdown, Total Return  
> Weight plots → Stacked area chart of portfolio allocation over time  

---

## 📂 Project Structure
stock-portfolio-optimization/
│── main.py # Main script (data fetching, stock selection, optimization, backtesting)
│── requirements.txt # Python dependencies
│── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Setup

> Clone the repo
```bash
git clone https://github.com/your-username/stock-portfolio-optimization.git
cd stock-portfolio-optimization
Install dependencies

bash
Copy code
pip install -r requirements.txt
Set your Alpha Vantage API Key
Open quant/main.py and replace:

python
Copy code
API_KEY = "your_api_key_here"
with your personal API key from Alpha Vantage.

🧮 Usage
Run the main script:

bash
Copy code
python main.py
This will:

Select top 10 stocks based on momentum–volatility score
Perform risk parity optimization for allocations
Backtest with monthly rebalancing
Plot allocation weights over time
Print key metrics (Sharpe Ratio, Max Drawdown, Total Return)

📊 Example Output
Performance Metrics (sample run):

Sharpe Ratio: 8.78
Max Drawdown: -1.03%
Total Return: 7.11%

📈 The script also generates a stacked area chart of portfolio weights over time:

<img width="820" height="454" alt="image" src="https://github.com/user-attachments/assets/5b3c3475-9dd7-4172-adac-0575574dba75" />
🔮 Improvements (Future Work)
Add more factors → Liquidity, Fundamentals (P/E, ROE), Macro indicators
Try alternative optimization methods → Minimum Variance, Hierarchical Risk Parity (HRP)
Extend backtesting window for longer-term analysis

🛠️ Tech Stack
Python → pandas, numpy, scipy, matplotlib, requests
Finance APIs → Alpha Vantage, yfinance
Optimization → Risk Parity (via scipy.optimize.minimize)

📜 License
MIT License – free to use and modify.

✨ Built for Quant Finance, Portfolio Optimization & Systematic Trading Research. ✨
