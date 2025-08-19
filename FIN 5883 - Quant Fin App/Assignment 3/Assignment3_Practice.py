
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:26:25 2024

@author: anind
"""

###Simple moving average
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy.stats import ttest_1samp
from time import time
from datetime import datetime
import matplotlib.dates as mdates



def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_sma(prices, window):
    return prices.rolling(window=window).mean()

def generate_signals(prices, short_window, long_window):
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0
    
    # Short and long simple moving averages
    signals['short_moving_avg'] = calculate_sma(prices, window=short_window)
    signals['long_moving_avg'] = calculate_sma(prices, window=long_window)
    
    signals['signal'][short_window:] = np.where(signals['short_moving_avg'][short_window:] 
                                                > signals['long_moving_avg'][short_window:], 1.0, 0.0)
    
    signals['positions'] = signals['signal'].diff()
    return signals

def backtest_strategy(prices, signals):
    initial_capital = float(10000.0)
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['holdings'] = signals['signal'] * prices * 100 # Assuming 100 shares
    portfolio['cash'] = initial_capital - (signals['positions'] * prices * 100).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    
    plt.figure(figsize=(10, 5))
    plt.title('Strategy Backtesting')
    plt.plot(portfolio['total'], label='Portfolio Value')
    plt.plot(prices, label='Asset Price', alpha=0.5)
    plt.legend()
    plt.show()

# Parameters
ticker = 'TSLA'
start_date = '2020-01-01'
end_date = datetime.now()
short_window = 40
long_window = 100

prices = fetch_data(ticker, start_date, end_date)

signals = generate_signals(prices, short_window, long_window)


def compute_drawdowns(portfolio_value):
    cumulative_max = portfolio_value.cummax()
    drawdown = (portfolio_value - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown

def compute_monthly_returns(portfolio):
    monthly_returns = portfolio['total'].resample('M').ffill().pct_change()
    return monthly_returns

def compute_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe_ratio

def compute_portfolio_turnover(signals):
    turnover = signals['positions'].abs().sum() / len(signals)
    return turnover

def backtest_strategy_extended(prices, signals):
    start_time = time()
    
    initial_capital = float(10000.0)
    positions = signals['signal'] * 100  # Assuming 100 shares
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['positions'] = positions * prices
    portfolio['cash'] = initial_capital - (positions.diff() * prices).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['positions']
    portfolio['returns'] = portfolio['total'].pct_change()

    avg_time_per_decision = (time() - start_time) / len(signals)
    max_drawdown = compute_drawdowns(portfolio['total'])
    monthly_returns = compute_monthly_returns(portfolio)
    sharpe_ratio = compute_sharpe_ratio(portfolio['returns'])
    turnover = compute_portfolio_turnover(signals)
    
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, prices, label='Asset Price', color='blue', alpha=0.5)
    plt.plot(prices.index, signals['short_moving_avg'], label='Short SMA', color='green', alpha=0.75)
    plt.plot(prices.index, signals['long_moving_avg'], label='Long SMA', color='orange', alpha=0.75)
    plt.scatter(signals.loc[signals.positions == 1].index, prices[signals.positions == 1], label='Buy', marker='^', color='green')
    plt.scatter(signals.loc[signals.positions == -1].index, prices[signals.positions == -1], label='Sell', marker='v', color='red')
    plt.title('Asset Price, SMA and Buy/Sell Signals')
    plt.legend()
    plt.show()

    print(f"Average time per decision: {avg_time_per_decision} seconds")
    print(f"Cumulative returns: {(portfolio['total'][-1] / initial_capital) - 1}")
    print("Monthly returns:")
    print(monthly_returns)
    print(f"Max drawdown: {max_drawdown}")
    print(f"Average portfolio turnover: {turnover}")
    print(f"Strategy's Sharpe ratio: {sharpe_ratio}")
    
    t_stat, p_value = ttest_1samp(portfolio['returns'].dropna(), 0)
    print(f"Returns significantly different from 0: {'Yes' if p_value < 0.05 else 'No'} (p-value: {p_value})")
    
    buy_and_hold_returns = (prices[-1] / prices[0]) - 1
    strategy_returns = (portfolio['total'][-1] / initial_capital) - 1

    print(f"Buy-and-Hold returns: {buy_and_hold_returns}")
    print(f"Strategy returns: {strategy_returns}")
    print("Outperform Buy-and-Hold in frictionless market:", "Yes" if strategy_returns > buy_and_hold_returns else "No")
    
    per_trade_expense = 5  # Example: $5 per trade
    annual_management_fee_percentage = 0.02  # Example: 2% annual management fee
    number_of_trades = signals['positions'].abs().sum()
    trading_expenses_total = number_of_trades * per_trade_expense
    management_fees_total = annual_management_fee_percentage * initial_capital
    net_strategy_returns_after_expenses = strategy_returns - (trading_expenses_total + management_fees_total) / initial_capital
    
    print(f"Strategy returns after expenses: {net_strategy_returns_after_expenses}")
    print("Outperform Buy-and-Hold with expenses:", "Yes" if net_strategy_returns_after_expenses > buy_and_hold_returns else "No")

backtest_strategy_extended(prices, signals)

##RSI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import ttest_1samp
from time import time
from datetime import datetime
import scipy

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_rsi_signals(prices, rsi_window=14, buy_threshold=30, sell_threshold=70):
    rsi = calculate_rsi(prices, window=rsi_window)
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0
    signals['rsi'] = rsi
    
    # Buy signal
    signals.loc[rsi < buy_threshold, 'signal'] = 1.0
    
    # Sell signal
    signals.loc[rsi > sell_threshold, 'signal'] = -1.0
    
    signals['positions'] = signals['signal'].diff()
    return signals


def compute_drawdowns(portfolio_value):
    cumulative_max = portfolio_value.cummax()
    drawdown = (portfolio_value - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown

def compute_monthly_returns(portfolio):
    monthly_returns = portfolio['total'].resample('M').ffill().pct_change()
    return monthly_returns

def compute_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe_ratio

def compute_portfolio_turnover(signals):
    turnover = signals['positions'].abs().sum() / len(signals)
    return turnover



def backtest_strategy_extended(prices, signals):
    start_time = time()
    
    initial_capital = float(10000.0)
    positions = signals['signal'] * 100  # Assuming 100 shares
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['positions'] = positions * prices
    portfolio['cash'] = initial_capital - (positions.diff() * prices).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['positions']
    portfolio['returns'] = portfolio['total'].pct_change()

    avg_time_per_decision = (time() - start_time) / len(signals)
    cumulative_returns = (portfolio['total'][-1] / initial_capital) - 1
    monthly_returns = compute_monthly_returns(portfolio)
    max_drawdown = compute_drawdowns(portfolio['total'])
    sharpe_ratio = compute_sharpe_ratio(portfolio['returns'])
    turnover = compute_portfolio_turnover(signals)
    buy_and_hold_returns = (prices[-1] / prices[0]) - 1
    strategy_returns = cumulative_returns

    per_trade_expense = 5  # Example: $5 per trade
    annual_management_fee_percentage = 0.02  # Example: 2% annual management fee
    number_of_trades = signals['positions'].abs().sum()
    trading_expenses_total = number_of_trades * per_trade_expense
    management_fees_total = annual_management_fee_percentage * initial_capital
    net_strategy_returns_after_expenses = strategy_returns - (trading_expenses_total + management_fees_total) / initial_capital

    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, prices, label='Asset Price', color='blue', alpha=0.5)
    plt.scatter(signals.loc[signals.positions == 1].index, prices[signals.positions == 1], label='Buy', marker='^', color='green')
    plt.scatter(signals.loc[signals.positions == -1].index, prices[signals.positions == -1], label='Sell', marker='v', color='red')
    plt.title('Asset Price and Buy/Sell Signals based on RSI')
    plt.legend()
    plt.show()

    print(f"Average time per decision: {avg_time_per_decision:.6f} seconds")
    print(f"Cumulative returns: {cumulative_returns * 100:.2f}%")
    print("Monthly returns:")
    print(monthly_returns)
    print(f"Max drawdown: {max_drawdown * 100:.2f}%")
    print(f"Average portfolio turnover: {turnover:.2f}")
    print(f"Strategy's Sharpe ratio: {sharpe_ratio:.2f}")
    
    t_stat, p_value = ttest_1samp(portfolio['returns'].dropna(), 0)
    print(f"Returns significantly different from 0: {'Yes' if p_value < 0.05 else 'No'} (p-value: {p_value:.6f})")
    
    print(f"Buy-and-Hold returns: {buy_and_hold_returns * 100:.2f}%")
    print(f"Strategy returns: {strategy_returns * 100:.2f}%")
    print("Outperform Buy-and-Hold in frictionless market:", "Yes" if strategy_returns > buy_and_hold_returns else "No")
    
    print(f"Strategy returns after expenses: {net_strategy_returns_after_expenses * 100:.2f}%")
    print("Outperform Buy-and-Hold with expenses:", "Yes" if net_strategy_returns_after_expenses > buy_and_hold_returns else "No")


ticker = 'NVDA'
start_date = '2020-01-01'
end_date = datetime.now()
prices = fetch_data('NVDA', '2020-01-01', end_date)
signals = generate_rsi_signals(prices, 14, 30, 70)

backtest_strategy_extended(prices, signals)


#Bollinger Band strategy

start_time = datetime.now()

# Stock return dataframe
returns_df = yf.Ticker('NVDA').history(period='5y', interval='1d').Close.to_frame()
returns_df['return'] = returns_df['Close'].pct_change().dropna()

rolling_mean = returns_df['Close'].rolling(window=25).mean().dropna()
rolling_std = returns_df['Close'].rolling(window=25).std()

returns_df['Upper_Band'] = rolling_mean + 2 * rolling_std
returns_df['Lower_Band'] = rolling_mean - 2 * rolling_std
returns_df = returns_df.dropna()


# Calculate buy and sell signals
returns_df['diff1'] = returns_df['Upper_Band'] - returns_df['Close']
returns_df['diff2'] = returns_df['Lower_Band'] - returns_df['Close']
buy_signals = returns_df.loc[(returns_df['diff2'] > 0) & (returns_df['diff2'].shift(-1) < 0)]
sell_signals = returns_df.loc[(returns_df['diff1'] < 0) & (returns_df['diff1'].shift(-1) > 0)]


# Plot return graph and Plot buy and sell signals
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xticklabels(returns_df.index, rotation=50)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.plot(returns_df.index, returns_df['Close'], color='black', label='Price')
ax.plot(returns_df.index, rolling_mean, color='blue', label='SMA')
ax.plot(returns_df.index, returns_df['Upper_Band'], color='green', label='Upper Band')
ax.plot(returns_df.index, returns_df['Lower_Band'], color='orange', label='Lower Band')
ax.set_ylabel('Price')
ax.scatter(buy_signals.index, buy_signals['Lower_Band'], marker='^', color='green', label='Buy Signal')
ax.scatter(sell_signals.index, sell_signals['Upper_Band'], marker='v', color='red', label='Sell Signal')
plt.title('NVDA (Buy/Sell Signal by Bollinger Bands)')
ax.legend(loc='upper left', borderpad=0.6, labelspacing=0.6)
plt.show()

# Backtest strategy 
Investment= 1000000
trading_cost = 0.01
gain = 0
gain_plus = 0
portfolio = 0
proceed = 0
cost = 0
cost_plus = 0
condition = 'Buy'
signal = []
cf = []
portfolio_shares = []
profit = []
profit_friction = []
portfolio_value = []

close_prices = returns_df['Close'].values
diff1_values = returns_df['diff1'].values
diff2_values = returns_df['diff2'].values

for i in range(len(returns_df)):
    if diff2_values[i] > 0 and condition == 'Buy':
        close_price = round(close_prices[i], 4)
        buy_volume = int(Investment / close_price)
        portfolio += buy_volume
        proceed = (buy_volume * close_price * (-1))
        cost_plus = proceed * (1 + trading_cost)
        cost += proceed
        portfolio_value.append(portfolio * close_price)
        cf.append(proceed)
        profit.append(gain)
        portfolio_shares.append(portfolio)
        profit_friction.append(gain_plus)
        condition = 'Sell'
        signal.append('Buy')
    elif diff1_values[i] < 0 and condition == 'Sell':
        sell_volume = portfolio
        close_price = round(close_prices[i], 4)
        proceed = (sell_volume * close_price)
        proceed_plus = proceed * (1 - trading_cost)
        gain += proceed + cost
        gain_plus += proceed_plus + cost_plus
        cost = 0
        cost_plus = 0
        portfolio_value.append(proceed)
        cf.append(proceed)
        profit.append(gain)
        profit_friction.append(gain_plus)
        portfolio = portfolio - sell_volume
        portfolio_shares.append(portfolio)
        condition = 'Buy'
        signal.append('Sell')
    else:
        gain += 0
        gain_plus += 0
        proceed = 0
        portfolio += 0
        if i > 0:
            port_val = portfolio_value[i - 1] + portfolio * (close_prices[i] - close_prices[i - 1])
            portfolio_value.append(port_val)
        else:
            portfolio_value.append(0)
        signal.append('---')
        cf.append(proceed)
        portfolio_shares.append(portfolio)
        profit.append(gain)
        profit_friction.append(gain_plus)

returns_df['signal'] = signal
returns_df['portfolio_shares'] = portfolio_shares
returns_df['cf'] = cf
returns_df['profit'] = profit
returns_df['profit_friction'] = profit_friction
returns_df['portfolio_value'] = portfolio_value



profits = returns_df['profit']
profits_friction = returns_df['profit_friction']

cumulative_profit = profits[-1]
cumulative_return = cumulative_profit / Investment
returns_df['cumulative_return'] = profits / Investment
returns_df['cumulative_return_friction'] = profits_friction / Investment

monthly_returns = returns_df['cumulative_return'].resample('1m').last().to_frame()
monthly_returns['cumulative_return_friction'] = returns_df['cumulative_return_friction'].resample('1m').last()
monthly_returns['monthly_return'] = monthly_returns.cumulative_return.diff()
monthly_returns['monthly_return'].iloc[0] = monthly_returns['cumulative_return'].iloc[0]
monthly_returns['monthly_return_friction'] = monthly_returns.cumulative_return_friction.diff()
monthly_returns['monthly_return_friction'].iloc[0] = monthly_returns['cumulative_return_friction'].iloc[0]

monthly_return_mean = monthly_returns.monthly_return.mean()
monthly_return_std = monthly_returns.monthly_return.std()
annualized_return = monthly_return_mean * 12
annualized_std = monthly_return_std * np.sqrt(12)
sharpe_ratio = annualized_return / annualized_std

t_value = (monthly_return_mean - 0) / monthly_return_std
t_critical = scipy.stats.t.ppf(q=1 - 0.10 / 2, df=len(monthly_returns) - 2)
if t_value < t_critical:
    print('Fail to Reject Null Hypothesis')
else:
    print('Reject Null Hypothesis')

annualized_return_friction = monthly_returns['monthly_return_friction'].mean() * 12
annualized_std_friction = monthly_returns['monthly_return_friction'].std() * np.sqrt(12)
sharpe_ratio_friction = annualized_return_friction / annualized_std_friction

# Plot returns
plt.plot(monthly_returns.cumulative_return, color='blue', label='Cumulative Return')
plt.plot(monthly_returns.monthly_return, color='black', label='Monthly Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('ROI')
plt.legend()
plt.show()

cumulative_returns = monthly_returns['cumulative_return']
max_drawdown_end = np.argmax(np.maximum.accumulate(cumulative_returns) - cumulative_returns) # end of the period
max_drawdown_start = np.argmax(cumulative_returns[:max_drawdown_end])
max_drawdown_return = cumulative_returns[max_drawdown_start] - cumulative_returns[max_drawdown_end]
print('Maximum Drawdown Return: ' + str(round(max_drawdown_return, 4)))

print('Annualized Return (no friction): {:.4%}'.format(annualized_return))
print('Annualized Standard Deviation (no friction): {:.4%}'.format(annualized_std))
print('Sharpe Ratio (no friction): {:.4}'.format(sharpe_ratio))
print('Annualized Return (with friction): {:.4%}'.format(annualized_return_friction))
print('Annualized Standard Deviation (with friction): {:.4%}'.format(annualized_std_friction))
print('Sharpe Ratio (with friction): {:.4}'.format(sharpe_ratio_friction))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))