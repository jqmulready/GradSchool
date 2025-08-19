'''
Jake Mulready 
04/08/2024
'''

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_vollib.black_scholes.implied_volatility import implied_volatility
from mpl_toolkits.mplot3d import Axes3D
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.interpolate import SmoothBivariateSpline
from scipy.stats import norm
from scipy.optimize import minimize
import math
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv



file_path = r"C:\Users\John Mulready\Desktop\John Mulready\Downloads\Spring 2024\FIN 5883 - Quant Fin App\WTI_data_20240326.csv"
data = pd.read_csv(file_path, sep='\t')
# Splitting the string values and creating separate columns
df = pd.DataFrame(data['Strike,Price,PctChange'].str.split(',').tolist(), columns=['Strike', 'Price', 'PctChange'])

# Converting string columns to numeric
df[['Strike', 'Price', 'PctChange']] = df[['Strike', 'Price', 'PctChange']].astype(float)

print(df)

# Given trade day and expiry day
trade_day = pd.to_datetime("2024/03/25")
expiry_day = pd.to_datetime("2024/04/30")
WTI_price = 81.2
r = 0.07  # Assume a risk-free rate of 5%
t = (expiry_day - trade_day).days / 365.0 # Calculate time to expiry in years


import numpy as np
from scipy.stats import norm

# Define Black-Scholes Asian call option pricing function

def asian_call_bs(S, K, T, r, sigma, n):

    """

    S: Current stock price

    K: Strike price

    T: Time to maturity (in years)

    r: Risk-free interest rate

    sigma: Volatility

    n: Number of averaging periods

    """

    dt = T / n

    nu = r - 0.5 * sigma**2

    si = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))

    d1 = (np.log(S / K) + (nu + 0.5 * sigma**2) * T) / (si * np.sqrt(T))

    d2 = d1 - si * np.sqrt(T)

    A = np.exp(-r * T) * (S * np.exp(nu * T) * norm.cdf(d1) - K * norm.cdf(d2))
 
    return A

def implied_volatility(option_price, S, K, T, r):

    if option_price <= 0:
        return np.nan  # Skip implied volatility calculation for zero or negative option prices

    sigma_est = 0.2  # Initial guess for implied volatility
    for _ in range(100):
        price = asian_option_price(S, K, T, r, sigma_est)
        vega = option_price * S * np.sqrt(T)
        if vega == 0:
            return np.nan  # Skip implied volatility calculation if vega is zero
        sigma_est -= (price - option_price) / vega
    return sigma_est



from py_vollib.black_scholes.implied_volatility import implied_volatility as iv

call_ivs = []

for strike, option_price in zip(df['Strike'], df['Price']):
    try:
        iv_value = iv(option_price, WTI_price, strike, t, r, 'c')
        call_ivs.append(iv_value)
    except:
        call_ivs.append(np.nan)

from scipy.interpolate import UnivariateSpline

# Filter out NaN values
valid_indices = np.isfinite(call_ivs)
strikes = df['Strike'][valid_indices].values
ivs = [iv for iv, valid in zip(call_ivs, valid_indices) if valid]

# Sort the data by strike price
sorted_indices = np.argsort(strikes)
strikes = strikes[sorted_indices]
ivs = [ivs[i] for i in sorted_indices]

# Create a spline model
spline = UnivariateSpline(strikes, ivs, k=3, s=1e-6)

# Evaluate the spline model on a finer grid of strike prices
strike_grid = np.linspace(strikes.min(), strikes.max(), 100)
iv_grid = spline(strike_grid)

# Plot the original data and the spline model
plt.figure(figsize=(10, 6))
plt.scatter(strikes, ivs, label='Original Data')
plt.plot(strike_grid, iv_grid, label='Spline Model')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Call Option Implied Volatility Curve')
plt.legend()
plt.show()



# Filter out NaN values
valid_indices = np.isfinite(call_ivs)
strikes = df['Strike'][valid_indices].values
ivs = [iv for iv, valid in zip(call_ivs, valid_indices) if valid]

# Sort the data by strike price
sorted_indices = np.argsort(strikes)
strikes = strikes[sorted_indices]
ivs = [ivs[i] for i in sorted_indices]

# Create a spline model
spline = UnivariateSpline(strikes, ivs, k=3, s=1e-6)

# a. Make a grid of moneyness (K/S(0)) ranging from 0.9 to 1.1 by 0.005
S0 = WTI_price
moneyness_grid = np.arange(0.9, 1.11, 0.005)
strike_grid = moneyness_grid * S0

# b. Fill in the grid of implied volatilities as a function of K/S(0)
iv_grid = spline(strike_grid)

# c. Calculate the Black-Scholes Asian call option prices for each implied volatility
option_prices = []
for strike, iv in zip(strike_grid, iv_grid):
    option_price = asian_call_bs(S0, strike, t, r, iv,30)
    option_prices.append(option_price)

# Plot the Black-Scholes Asian call option prices
plt.figure(figsize=(10, 6))
plt.plot(moneyness_grid, option_prices)
plt.xlabel('Moneyness (K/S(0))')
plt.ylabel('Option Price')
plt.title('Black-Scholes Asian Call Option Prices')
plt.show()





import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Load data
yf_file_path = r"C:\Users\John Mulready\Desktop\John Mulready\Downloads\Spring 2024\FIN 5883 - Quant Fin App\YF WTI Price data.csv"
yf_data = pd.read_csv(yf_file_path, sep='\t')

# Split 'Date,Close' column
yf_data[['Date', 'Close']] = yf_data['Date,Close'].str.split(',', expand=True)

# Convert 'Close' column to numeric type
yf_data['Close'] = pd.to_numeric(yf_data['Close'])

# Extract 'Close' prices
prices = yf_data['Close']

# Calculate log returns
log_returns = np.diff(np.log(prices))

# Obtain daily WTI price data for the past year
today = datetime.now().date()
start_date = today - timedelta(days=365)
end_date = today

# Number of trading days until option maturity
expiry_day = datetime(2024, 6, 1)  # Assuming June 1st, 2024 as the expiry day
trade_day = datetime(2024, 4, 1)   # Assuming April 1st, 2024 as the trade day
num_days = (expiry_day - trade_day).days

# Function to simulate a single path
def simulate_path(log_returns, num_days):
    simulated_returns = np.random.choice(log_returns, size=num_days, replace=True)
    simulated_prices = np.cumprod(np.exp(np.append([log_returns[0]], simulated_returns)))
    geometric_average = np.exp(np.log(simulated_prices).mean())
    return geometric_average

# Bootstrap simulation
num_sims = 1000
simulated_averages = np.array([simulate_path(log_returns, num_days) for _ in range(num_sims)])

# Calculate Asian call option prices for each strike price
bootstrapped_prices = []
for strike in strike_grid:
    payoffs = np.maximum((simulated_averages*S0) - strike, 0)
    option_price = np.mean(payoffs) * np.exp(-r * t)
    bootstrapped_prices.append(option_price)

# Calculate RMSE relative to Black-Scholes prices
rmse = np.sqrt(np.mean((np.array(bootstrapped_prices) - np.array(option_prices)) ** 2))

print(f"RMSE relative to Black-Scholes prices (Bootstrap): {rmse:.4f}")


# 6. Edgeworth Expansion
from scipy.stats import skew, kurtosis

# Calculate cumulant coefficients
cumulant_1 = np.mean(simulated_averages*S0)
cumulant_2 = np.var(simulated_averages*S0)
cumulant_3 = skew(simulated_averages*S0) * cumulant_2 ** (3/2)
cumulant_4 = kurtosis(simulated_averages*S0, fisher=False) * cumulant_2 ** 2

print(f"Cumulant coefficients:")
print(f"Cumulant 1: {cumulant_1:.4f}")
print(f"Cumulant 2: {cumulant_2:.4f}")
print(f"Cumulant 3: {cumulant_3:.4f}")
print(f"Cumulant 4: {cumulant_4:.4f}")



# 7. Binomial Tree Method
from scipy.stats import norm

def asian_call_binomial(S, K, T, r, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    s = [S]
    for i in range(1, n+1):
        s.append(s[-1] * (u if i % 2 == 0 else d))

    avg = np.mean(s[1:])
    call_value = 0
    for i in range(n):
        if s[i+1] >= K:
            call_value += (s[i+1] - K) * (p ** i) * ((1-p) ** (n-i))
    call_value *= np.exp(-r * T)
    return call_value

# Calculate binomial tree Asian call option prices
binomial_prices = []
for strike, iv in zip(strike_grid, iv_grid):
    option_price = asian_call_binomial(S0, strike, t, r, iv, 30)
    binomial_prices.append(option_price)

# Calculate RMSE relative to Black-Scholes prices
rmse_binomial = np.sqrt(np.mean((np.array(binomial_prices) - np.array(option_prices)) ** 2))

print(f"RMSE relative to Black-Scholes prices (Binomial Tree): {rmse_binomial:.4f}")

