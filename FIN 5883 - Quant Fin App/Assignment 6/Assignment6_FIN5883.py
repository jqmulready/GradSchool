# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:54:07 2020

@author: Jake Mulready
"""
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



def get_options_data(symbol):
    ticker = yf.Ticker(symbol)
    expirations = ticker.options[-50:]

    all_options = pd.DataFrame()
    current_date = datetime.today()

    for expiration in expirations:
        option_chain = ticker.option_chain(expiration)
        if option_chain.calls.empty:
            continue
        opt = option_chain.calls
        opt['expirationDate'] = expiration
        
        # Calculate time to maturity
        exp_date = datetime.strptime(expiration, "%Y-%m-%d")
        opt['time_to_maturity'] = (exp_date - current_date).days
        
        all_options = pd.concat([all_options, opt], ignore_index=True)

    all_options[['bid', 'ask', 'strike']] = all_options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    all_options['avg_price'] = (all_options['bid'] + all_options['ask']) / 2

    # Calculate moneyness
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    all_options['moneyness'] = all_options['strike'] / current_price
    #all_options['moneynesssq'] = (all_options['moneyness'])^2
    #all_options['moneynesscube'] = (all_options['moneyness'])^3
    all_options = all_options[all_options['moneyness'] <= 1.1]

    return all_options

# Example usage:
symbol = "^SPX"  # S&P 500 index ticker
options_data = get_options_data(symbol)
options_data.head()

options_data = options_data[(options_data['time_to_maturity'] >= 0) & (options_data['time_to_maturity'] <= 365)]


# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract data
x = options_data['time_to_maturity']
y = options_data['moneyness']
z = options_data['impliedVolatility']

# Plot the surface
surface = ax.plot_trisurf(x, y, z, cmap='viridis')

# Set labels and title
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Moneyness')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface')

# Set z-axis limits
ax.set_zlim(0, 2)

# Fit the spline model
spline_model = SmoothBivariateSpline(x, y, z)

# Generate grid points
x_grid = np.linspace(min(x), max(x), 100)
y_grid = np.linspace(min(y), max(y), 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Evaluate the spline at grid points
Z = spline_model.ev(X, Y)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Moneyness')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface')

# Show the plot
plt.show()







#INITIAL PARAMETERS

import numpy as np
from datetime import datetime, timedelta

# Calculate historical volatility
def calculate_historical_volatility(volatilities, days):
    log_returns = np.log(1 + volatilities.pct_change().dropna())
    volatility = log_returns.rolling(window=days).std() * np.sqrt(252)  # Annualized volatility
    return volatility

# Assuming you have the options_data DataFrame available
options_data['expirationDate'] = pd.to_datetime(options_data['expirationDate'])
options_data = options_data.sort_values(by='expirationDate')

# Calculate historical volatilities for different time periods
short_term_volatility = calculate_historical_volatility(options_data['impliedVolatility'], 30)
long_term_volatility = calculate_historical_volatility(options_data['impliedVolatility'], 365)

# Suggest initial parameters based on historical volatilities
initial_theta = long_term_volatility.mean() ** 2  # Long-run variance
initial_v0 = short_term_volatility.iloc[-1] ** 2  # Initial variance

# You can use heuristics or rules of thumb for the other parameters
initial_kappa = 2.0  # Mean-reversion speed
initial_sigma = 0.3  # Volatility of volatility
initial_rho = -0.5   # Correlation between asset and volatility

# Print the suggested initial parameters
print("Suggested Initial Parameters for the Heston Model:")
print(f"Kappa (mean-reversion speed): {initial_kappa}")
print(f"Theta (long-run variance): {initial_theta}")
print(f"Sigma (volatility of volatility): {initial_sigma}")
print(f"Rho (correlation): {initial_rho}")
print(f"V0 (initial variance): {initial_v0}")














import numpy as np
from scipy.optimize import minimize

def heston_model(params, S, K, T, r, q):
    kappa, theta, sigma, rho, v0 = params
    
    d1 = (np.log(S / K) + (r - q + theta) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Integrate the variance process
    integration_factor = (1 - np.exp(-kappa * T)) / (kappa * T)
    variance = v0 * np.exp(-kappa * T) + theta * integration_factor
    
    # Calculate the Heston model option price
    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # Adjust for correlation
    term1 = rho * sigma * variance * np.exp(-q * T) * norm.cdf(d1)
    term2 = rho * sigma * variance * K * np.exp(-r * T) * norm.cdf(-d2)
    price += term1 - term2
    
    return price

def heston_calibration(params, data):
    errors = []
    for i in range(len(data)):
        S = data.iloc[i]['lastPrice']  # Stock Price
        K = data.iloc[i]['strike']  # Strike Price
        T = data.iloc[i]['time_to_maturity'] / 365.0  # Maturity (Years)
        r = 0.05  # Assuming a risk-free rate of 5%
        q = 0.0  # No dividend yield 
        iv = data.iloc[i]['impliedVolatility']  # Implied Volatility
        kappa, theta, sigma, rho, v0 = params
        d1 = (np.log(S / K) + (r - q + 0.5 * v0) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        model_value = heston_model(params, S, K, T, r, q)
        market_value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        error = iv - model_value / market_value
        errors.append(error)
    return np.array(errors)


initial_params = [initial_kappa, initial_theta, initial_sigma, initial_rho, initial_v0]
bounds = [(0.01, 5), (0.01, 1), (0.01, 1), (-1, 1), (0.01, 1)]
result = minimize(lambda params: np.sum(heston_calibration(params, options_data)**2), initial_params, bounds=bounds)
heston_params = result.x
print('Heston Model Parameters:')
print('Kappa:', heston_params[0])
print('Theta:', heston_params[1])
print('Sigma:', heston_params[2])
print('Rho:', heston_params[3])
print('V0:', heston_params[4])


# Calculate the risk-free rate and dividend yield (or set them to appropriate values)
r = 0.05  # Assuming a risk-free rate of 5%
q = 0.0   # Assuming no dividend yield

# Create a new column in options_data to store the Heston model prices
options_data['heston_price'] = np.nan

for idx, row in options_data.iterrows():
    S = row['lastPrice']    # Stock price
    K = row['strike']       # Strike price
    T = row['time_to_maturity'] / 365.0  # Time to maturity in years
    
    options_data.at[idx, 'heston_price'] = heston_model(heston_params, S, K, T, r, q)



import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.interpolate import SmoothBivariateSpline

def black_scholes_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return value

def find_implied_volatility(S, K, T, r, q, price):
    def objective(sigma):
        return black_scholes_call(S, K, T, r, q, sigma) - price
    implied_vol = fsolve(objective, 0.2)[0]  # Initial guess of 0.2
    return implied_vol

# Calculate the risk-free rate and dividend yield (or set them to appropriate values)
r = 0.05  # Assuming a risk-free rate of 5%
q = 0.0   # Assuming no dividend yield

# Create a new column in options_data to store the Black-Scholes implied volatilities
options_data['bs_implied_vol'] = np.nan

for idx, row in options_data.iterrows():
    S = row['lastPrice']    # Stock price
    K = row['strike']       # Strike price
    T = row['time_to_maturity'] / 365.0  # Time to maturity in years
    price = row['heston_price']
    
    implied_vol = find_implied_volatility(S, K, T, r, q, price)
    options_data.at[idx, 'bs_implied_vol'] = implied_vol

# Extract data for original surface
x_original = options_data['time_to_maturity']
y_original = options_data['moneyness']
z_original = options_data['bs_implied_vol']

# Fit a smooth spline surface to the implied volatility data
spline_model = SmoothBivariateSpline(x_original, y_original, z_original)

# Generate a grid of points for plotting
x_grid = np.linspace(min(x_original), max(x_original), 100)
y_grid = np.linspace(min(y_original), max(y_original), 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Evaluate the spline at grid points
Z_smoothed = spline_model.ev(X, Y)

# Import necessary libraries for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and a 3D axis for the original surface
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_trisurf(x_original, y_original, z_original, cmap='viridis')
ax1.set_title('Original Black-Scholes Implied Volatility Surface')
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Moneyness')
ax1.set_zlabel('Implied Volatility')

# Create a figure and a 3D axis for the smoothed surface
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
surf = ax2.plot_surface(X, Y, Z_smoothed, cmap='viridis')
ax2.set_title('Smoothed Black-Scholes Implied Volatility Surface')
ax2.set_xlabel('Time to Maturity')
ax2.set_ylabel('Moneyness')
ax2.set_zlabel('Implied Volatility')

# Show the plots
plt.show()

# Function to calculate root mean square error (RMSE)
def calculate_rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))


heston_rmse = calculate_rmse(options_data['lastPrice'], options_data['heston_price'])
bs_rmse = calculate_rmse(options_data['impliedVolatility'], options_data['bs_implied_vol'])


