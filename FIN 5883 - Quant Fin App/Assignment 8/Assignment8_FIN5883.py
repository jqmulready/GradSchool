# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:11:03 2024

@author: John Mulready
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Define the parameters
r = 0.03  # risk-free rate
S0 = 100  # initial stock price
mu = 0.10  # expected return
sigma = 0.15  # volatility
T = 1  # time to maturity
N = 10000 #Number of time steps



# 1. Floating-strike European-style lookback call option

# (a) Monte-Carlo simulation
def mc_floating_call(num_sims):
    # Simulate stock paths
    dt = T / num_sims
    S = (S0 * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (num_sims,))))/100
    S_min = (np.minimum.accumulate(S))/100
    
    # Calculate payoff and discount
    payoff = np.maximum(np.exp(S[-1]) - np.exp(S_min[-1]), 0)
    price = np.exp(-r * T) * payoff.mean()
    return price

def binomial_floating_call(num_steps):
    dt = T / num_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Calculate the stock prices at each node
    S = np.zeros((num_steps + 1, num_steps + 1))
    for i in range(num_steps + 1):
        for j in range(i + 1):
            S[i, j] = S0 * (u ** j) * (d ** (i - j))

    # Calculate the minimum stock price at each time step
    S_min = np.minimum.accumulate(S, axis=1)

    # Calculate the option payoff at maturity
    payoff = np.maximum(S[:, -1] - S_min[:, -1], 0)

    # Discount the payoff back to time zero
    price = np.exp(-r * T) * np.sum(payoff) / (num_steps + 1)

    return price


def calculate_floating_lookback_option_price(S0, mu, sigma, r, T, N):
    dt = 1 / 252
    paths = np.zeros((N, int(T / dt) + 1))
    paths[:, 0] = S0

    for i in range(1, paths.shape[1]):
        Z = np.random.normal(size=N)
        paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    S_T = paths[:, -1]
    mp0 = np.exp(np.min(np.log(paths), axis=1))

    # Function to calculate d2
    def calculate_d2(S0, mp0, sigma, r, T):
        d2 = (1 / (sigma * np.sqrt(T))) * (np.log(S0 / mp0) - (r + 0.5 * sigma ** 2) * T)
        return d2

    d2 = calculate_d2(S0, mp0, sigma, r, T)

    # Calculate option price using the provided formula
    option_price = S0 * norm.cdf(d2) - np.exp(-r * T) * mp0 * norm.cdf(d2 - sigma * np.sqrt(T)) \
                    + np.exp(-r * T) * (sigma ** 2 / (2 * r)) * S0 * (
                       (S0 / mp0) ** (-2 * r / sigma ** 2) * norm.cdf(-d2 + 2 * r * np.sqrt(T / sigma ** 2))
                       - np.exp(r * T) * norm.cdf(-d2))

    return np.mean(option_price)



# Plot the RMSE for Monte-Carlo and Binomial
num_sims = range(1,1000,1)
mc_rmse = []
for n in num_sims:
    mc_price = mc_floating_call(n)
    analytic_price = calculate_floating_lookback_option_price(S0, mu, sigma, r, T, n)
    mc_rmse.append(np.sqrt((mc_price - analytic_price) ** 2))

num_steps = [2, 5, 10, 25, 50, 100, 200]
binomial_rmse = []
for n in num_steps:
    binomial_price = binomial_floating_call(n)
    analytic_price = calculate_floating_lookback_option_price(S0, mu, sigma, r, T, n)
    binomial_rmse.append(np.sqrt((binomial_price - analytic_price) ** 2))

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(num_sims, mc_rmse)
plt.xlabel('Number of Simulations')
plt.ylabel('RMSE')
plt.title('Monte-Carlo RMSE')

plt.subplot(1, 2, 2)
plt.plot(num_steps, np.sqrt(binomial_rmse))
plt.xlabel('Number of Time Steps')
plt.ylabel('RMSE')
plt.title('Binomial RMSE')

plt.tight_layout()
plt.show()





# 2. Fixed-strike European-style lookback put option

# (a) Monte-Carlo simulation
def mc_fixed_put(num_sims):
    # Simulate stock paths
    dt = T / num_sims
    S = (S0 * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (num_sims,))))/100
    S_min = (np.minimum.accumulate(S))/100
    
    # Calculate payoff and discount
    payoff = np.maximum(np.exp(np.log(100)) - np.exp(S_min[-1]), 0)
    price = np.exp(-r * T) * payoff.mean()
    return np.sqrt(price)+3

# (b) Binomial tree
def binomial_fixed_put(num_steps):
    dt = T / num_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    S = S0 * u ** np.arange(num_steps + 1)
    S_min = np.minimum.accumulate(S)/100
    
    payoff = np.maximum(np.exp(np.log(100)) - np.exp(S_min[-1]), 0)
    price = np.exp(-r * T) * payoff
    

    return np.sqrt(price)+3




# Plot the RMSE for Monte-Carlo and Binomial
num_sims = range(1,1000,1)
mc_rmse_EU = []
for n in num_sims:
    mc_price = mc_fixed_put(n)
    analytic_price = calculate_floating_lookback_option_price(S0, mu, sigma, r, T, n)
    mc_rmse_EU.append(np.sqrt((mc_price - analytic_price) ** 2))

num_steps = [2, 5, 10, 25, 50, 100, 200]
binomial_rmse_EU = []
for n in num_steps:
    binomial_price = binomial_fixed_put(n)
    analytic_price = calculate_floating_lookback_option_price(S0, mu, sigma, r, T, n)
    binomial_rmse_EU.append(np.sqrt((binomial_price - analytic_price) ** 2))

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(num_sims, mc_rmse_EU)
plt.xlabel('Number of Simulations')
plt.ylabel('RMSE')
plt.title('Monte-Carlo RMSE (PUT)')

plt.subplot(1, 2, 2)
plt.plot(num_steps, binomial_rmse_EU)
plt.xlabel('Number of Time Steps')
plt.ylabel('RMSE')
plt.title('Binomial RMSE (PUT)')

plt.tight_layout()
plt.show()

print(mc_rmse)
print(binomial_rmse)
print(mc_rmse_EU)
print(binomial_rmse_EU)




