import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from py_vollib.black_scholes.implied_volatility import implied_volatility

# Define the ticker symbol
ticker_symbol = '^SPX'  # Example: Apple Inc.

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

# Get the options chain
options_chain = ticker.option_chain

# Define the expiration date for the options you are interested in
expiration_date = '2024-05-24'  # Example: March 15, 2024

# Get the options data for the specified expiration date
options_data = options_chain(expiration_date)

# Filter for put options
put_options = options_data.puts

# Extract relevant data for each put option
strike_prices = put_options['strike']
option_prices = put_options['lastPrice']
interest_rate = 0.01  # Example: 1% annual interest rate, you can adjust it accordingly
days_to_expiration = (np.datetime64(expiration_date) - np.datetime64('today')).astype(int)  # Number of days to expiration

# Calculate implied volatilities
implied_vols = []
for i in range(len(put_options)):
    try:
        implied_vol = implied_volatility(option_prices[i], strike_prices[i], ticker.history().iloc[-1]['Close'], interest_rate, days_to_expiration,'c')
        implied_vols.append(implied_vol)
    except:
        implied_vols.append(np.nan)  # Handle cases where implied volatility cannot be calculated

# Plot implied volatilities
plt.figure(figsize=(10, 6))
plt.plot(strike_prices, implied_vols, marker='o', linestyle='-')
plt.title('Implied Volatility vs Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.grid(True)
plt.show()


import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility

# Function to collect options chain data for an American put option
def get_put_options(ticker_symbol, expiration_date):
    ticker = yf.Ticker(ticker_symbol)
    options_chain = ticker.option_chain(expiration_date)
    put_options = options_chain.puts
    return put_options

# Function to calculate Black-Scholes European option price
def black_scholes_european(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry):
    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    return call_price

# Function to calculate Cox-Ross-Rubinstein (CRR) American option price
def binomial_tree_american(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, steps):
    dt = time_to_expiry / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(risk_free_rate * dt) - d) / (u - d)

    prices = np.zeros((steps + 1, steps + 1))
    prices[0, 0] = spot_price

    for i in range(1, steps + 1):
        prices[i, 0] = prices[i - 1, 0] * u
        for j in range(1, i + 1):
            prices[i, j] = prices[i - 1, j - 1] * d

    option_values = np.maximum(0, prices - strike_price)

    for j in range(steps):
        option_values[steps, j] = np.maximum(option_values[steps, j], strike_price - prices[steps, j])

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-risk_free_rate * dt) * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1])

    return option_values[0, 0]

# Function to calculate Monte Carlo American option price
def monte_carlo_american(spot_price, strike_price, risk_free_rate, volatility, time_to_expiry, sims):
    np.random.seed(42)
    z = np.random.normal(size=sims)
    ST = spot_price * np.exp((risk_free_rate - 0.5 * volatility ** 2) * time_to_expiry + volatility * np.sqrt(time_to_expiry) * z)
    payoffs = np.maximum(ST - strike_price, 0)
    option_price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(payoffs)
    return option_price

# Function to calculate implied volatilities from options data
def calculate_implied_volatility(options_data):
    return options_data['impliedVolatility']

# Function to calculate root mean squared error (RMSE)
def calculate_rmse(actual_values, predicted_values):
    rmse_dict = {}
    for column in predicted_values.columns:
        error = actual_values - predicted_values[column]
        squared_error = error ** 2
        mean_squared_error = squared_error.mean()
        rmse = np.sqrt(mean_squared_error)
        rmse_dict[column] = rmse
    rmse_df = pd.DataFrame(rmse_dict, index=['RMSE'])
    return rmse_df

# Main function
def main(spot_price, ticker_symbol, expiration_date):
    # Define parameters
    strike_price = 180
    risk_free_rate = 0.03
    time_to_expiry = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365

    # Get put options data
    put_options = get_put_options(ticker_symbol, expiration_date)

    # Extract strike prices and implied volatilities
    strike_prices = put_options['strike']
    implied_vols = calculate_implied_volatility(put_options)

    # Price options using different methods
    bs_european_prices = []
    crr_american_prices = []
    mc_american_prices = []
    for implied_volatility in implied_vols:
        bs_european_price = black_scholes_european(spot_price, strike_price, risk_free_rate, implied_volatility, time_to_expiry)
        crr_american_price = binomial_tree_american(spot_price, strike_price, risk_free_rate, implied_volatility, time_to_expiry, 200)
        mc_american_price = monte_carlo_american(spot_price, strike_price, risk_free_rate, implied_volatility, time_to_expiry, 500)
        bs_european_prices.append(bs_european_price)
        crr_american_prices.append(crr_american_price)
        mc_american_prices.append(mc_american_price)

    # Convert lists to DataFrame
    bs_european_prices_df = pd.DataFrame(bs_european_prices, columns=['BS European'])
    crr_american_prices_df = pd.DataFrame(crr_american_prices, columns=['CRR American'])
    mc_american_prices_df = pd.DataFrame(mc_american_prices, columns=['Monte Carlo American'])

    # Calculate RMSE for each pricing method
    bs_european_rmse = calculate_rmse(put_options['lastPrice'], bs_european_prices_df)
    crr_american_rmse = calculate_rmse(put_options['lastPrice'], crr_american_prices_df)
    mc_american_rmse = calculate_rmse(put_options['lastPrice'], mc_american_prices_df)

    # Display RMSE values
    rmse_df = pd.DataFrame({
        'Method': ['Black-Scholes European', 'CRR American', 'Monte Carlo American'],
        'RMSE': [bs_european_rmse, crr_american_rmse, mc_american_rmse]
    })
    print(rmse_df)

    # Integrate CRR model
    crr_prices = []  # List to store calculated prices
    for i in range(len(put_options)):
        K = put_options.loc[i, 'strike']  # Extracting strike price for the current option
        sigma = put_options.loc[i, 'impliedVolatility']  # Extracting implied volatility for the current option
        put_price = binomial_tree_american(spot_price, K, risk_free_rate, sigma, time_to_expiry, 200)
        crr_prices.append(put_price)  # Append the calculated price to the list

    put_options['CRR Price'] = crr_prices

    plt.figure(figsize=(8, 6))
    plt.plot(put_options['strike'], put_options['CRR Price'], label='CRR Prices', marker='o')  
    plt.plot(put_options['strike'], put_options['lastPrice'], label='Actual Price', marker='s')  
    plt.xlabel('Strikes')
    plt.ylabel('CRR and Actual')
    plt.title('CRR and Actual against Different Strikes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate implied volatilities using py_vollib
    implied_vols_py_vollib = []
    for i in range(len(put_options)):
        try:
            implied_vol = implied_volatility(put_options['lastPrice'][i], put_options['strike'][i], spot_price, risk_free_rate, time_to_expiry, 'p')
            implied_vols_py_vollib.append(implied_vol)
        except:
            implied_vols_py_vollib.append(np.nan)

    put_options['Implied Volatility (Py_Vollib)'] = implied_vols_py_vollib
    # Plot implied volatilities
    plt.figure(figsize=(10, 6))
    plt.plot(strike_prices, implied_vols, marker='o', linestyle='-')
    plt.title('Implied Volatility vs Strike Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    spot_price = 188  # Example: Current spot price
    ticker_symbol = 'TSLA'  # Example: Apple Inc.
    expiration_date = '2024-05-24'  # Example: March 15, 2024
    main(spot_price, ticker_symbol, expiration_date)
