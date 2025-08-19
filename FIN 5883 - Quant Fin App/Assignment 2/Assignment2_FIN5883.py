import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge
import scipy.stats as stats


TICKERS = ["AAPL","BOKF","AMZN","XOM","TSLA","SPX"]
RISK_FREE = "^TNX"

# Pull in stock data and resample to 1 month
df = yf.Tickers(TICKERS).history('10y').Close.resample('1M').last().pct_change().dropna()
rf_df = (yf.Ticker(RISK_FREE).history('10y').Close.resample('1M').last()).dropna()/12
rf_df.index = rf_df.index.date

# Define the market return and risk-free rate
market_return = df['SPX']
risk_free_rate = rf_df

# Dictionary to store results
results = []

for ticker in TICKERS:
    # Define the dependent varaiable (stock return)
    stock_return = df[ticker]

    # Ordinary Least Squares (OLS)
    ols_model = sm.OLS(stock_return, sm.add_constant(market_return)).fit()
    #results[f"{ticker}_OLS"] = ols_model.params

    # Least Absolute Deviations (LAD)
    lad_model = sm.RLM(stock_return, sm.add_constant(market_return), M=sm.robust.norms.HuberT()).fit()
    #results[f"{ticker}_LAD"] = lad_model.params

    # Shrinkage Estimator (Ridge Regression)
    ridge_model = sm.OLS(stock_return, sm.add_constant(market_return)).fit_regularized(alpha=0.5, L1_wt=0.0)
    #results[f"{ticker}_Ridge"] = ridge_model.params

    # Store coefficients in results_data
    results.append({
        'Stock': i,
        'OLS Alpha': ols_model.params[0],  # OLS Intercept
        'OLS Beta': ols_model.params[1],  # OLS Coefficient
        'LAD Alpha': lad_model.params[0],  # LAD Intercept
        'LAD Beta': lad_model.params[1],  # LAD Coefficient
        'Ridge Alpha': ridge_model.params[0],  # Ridge Intercept
        'Ridge Beta': ridge_model.params[1],  # Ridge Coefficient
    })

    
    # Bayesian Regression

    #Using the normal distribution for the prior alpha and beta
    prior_mean_alpha = 0
    prior_std_alpha = 1
    prior_mean_beta = 0
    prior_std_beta = 1

    posterior_samples = []
    num_samples = 10000

    for _ in range(num_samples):
        # Sample from the priors
        alpha = np.random.normal(prior_mean_alpha, prior_std_alpha)
        beta = np.random.normal(prior_mean_beta, prior_std_beta)

        # Calculate the likelihood
        sigma_sq = 1  # Assuming constant variance for simplicity
        error = stock_return - (alpha + beta * market_return)
        likelihood_values = stats.norm.pdf(error, loc=0, scale=np.sqrt(sigma_sq))
        likelihood_value = np.prod(likelihood_values)

        # Calculate the unnormalized posterior
        unnormalized_posterior = likelihood_value

        # Store the samples
        posterior_samples.append((alpha, beta))

    # Convert posterior samples to DataFrame for easier analysis
    posterior_df = pd.DataFrame(posterior_samples, columns=['alpha', 'beta'])

    # Plot posterior distributions
    posterior_df.plot(kind='density', figsize=(10, 6))
    plt.title(f'Posterior Distributions of Intercept and Slope Coefficients for {ticker}')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Density')
    #plt.show()

# Create a DataFrame to store results
results_df = pd.DataFrame(results)
print(results_df)



''' # Bayesian Regression
    with pm.Model() as model:
        # Priors for the regression coefficients
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        alpha = pm.Normal("alpha", mu=0, sigma=10)

        # Model error
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Linear regression model
        mu = alpha + beta[0]*market_return + beta[1]*rf_df
        likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=rf_df)

        # Sample from the posterior distribution
        trace = pm.sample(2000, tune=1000)

    results_bayesian[i] = {"alpha": trace['alpha'].mean(), "beta_market": trace['beta'][0].mean(), "beta_risk_free": trace['beta'][1].mean()}

# Create a DataFrame to store results
results_df = pd.DataFrame(results)
#results_bayesian_df = pd.DataFrame(results_bayesian)

# Display coefficient results
print(results_df)
#print(results_bayesian_df)
'''


'''import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

TICKERS = ["SPX","AAPL", "BOKF", "AMZN", "XOM", "TSLA"]
RISK_FREE = "^TNX"

# Pull in stock data and resample to 1 month
df = yf.Tickers(TICKERS).history('5y').Close.resample('1M').last().pct_change()
df = df.dropna()

rf_df = (yf.Ticker(RISK_FREE).history('5y').Close.resample('1M').last())/12 # MAYBE NOT DIVIDE 12
rf_df = rf_df.dropna()
rf_df.index = rf_df.index.date

results_data = []  # List to store results

for i in TICKERS:
    # Define the independent variable (market return) and the dependent variable (stock return)
    X = df['SPX'] - rf_df  # Assuming 'SPX' is the market return column
    X = X.dropna()
    y = df[i] - rf_df  # Excess returns
    y = y.dropna()

    # Ordinary Least Squares (OLS)
    ols_model = sm.OLS(y, sm.add_constant(X)).fit()

    # Least Absolute Deviations (LAD)
    lad_model = sm.RLM(y, sm.add_constant(X), M=sm.robust.norms.LeastSquares()).fit()

    # Shrinkage Estimator (Ridge Regression)
    ridge_model = sm.OLS(y, sm.add_constant(X)).fit_regularized(alpha=0.1, L1_wt=0.0)

    # Store coefficients in results_data
    results_data.append({
        'Stock': i,
        'OLS Alpha': ols_model.params['const'],  # OLS Intercept
        'OLS Beta': ols_model.params[0],  # OLS Coefficient
        'LAD Alpha': lad_model.params['const'],  # LAD Intercept
        'LAD Beta': lad_model.params[0],  # LAD Coefficient
        'Ridge Alpha': ridge_model.params[0],  # Ridge Intercept
        'Ridge Beta': ridge_model.params[1]  # Ridge Coefficient
    })

# Create a DataFrame from the results_data list
results_df = pd.DataFrame(results_data)

# Display the results DataFrame
print("Alpha and Beta Results:")
print(results_df)




















import pandas as pd
import statsmodels.api as sm

apple_returns = df['AAPL']
market_return =df['SPX']


# Add a constant to the independent variable (market returns)
market_returns = sm.add_constant(market_return)

# Fit the OLS regression model
model = sm.OLS(apple_returns, market_returns).fit()

# Extract the beta coefficient
beta = model.params[0]  # The first parameter is the beta coefficient

print("Beta coefficient for Apple:", beta)
'''
