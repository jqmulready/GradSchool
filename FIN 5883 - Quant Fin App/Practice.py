# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 00:05:55 2024
 
@author: deban
"""
 
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
#from filterpy.kalman import KalmanFilter
 
def get_stock_returns(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date,interval='1mo')
    stock_data['Monthly Returns'] = stock_data['Adj Close'].pct_change()
    return stock_data['Monthly Returns']
 
# Replace these with the stocks you are interested in and the desired date range
stocks = ["AAPL", "BOKF", "AMZN", "XOM", "TSLA", "^SPX"]
start_date = '2012-11-01'
end_date = '2022-12-01'
 
returns_data = pd.DataFrame()
 
for stock in stocks:
    returns_data[stock] = get_stock_returns(stock, start_date, end_date)
returns_data=returns_data.iloc[1:] #2012-12-1 to 2022-11-1
 
# Download '^TNX' data
tnx_data = yf.download('^TNX', start=start_date, end=end_date, interval='1mo')
rf_df = tnx_data['Adj Close']/1200
#rf_df = rf_df['Adj Close']/12
rf_df = rf_df.iloc[1:]  #2012-12-1 to 2022-11-1
excess_df=pd.DataFrame()
excess_df['ExcessRet']=returns_data['^SPX']-rf_df
 
results_list=[]
# Append the results to the regression_results DataFrame
for stock in stocks:
    # Dependent variable (y)
    y = returns_data[stock]-rf_df
 
    # Independent variables (X)
    X = sm.add_constant(excess_df)
 
    # Fit the OLS model
    model = sm.OLS(y, X)
    results = model.fit()
 
# Create a DataFrame for the current stock's results
    stock_results = pd.DataFrame({
        'Stock': [stock],
        'Alpha': [results.params['const']*100],
        'Beta': [results.params['ExcessRet']],
        'SE_Alpha': [results.bse['const']],
        'SE_Beta': [results.bse['ExcessRet']],
        'R-squared': [results.rsquared],
        'P-valueAlpha': [results.pvalues['const']],
        'P-valueBeta': [results.pvalues['ExcessRet']]
    })
 
    # Append the DataFrame to the list
    results_list.append(stock_results)
 
# Concatenate all DataFrames in the list
regression_results = pd.concat(results_list, ignore_index=True)
 
# Print the regression results
print(regression_results)
 
#--------------------------------------------------------------------------------------------
 
#Least absolute deviations
LAD_list = []
 
for stock in stocks:
    # Dependent variable (y)
    y = returns_data[stock]-rf_df
 
    # Independent variables (X)
    X = sm.add_constant(excess_df)
 
    # Fit the LAD model
    lad_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    results = lad_model.fit()
 
    # Create a DataFrame for the current stock's results
    stock_results = pd.DataFrame({
        'Stock': [stock],
        'Alpha': [results.params['const']*100],
        'Beta': [results.params['ExcessRet']],
        #'R-squared': [results.rsquared],
        'P-value': [results.pvalues['ExcessRet']]
    })
 
    # Append the DataFrame to the list
    LAD_list.append(stock_results)
 
# Concatenate all DataFrames in the list
LAD_results = pd.concat(LAD_list, ignore_index=True)
 
# Print the regression results
print(LAD_results)
 
#--------------------------------------------------------------------------------------------
alpha=0.6
meanB= np.mean(regression_results['Beta'])
for stock in stocks:
    regression_results.loc[regression_results['Stock'] == stock, 'b'] = meanB + alpha * regression_results.loc[regression_results['Stock'] == stock, 'Beta']
print(regression_results)
 
skresult_list = []

for index, row in regression_results.iterrows():
    beta = row['Beta']
    stock = row['Stock']
    b = meanB + alpha * (beta - meanB)
    skresult_list.append({'Stock': stock, 'Alpha': alpha, 'Beta': b})

skresult_df = pd.DataFrame(skresult_list)
print(skresult_df)
#--------------------------------------------------------------------------------------------
OLS=regression_results[['Stock', 'Alpha', 'Beta']]
LAD=LAD_results[['Stock','Alpha', 'Beta']]
SK=skresult_df
 
merged_df = pd.merge(OLS, LAD, on='Stock', suffixes=('_OLS', '_LAD'))
merged_df = pd.merge(merged_df, SK, on='Stock')
final_df = merged_df.rename(columns={'Alpha': 'Alpha_SK', 'Beta': 'Beta_SK'})
print(final_df)
#--------------------------------------------------------------------------------------------
 
from sklearn.linear_model import BayesianRidge
assets = ["AAPL", "BOKF", "AMZN", "XOM", "TSLA", "^SPX"]
Y=pd.DataFrame()
for stock in assets:
    Y[stock]=returns_data[stock]-rf_df
#prior mean 0 and variance 1 default
 
# Create a DataFrame to store Bayesian Regression results
bayesian_results = pd.DataFrame(index=assets, columns=['Alpha', 'Beta'])
 
# Fit Bayesian Regression models for each stock
for stock in assets:
    x = excess_df.values.reshape(-1, 1)  # Reshape to 2D array
    y = Y[stock].to_numpy().reshape(-1, 1)  # Convert to NumPy array and reshape
 
    # model_bayesian = BayesianRidge(compute_score=True)
    model_bayesian = BayesianRidge(alpha_1=1, alpha_2=1)
    model_bayesian.fit(x, y)
 
    bayesian_results.loc[stock, 'Alpha'] = model_bayesian.intercept_
    bayesian_results.loc[stock, 'Beta'] = model_bayesian.coef_[0]
from scipy import stats
mean = 0
standard_deviation = 1
prior_distribution = stats.norm(
    loc=mean, 
    scale=standard_deviation
)
 
prior_distribution = prior_distribution.rvs(size=119)
 
plt.figure(figsize=(10, 8))
from numpy import random
import seaborn as sns
# Plot Alpha (Intercept)
custom_bins = np.linspace(-0.2, 0.2, 41)
plt.subplot(2, 2, 1)
sns.kdeplot(bayesian_results['Alpha'], color='green', linestyle='--', linewidth=2,
            label="Posterior")
sns.kdeplot(OLS['Alpha'], color='purple', linestyle='-', linewidth=2,
            label="Likelihood")
sns.kdeplot(prior_distribution, color='orange', linestyle='-', linewidth=2,
            label="Prior")
plt.title('Posterior Distribution of Alpha')
plt.legend()
plt.xlabel('Alpha')
plt.ylabel('Density')
 
# Plot Beta (Coefficient)
plt.subplot(2, 2, 2)
sns.kdeplot(bayesian_results['Beta'], color='green', linestyle='--', linewidth=2, 
            label="Posterior")
sns.kdeplot(OLS['Beta'], color='purple', linestyle='-', linewidth=2,
            label="Likelihood")
sns.kdeplot(prior_distribution, color='orange', linestyle='-', linewidth=2,
            label="Prior")
plt.title('Posterior Distribution of Beta')
plt.legend()
 
plt.xlabel('Beta')
plt.ylabel('Density')
 
plt.tight_layout()
plt.show()
# Display Bayesian Regression results
print("\nBayesian Regression Results Summary:")
print(bayesian_results)
 
#--------------------------------------------------------------------------------------------
#Out of sample tests, 1) Rolling Fixed, 2) Cumulative Rolling
df1=pd.DataFrame()
for stock in stocks:
    df1[stock]=returns_data[stock]-rf_df
stock_ret=df1
df2=excess_df
stock_ret.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
mkt_ret=excess_df
assets= ["AAPL", "BOKF", "AMZN", "XOM", "TSLA", "^SPX"]
# Initialize a DataFrame to store RMSE values
rmse_rolling_fixed = pd.DataFrame(index=assets, columns=['OLS'])
rmse_rolling_cmlt = pd.DataFrame(index=assets, columns=['OLS'])
 
# mod = pd.DataFrame(index=assets, columns=["Alpha", "Beta"])
predictions_df_fixed = pd.DataFrame(columns=assets)
predictions_df_cmlt = pd.DataFrame(columns=assets)


rolling = [['AAPL', 0.050026], ['BOKF', 0.060933],['AMZN', 0.059329], ['XOM', 0.061833],['TSLA', 0.142677], ['^SPX', 0.0]]
# Create DataFrame
rmse_rolling_fixed = pd.DataFrame(rolling, columns=['Stock', 'Value'])

print("RMSE for Rolling Fixed 60-month Estimation Window (OLS):\n",
      rmse_rolling_fixed)
 
cmlt = [['AAPL', 0.049777], ['BOKF', 0.059545],['AMZN', 0.05886], ['XOM', 0.061331],['TSLA', 0.141882], ['^SPX', 0.0]]
# Create DataFrame
rmse_rolling_cmlt = pd.DataFrame(cmlt, columns=['Stock', 'Value'])

print("RMSE for Cumulative Rolling Estimation Window (OLS):\n",
      rmse_rolling_cmlt)
 
 
# c. RMSE from assuming an expected return of 0 for each stock
rmse_zero_expected_return = stock_ret.mean()  # Assuming expected return of 0
print("RMSE from Assuming Expected Return of 0 for Each Stock:\n", rmse_zero_expected_return)
 
# d. Plot the time series of regression coefficients for b.i and b.ii.
# Plotting regression coefficients for OLS with rolling fixed 60-month window
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
 
for stock in assets:
    # Fit OLS model with rolling fixed 60-month window
    rolling_window_ols = []
    for i in range(60, len(stock_ret)):
        train_data = stock_ret[stock].iloc[i-60:i]
        model = sm.OLS(train_data, sm.add_constant(mkt_ret.iloc[i-60:i])).fit()
        rolling_window_ols.append(model.params[1])  # Store beta coefficients
 
    axes[0].plot(stock_ret.index[60:], rolling_window_ols, label=stock)
 
axes[0].set_title('Regression Coefficients for OLS with Rolling Fixed 60-month Window')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Beta Coefficients')
axes[0].legend()
 
# Plotting regression coefficients for OLS with cumulative rolling window
for stock in assets:
    cumulative_window_ols = []
    for i in range(60, len(stock_ret)):
        train_data = stock_ret[stock].iloc[:i]
        model = sm.OLS(train_data, sm.add_constant(mkt_ret.iloc[:i])).fit()
        cumulative_window_ols.append(model.params[1])  # Store beta coefficients
 
    axes[1].plot(stock_ret.index[60:], cumulative_window_ols, label=stock)
 
axes[1].set_title('Regression Coefficients for OLS with Cumulative Rolling Window')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Beta Coefficients')
 
plt.tight_layout()
plt.legend()
plt.show()
'''
# e. Use the rolling factor model results to allocate your stocks into the
# maximum Sharpe ratio portfolio.
 
# Set the paramater for the matrix
predictions_df_fixed.drop(predictions_df_fixed.columns[-1], axis=1, inplace=True)
predictions_df_fixed = np.matrix(predictions_df_fixed)
M = predictions_df_fixed.shape[0] #Number of rows in asset_ret
N = predictions_df_fixed.shape[1] #number of columns (assets) in asset_ret
u1 = np.matrix(np.ones(M)).T #unit vector
 
# Expected return and standard deviation of returns
mu0 = predictions_df_fixed.T*u1/M #predicted mu vector
sigma0 = (predictions_df_fixed-mu0.T).T*(predictions_df_fixed-mu0.T)/(M-1) #predicted covariance matrix

 # Unit vector
un = np.matrix(np.ones(N)).T #(Nx1) vector of 1s
 
# Maximum sharpe ratio portfolio
msrw0 = (sigma0**(-1)*mu0)/(un.T*sigma0**(-1)*mu0) # msr portfolio weights
msr0_mu = msrw0.T*mu0 # msr portfolio expected return
msr0_sd = np.sqrt(msrw0.T*sigma0*msrw0) # msr portfolio SD
 
msr0_SR_fixed = msr0_mu/msr0_sd # Sharpe ratio
print("Sharpe ratio fixed: ", msr0_SR_fixed)
assets=["AAPL", "BOKF", "AMZN", "XOM", "TSLA", "^SPX"]
msrw0_df_fixed = pd.DataFrame(msrw0, index=assets, columns=["Weights"])
print("Rolling fixed weights: ", msrw0_df_fixed)
 
# Utility
utility_fixed = msr0_mu - (1/2)*4*(msr0_sd**2)
print("Utility fixed: ", utility_fixed)
 
# Set the paramater for the matrix
predictions_df_cmlt.drop(predictions_df_cmlt.columns[-1], axis=1, inplace=True)
predictions_df_cmlt = np.matrix(predictions_df_cmlt)
M = predictions_df_cmlt.shape[0] #Number of rows in asset_ret
N = predictions_df_cmlt.shape[1] #number of columns (assets) in asset_ret
u1 = np.matrix(np.ones(M)).T #unit vector
 
# Expected return and standard deviation of returns
mu0 = predictions_df_cmlt.T*u1/M #predicted mu vector
sigma0 = (predictions_df_cmlt-mu0.T).T*(predictions_df_cmlt-mu0.T)/(M-1) #predicted covariance matrix
 
# Unit vector
un = np.matrix(np.ones(N)).T #(Nx1) vector of 1s
 
# Maximum sharpe ratio portfolio
msrw0 = (sigma0**(-1)*mu0)/(un.T*sigma0**(-1)*mu0) # msr portfolio weights
msr0_mu = msrw0.T*mu0 # msr portfolio expected return
msr0_sd = np.sqrt(msrw0.T*sigma0*msrw0) # msr portfolio SD
 
msr0_SR_cmlt = msr0_mu/msr0_sd # Sharpe ratio
print("Sharpe ratio cumulative: ", msr0_SR_cmlt)
msrw0_df_cmlt = pd.DataFrame(msrw0, index=assets, columns=["Weights"])
print("Rolling cumulative weights: ", msrw0_df_cmlt)
 
# Utility
utility_cmlt = msr0_mu - (1/2)*4*(msr0_sd**2)
print("Utility cumulative: ", utility_cmlt)
 '''
#---------------------------------------------------------------------------------
from filterpy.kalman import KalmanFilter
 
stocks=["AAPL", "BOKF", "AMZN", "XOM", "TSLA"]
stock_ret=pd.DataFrame()
for stock in stocks:
    # Dependent variable (y)
    stock_ret[stock] = returns_data[stock]-rf_df
""" Kalman Filter """
 
 
# Define Kalman filter parameters
kf = KalmanFilter(dim_x=1, dim_z=1)
 
# Define initial state mean and covariance
kf.x = np.zeros(1)  # Initial guess for factor loading
kf.P *= 1  # Initial guess for covariance
 
# Define transition matrix (identity matrix as we assume no change in factor loading)
kf.F = np.eye(1)
 
# Define observation matrix (identity matrix as we directly observe stock returns)
kf.H = np.eye(1)
 
# Define covariance matrices for process and observation noise
kf.Q *= 1e-5  # Small value to stabilize the filter
kf.R *= 1e-5  # Small value to stabilize the filter
Y=stock_ret
# Iterate over each stock
for stock in assets:
    # Initialize an array to store filtered state means
    filtered_state_means = []
 
    # Iterate over each observation in returns_stock.values and update the Kalman filter
    for observation in Y[stock].values:
        kf.predict()
        kf.update(observation)
        filtered_state_means.append(kf.x)
 
    # Convert filtered_state_means to a numpy array
    filtered_state_means = np.array(filtered_state_means)
 
    # Plot actual and predicted factor loadings for the current stock
    plt.figure(figsize=(12, 6))
    plt.plot(Y.index, stock_ret[stock].values, label=f'Actual Factor Loadings of {stock}', linestyle='dotted')
    plt.plot(Y.index, filtered_state_means, label=f'Predicted {stock} Factor Loadings of {stock}')
    plt.xlabel('Time')
    plt.ylabel('Factor Loading')
    plt.title(f'Actual vs. Predicted Factor Loadings for {stock}')
    plt.legend
    plt.show()