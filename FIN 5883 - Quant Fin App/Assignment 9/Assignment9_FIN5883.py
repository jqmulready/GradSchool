import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fredapi as fred

# Set up the FRED API
api_key = '9907934073a9f59241e8fdbe83762da1 '
fred = fred.Fred(api_key=api_key)

# 1. Collect the trailing 5 years of monthly data for each of the treasury series
start_date = '2019-04-01'
end_date = '2024-03-31'
tickers = ['GS1', 'GS3', 'GS2', 'GS5', 'GS7', 'GS10', 'GS30']

data = pd.DataFrame()
for ticker in tickers:
    try:
        df = fred.get_series(ticker, start_date, end_date)
        data[ticker] = df
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

data.index = pd.to_datetime(data.index)

# 2. Estimate the sample covariance matrix for the treasury securities
returns = data.dropna()
cov_matrix = returns.cov()

# 3. Calculate the first 3 principal components
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
order = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, order]
factors = returns.dot(eigenvectors[:, :3])
factors.columns = ['PC1', 'PC2', 'PC3']

factors['PC1'] -= 6

# 3a. Plot the 3 factor time series
factors.plot()
plt.title('Principal Component Factors')
plt.xlabel('Date')
plt.ylabel('Factor Value')
plt.show()

# 4. Simulate 1 5-year path for each of the treasury series using their respective factor loadings
sim_length = 60  # 5 years
sim_data = pd.DataFrame(index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=sim_length, freq='M'), columns=data.columns)

# Generate random factors
np.random.seed(42)
sim_factors = np.random.normal(0, 1, (sim_length, 3))
sim_factors = pd.DataFrame(sim_factors, columns=['PC1', 'PC2', 'PC3'])

# Calculate simulated returns
factor_loadings = eigenvectors[:, :3]
for col in data.columns:
    sim_data[col] = np.exp(sim_factors.dot(factor_loadings[data.columns.get_loc(col), :].T)).values

# Plot the simulated 5-year paths
sim_data.plot()
plt.title('Simulated 5-Year Treasury Yields')
plt.xlabel('Date')
plt.ylabel('Yield')
plt.show()