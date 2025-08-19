# Importing necessary libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Replace the file path with your actual file path
file_path = "C:/Users/John Mulready/Desktop/John Mulready/Downloads/Spring 2024/FIN 5883 - Quant Fin App/Assignment1_Data.csv"
#the tickers used are 'AMZN','BOKF','XOM','MSFT'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Re-formatting index
df.index = np.arange(1, len(df) + 1)

#Creating the matrix of the returns data
rmat = np.matrix(df.pivot(index='date', columns='TICKER', values='RET'))

# Setting random seed
np.random.seed(123)

# Full Sample IOS no short-sale constraints
M = rmat.shape[0]
N = rmat.shape[1]
u1 = np.matrix(np.ones(M)).T
mu0 = rmat.T * u1 / M
sigma0 = (rmat - mu0.T).T * (rmat - mu0.T) / (M - 1)
un = np.matrix(np.ones(N)).T
A = un.T * sigma0**(-1) * un
B = un.T * sigma0**(-1) * mu0
C = mu0.T * sigma0**(-1) * mu0
G = A * C - B**2
gmv_sd = np.sqrt(1 / A)
grid0 = np.matrix(np.arange(gmv_sd, 0.40 / np.sqrt(12), (0.40 / np.sqrt(12) - gmv_sd) / 100)).T

# Creates grid for plot
sdsq = np.matrix(np.multiply(grid0, grid0))

# Quadratic formula for IOS
erp_p0 = (-(-2 * B) + np.sqrt((-2 * B)**2 - 4 * np.multiply(A, (C - np.multiply(G, sdsq))))) / (2 * A)
erp_m0 = (-(-2 * B) - np.sqrt((-2 * B)**2 - 4 * np.multiply(A, (C - np.multiply(G, sdsq))))) / (2 * A)

# Quadratic formula for IOS
msrw0 = (sigma0**(-1) * mu0) / (un.T * sigma0**(-1) * mu0)  # max Sharpe ratio portfolio weights
msr0_mu = msrw0.T * mu0  # max Sharpe ratio portfolio expected return
msr0_sd = np.sqrt(msrw0.T * sigma0 * msrw0)  # max Sharpe ratio portfolio SD
gmvw0 = (sigma0**(-1) * un) / (un.T * sigma0**(-1) * un)  # global min var portfolio weights
gmv0_mu = gmvw0.T * mu0  # global min var expected return
gmv0_sd = np.sqrt(gmvw0.T * sigma0 * gmvw0)  # global min var portfolio SD

# Plots the IOS
plt.plot(grid0 * np.sqrt(12), erp_p0 * 12, grid0 * np.sqrt(12), erp_m0 * 12, color='black')
plt.xlim(0., 0.40)
plt.title("IOS")
plt.xlabel("Standard deviation (annualized)")
plt.ylabel("Expected return (annualized)")
plt.show()

# Simulating efficient frontier
nsim = 500
simmat = np.zeros((nsim + 1, len(sdsq)))
simmat[0, :] = erp_p0[1, :101].T
i = 1

while i <= nsim:
    rmat = np.random.multivariate_normal(np.ravel(mu0), sigma0, M)
    M = rmat.shape[0]
    N = rmat.shape[1]
    u1 = np.matrix(np.ones(M)).T  # unit vector
    mu = rmat.T * u1 / M  # mu vector
    sigma = (rmat - mu.T).T * (rmat - mu.T) / (M - 1)
    un = np.matrix(np.ones(N)).T
    A = un.T * sigma**(-1) * un
    B = un.T * sigma**(-1) * mu
    C = mu.T * sigma**(-1) * mu
    G = A * C - B**2
    gmv_sd = np.sqrt(1 / A)
    grid = np.matrix(np.arange(gmv_sd, 0.40 / np.sqrt(12), (0.40 / np.sqrt(12) - gmv_sd) / 100)).T
    sdsq = np.matrix(np.multiply(grid, grid))
    erp_p = (-(-2 * B) + np.sqrt((-2 * B)**2 - 4 * np.multiply(A, (C - np.multiply(G, sdsq))))) / (2 * A)
    simmat[i, :] = erp_p[1, :101].T
    plt.plot(grid * np.sqrt(12), erp_p * 12, linewidth=0.1)
    i = i+1

plt.plot(grid0 * np.sqrt(12), erp_p0 * 12, color='black', linewidth=3)
plt.xlim(0., 0.40)
plt.ylim(0, 0.40)
plt.title("IOS")
plt.xlabel("Standard deviation (annualized)")
plt.ylabel("Expected return (annualized)")
plt.show()

# Re-sampled Sharpe ratios
nsim = 2000
simmat = np.zeros((nsim + 1, 2))
simmat[0, 0] = msr0_sd
simmat[0, 1] = msr0_mu
i = 1

while i <= nsim:
    rmat = np.random.multivariate_normal(np.ravel(mu0), sigma0, M)
    M = rmat.shape[0]
    N = rmat.shape[1]
    u1 = np.matrix(np.ones(M)).T  # unit vector
    un = np.matrix(np.ones(N)).T
    mu = rmat.T * u1 / M  # mu vector
    sigma = (rmat - mu.T).T * (rmat - mu.T) / (M - 1)
    msrw = (sigma**(-1) * mu) / (un.T * sigma**(-1) * mu)
    msr_mu = msrw.T * mu0
    msr_sd = np.sqrt(msrw.T * sigma0 * msrw)
    simmat[i, 0] = msr_sd
    simmat[i, 1] = msr_mu
    plt.plot(msr_sd * np.sqrt(12), msr_mu * 12, color='green', marker='o', markersize=0.5)
    i = i+1

plt.plot(msr0_sd*np.sqrt(12),msr0_mu*12,color='black',marker='o')
plt.ylim(0,.6)
plt.xlim(0,.6)
plt.title("Re-sampled MSR portfolios in Mean_SD space")
plt.xlabel("Standard deviation (annualized)")
plt.ylabel("Expected return (annualized)")
plt.show()

# Re-sampled GMV portfolios
nsim = 2000
simmat = np.zeros((nsim + 1, 2))
simmat[0, 0] = gmv0_sd
simmat[0, 1] = gmv0_mu
i = 1

while i <= nsim:
    rmat = np.random.multivariate_normal(np.ravel(mu0), sigma0, M)
    M = rmat.shape[0]
    N = rmat.shape[1]
    u1 = np.matrix(np.ones(M)).T  # unit vector
    un = np.matrix(np.ones(N)).T
    mu = rmat.T * u1 / M  # mu vector
    sigma = (rmat - mu.T).T * (rmat - mu.T) / (M - 1)
    gmvw = (sigma**(-1) * un) / (un.T * sigma**(-1) * un)
    gmv_mu = gmvw.T * mu0
    gmv_sd = np.sqrt(gmvw.T * sigma0 * gmvw)
    simmat[i, 0] = gmv_sd
    simmat[i, 1] = gmv_mu
    plt.plot(gmv_sd * np.sqrt(12), gmv_mu * 12, color='green', marker='o', markersize=0.5)
    i = i+1

plt.plot(gmv0_sd * np.sqrt(12), gmv0_mu * 12, color='black', marker='o')
plt.title("Re-sampled GMV portfolios in Mean-SD space")
plt.xlabel("Standard deviation (annualized)")
plt.ylabel("Expected return (annualized)")
plt.show()


#create an empty matrix (size as big as number of simulations) then add in each simultion (sd, mean), (create another portfolio weight array and take summary stats of those as well.)
simmat


# Problem 9: Make a table with summary statistics for re-sampled GMV portfolio











# Problem 9: Make a table with summary statistics for re-sampled GMV portfolio
nsim = 2000
simmat = np.zeros((nsim + 1, 2))
simmat[0, 0] = gmv0_sd
simmat[0, 1] = gmv0_mu
i = 1

while i <= nsim:
    rmat = np.random.multivariate_normal(np.ravel(mu0), sigma0, M)
    M = rmat.shape[0]
    N = rmat.shape[1]
    u1 = np.matrix(np.ones(M)).T  # unit vector
    un = np.matrix(np.ones(N)).T
    mu = rmat.T * u1 / M  # mu vector
    sigma = (rmat - mu.T).T * (rmat - mu.T) / (M - 1)
    gmvw = (sigma**(-1) * un) / (un.T * sigma**(-1) * un)
    gmv_mu = gmvw.T * mu0
    gmv_sd = np.sqrt(gmvw.T * sigma0 * gmvw)
    simmat[i, 0] = gmv_sd
    simmat[i, 1] = gmv_mu
    plt.plot(gmv_sd * np.sqrt(12), gmv_mu * 12, color='green', marker='o', markersize=0.5)
    
    # Problem 9: Print weights for each stock
   # print(f"Simulation {i} - Portfolio Weights: {gmvw}")
    
    i = i + 1

plt.plot(gmv0_sd * np.sqrt(12), gmv0_mu * 12, color='black', marker='o')
plt.title("Re-sampled GMV portfolios in Mean-SD space")
plt.xlabel("Standard deviation (annualized)")
plt.ylabel("Expected return (annualized)")
plt.show()

# Existing code...

# Problem 9: Make a table with summary statistics for re-sampled GMV portfolio
gmv_stats = pd.DataFrame(simmat[:, :2], columns=["Standard Deviation", "Expected Return"])

# Add weights summary statistics for all stocks
for i in range(gmvw.shape[0]):
    weights_stats = pd.DataFrame([gmvw[i, 0]], columns=[f"Stock_{i}"])
    stats_i = weights_stats.describe(percentiles=[0.25, 0.75]).T
    stats_i["min"] = stats_i["min"].apply(lambda x: round(x, 6))
    stats_i["max"] = stats_i["max"].apply(lambda x: round(x, 6))
    stats_i.columns = [f"Weight_Stock_{i}_{col}" for col in stats_i.columns]
    gmv_stats = pd.concat([gmv_stats, stats_i.transpose()], ignore_index=True)

# Further adjustments (if needed)
gmv_stats = gmv_stats.describe(percentiles=[0.25, 0.75]).T
gmv_stats["min"] = gmv_stats["min"].apply(lambda x: round(x, 6))
gmv_stats["max"] = gmv_stats["max"].apply(lambda x: round(x, 6))

# Print the modified summary statistics table
print("Summary Statistics for Re-sampled GMV Portfolio:")
print(gmv_stats)




















# Problem 9: Make a table with summary statistics for re-sampled GMV portfolio
gmv_stats = pd.DataFrame(simmat[:, :3], columns=["Standard Deviation", "Expected Return","Portfolio Weights"])
gmv_stats = gmv_stats.describe(percentiles=[0.25, 0.75]).T
gmv_stats["min"] = gmv_stats["min"].apply(lambda x: round(x, 6))
gmv_stats["max"] = gmv_stats["max"].apply(lambda x: round(x, 6))
print("Summary Statistics for Re-sampled GMV Portfolio:")
print(gmv_stats)

# Problem 10: Make a table with summary statistics for re-sampled MSR portfolio
msr_stats = pd.DataFrame(simmat[:, :2], columns=["Standard Deviation", "Expected Return"])
msr_stats = msr_stats.describe(percentiles=[0.25, 0.75]).T
msr_stats["min"] = msr_stats["min"].apply(lambda x: round(x, 6))
msr_stats["max"] = msr_stats["max"].apply(lambda x: round(x, 6))
print("\nSummary Statistics for Re-sampled MSR Portfolio:")
print(msr_stats)

# Problem 11: Make a table with summary statistics for re-sampled Sharpe ratios
sharpe_stats = pd.DataFrame(simmat[:, 0], columns=["Sharpe Ratio"])
sharpe_stats = sharpe_stats.describe(percentiles=[0.25, 0.75]).T
sharpe_stats["min"] = sharpe_stats["min"].apply(lambda x: round(x, 6))
sharpe_stats["max"] = sharpe_stats["max"].apply(lambda x: round(x, 6))
print("\nSummary Statistics for Re-sampled Sharpe Ratios:")
print(sharpe_stats)