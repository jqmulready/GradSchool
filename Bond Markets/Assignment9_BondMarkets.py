import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import datetime
from dateutil.relativedelta import relativedelta
import math
from scipy.optimize import minimize

# Reads in the data
today = datetime.date(2024, 4, 25)  # Assuming the assignment date is April 25, 2024
yrs = 20
start = today - relativedelta(years=yrs)
end = today
delta = 1 / 12

# Calibrate the Vasicek model for the 10-year real-rate
series = 'REAINTRATREARAT10Y'
df = web.DataReader(series, 'fred', start, end).rename(columns={series: 'Rf'}).fillna(method='ffill')
if df.Rf.isnull()[0]:
    df.Rf[0] = df.Rf[1]
df['U1'] = 1

yy = np.matrix(df.Rf[1:len(df)]).T
xx = np.concatenate((np.matrix(df.U1[0:len(df) - 1]).T, np.matrix(df.Rf[0:len(df) - 1]).T), axis=1)
beta = np.linalg.inv(xx.T @ xx) @ xx.T @ yy
k = float((1 - beta[1][0]) / delta)
theta = float(beta[0][0] / (k * delta))
err = yy - xx @ beta
sigma = err.std() / (np.sqrt(delta))

# Calibrate the Vasicek model for the 10-year TIPS-implied inflation expectation
series = 'T10YIE'
df1 = web.DataReader(series, 'fred', start, end).rename(columns={series: 'Rf'}).fillna(method='ffill')
df1monthly = df1.resample('M').mean()
df1monthly = df1monthly.fillna(method='ffill')
if df1monthly.Rf.isnull()[0]:
    df1monthly.Rf[0] = df1monthly.Rf[1]
df1monthly['U1'] = 1

yy1 = np.matrix(df1monthly.Rf[1:len(df1monthly)]).T
xx1 = np.concatenate((np.matrix(df1monthly.U1[0:len(df1monthly) - 1]).T, np.matrix(df1monthly.Rf[0:len(df1monthly) - 1]).T), axis=1)
beta1 = np.linalg.inv(xx1.T @ xx1) @ xx1.T @ yy1
k1 = float((1 - beta1[1][0]) / delta)
theta1 = float(beta1[0][0] / (k1 * delta))
err1 = yy1 - xx1 @ beta1
sigma1 = err1.std() / (np.sqrt(delta))

# Read the bond data
bills = pd.read_csv('bills2024-04-25.txt', names=['Maturity', 'Yield'], sep='\t')
bonds = pd.read_csv('bonds2024-04-25.txt', names=['Maturity', 'Yield'], sep='\t')

# Combine bills and bonds data
bond_data = pd.concat([bills, bonds], ignore_index=True)
bond_data.sort_values(by='Maturity', inplace=True)
bond_data.reset_index(drop=True, inplace=True)


# Convert Maturity and Yield to numeric
bond_data['Maturity'] = pd.to_numeric(bond_data['Maturity'], errors='coerce')
bond_data['Yield'] = pd.to_numeric(bond_data['Yield'], errors='coerce')

# Drop rows with NaN values
bond_data.dropna(inplace=True)

# Reset the index
bond_data.reset_index(drop=True, inplace=True)

# Convert yields to discount factors
bond_data['DiscountFactor'] = (1 + bond_data['Yield'] / 100) ** (-bond_data['Maturity'])

# Define the objective function
def objective_function(params, sigma1, sigma2, bond_data):
    a, b, rho = params
    errors = []
    for i, row in bond_data.iterrows():
        t = 0
        T = row['Maturity']
        market_value = row['DiscountFactor']
        x_func = lambda t: theta  # Assuming theta1 = theta2 = 0
        y_func = lambda t: theta1
        theoretical_value = P(t, T, a, b, x_func, y_func, rho, sigma1, sigma2)
        errors.append((theoretical_value - market_value) ** 2)
    return sum(errors)

# Define the pricing function
def P(t, T, a, b, x_func, y_func, rho, sigma1, sigma2):
    term1 = (sigma1 ** 2 / a ** 2) * (T - t + (2 / a) * math.exp(-a * (T - t)) - (1 / (2 * a)) * math.exp(-2 * a * (T - t)) - (3 / (2 * a)))
    term2 = (sigma2 ** 2 / b ** 2) * (T - t + (2 / b) * math.exp(-b * (T - t)) - (1 / (2 * b)) * math.exp(-2 * b * (T - t)) - (3 / (2 * b)))
    term3 = 2 * rho * sigma1 * sigma2 / (a * b) * (T - t + ((math.exp(-a * (T - t)) - 1) / a) + ((math.exp(-b * (T - t)) - 1) / b) - ((math.exp(-(a + b) * (T - t)) - 1) / (a + b)))
    return math.exp(- (1 - math.exp(-a * (T - t))) * x_func(t) / a - (1 - math.exp(-b * (T - t))) * y_func(t) / b + 0.5 * (term1 + term2 + term3))

# Calibrate the remaining parameters
initial_guess = [0.5, 0.5, 0.5]  # Initial guess for a, b, and rho
result = minimize(objective_function, initial_guess, args=(sigma, sigma1, bond_data), method='nelder-mead')
a, b, rho = result.x

# Plot the yield curves
ttm = np.arange(0, 30 + 1 / 12, 1 / 12)
market_yields = -1 / ttm * np.log(bond_data['DiscountFactor'].values)
model_yields = []
for t in ttm:
    x_func = lambda t: theta
    y_func = lambda t: theta1
    model_yields.append(-1 / t * np.log(P(0, t, a, b, x_func, y_func, rho, sigma, sigma1)))

plt.figure(figsize=(10, 6))
plt.plot(ttm, market_yields, label='Market Yield Curve')
plt.plot(ttm, model_yields, label='2-Factor Vasicek Model Yield Curve')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('Yield')
plt.title('Yield Curve Comparison')
plt.legend()
plt.show()

print(f"Calibrated parameters: a={a}, b={b}, rho={rho}")