# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:26:52 2024

@author: John Mulready
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import datetime
import dateutil
import bs4 as bs
import requests
from dateutil.parser import parse
import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import matplotlib as mpl
from scipy.optimize import minimize

mpl.rcParams['figure.dpi'] = 300  # Set figure DPI for higher resolution

# Create a data range to pull our interest rate data so we can determine the volatility


daily_rate_delta = 1 / 12.0 # trading days in the year approximetly
treasury_series = 'REAINTRATREARAT10Y'

end = datetime.datetime(2024,4,25)
# start=end-relativedelta(years=years_used)
start = datetime.datetime(2004,4,1)
years_used = 20 # 20 years of data as specified in the assignment

print('Rate data dates:')
print(f'Start date as Datetime: {start}')
print(f'End date as Datetime: {end}')
rate_data = web.DataReader(treasury_series,'fred',start,end).rename(columns={treasury_series:'Rf'})


rate_data

rate_data = rate_data.copy().ffill()
rate_data['Rf'] = rate_data['Rf'] / 100.0
if(rate_data['Rf'].isnull().iloc[0]):
    rate_data['Rf'].iloc[0]=rate_data['Rf'].iloc[1]
rate_data['U1']=1
rate_data

initial_rate = rate_data.iloc[-1]['Rf'] # initial rate used in outr models

'''
Create a linear regression model so we can determine the volatility and then an initial k and theta to use

'''

# the y (rate) series in our regression model
# drop the first value in the series
y_series = np.matrix(rate_data['Rf'][1:len(rate_data)]).T
x_series=np.concatenate((np.matrix(rate_data['U1'][0:len(rate_data)-1]).T,\
                   np.matrix(rate_data['Rf'][0:len(rate_data)-1]).T),axis=1)

inverse_matrix = np.linalg.inv(x_series.T @ x_series) # the inversion matrix we will use to calculate beta, simply the inverse matrix of the dot product of the x series transposed and the x series

# receive the coefficients for the Vasicek Model
beta_values = inverse_matrix @ x_series.T @ y_series

# now we determine the volatility based on the standard errors of the regression model
errs = y_series - x_series @ beta_values
volatility = float(errs.std() / (daily_rate_delta ** 0.5))


k_value = float(( 1 - beta_values[1][0] ) / daily_rate_delta)
theta_value = float(beta_values[0][0] / ( k_value * daily_rate_delta ))
print(f'{treasury_series} Rate on {rate_data.iloc[0].name.date()}: {round(initial_rate*100,3)}%')
print(f'K Value from linear regression: {k_value}')
print(f'Theta value from linear regression: {theta_value}')
print(f'Volatility from linear regression of rates: {volatility}')

volatility_rates = volatility
rate_data_store = rate_data

# now we run it through the inflation data
treasury_series = 'T10YIE'
rate_data = web.DataReader(treasury_series,'fred',start,end).rename(columns={treasury_series:'Rf'})
rate_data = rate_data.copy().ffill()
rate_data['Rf'] = rate_data['Rf'] / 100.0
if(rate_data['Rf'].isnull().iloc[0]):
    rate_data['Rf'].iloc[0]=rate_data['Rf'].iloc[1]
rate_data['U1']=1
rate_data

rate_data = rate_data.groupby(pd.Grouper(freq='M')).tail(1)

initial_rate = rate_data.iloc[-1]['Rf'] # initial rate used in outr models

'''
Create a linear regression model so we can determine the volatility and then an initial k and theta to use

'''

# the y (rate) series in our regression model
# drop the first value in the series
y_series = np.matrix(rate_data['Rf'][1:len(rate_data)]).T
x_series=np.concatenate((np.matrix(rate_data['U1'][0:len(rate_data)-1]).T,\
                   np.matrix(rate_data['Rf'][0:len(rate_data)-1]).T),axis=1)

inverse_matrix = np.linalg.inv(x_series.T @ x_series) # the inversion matrix we will use to calculate beta, simply the inverse matrix of the dot product of the x series transposed and the x series

# receive the coefficients for the Vasicek Model
beta_values = inverse_matrix @ x_series.T @ y_series

# now we determine the volatility based on the standard errors of the regression model
errs = y_series - x_series @ beta_values
volatility = float(errs.std() / (daily_rate_delta ** 0.5))


k_value = float(( 1 - beta_values[1][0] ) / daily_rate_delta)
theta_value = float(beta_values[0][0] / ( k_value * daily_rate_delta ))
print(f'{treasury_series} Rate on {rate_data.iloc[0].name.date()}: {round(initial_rate*100,3)}%')
print(f'K Value from linear regression: {k_value}')
print(f'Theta value from linear regression: {theta_value}')
print(f'Volatility from linear regression of inflation: {volatility}')

volatility_inflation = volatility

inflation_data = rate_data
rate_data = rate_data_store

inflation_data

# retrieve the initial rates for the models that we'll need later
initial_rate = rate_data.iloc[-1]['Rf']
initial_inflation = inflation_data.iloc[-1]['Rf']


delta = 1.0 / 12 # a 3 month delta so we can find times to maturities closest ot it
years_used=10
total_periods = int(years_used/delta)
times_to_maturities = np.zeros((total_periods, 1))
market_bond_prices = np.zeros((total_periods, 1))

maturity = delta # set the first maturity equal to the delta since we're starting after 1 month

# make the first row the price of a zero bound expiring at the current date

for index in range(0, total_periods):

    times_to_maturities[index, 0] = maturity
    market_bond_prices[index, 0] = bond_polynomial_function(maturity)
    maturity += delta # increment the maturity
    
print(f'Volatility from linear regression of rates: {volatility_rates}')
print(f'Volatility from linear regression of inflation: {volatility_inflation}')

def calibrate_two_factor_model(model_params):
    
    initial_rate
    initial_inflation
    
    volatility_rates
    volatility_inflation

    a_value = abs(model_params[0]) #Absolute value is used to make this an unconstrained opt. problem.
    b_value = abs(model_params[1]) #Absolute value is used to make this an unconstrained opt. problem.
    rho_value = abs(model_params[2]) #Absolute value is used to make this an unconstrained opt. problem., has to be between -1 and +1

    # get the V values we will need
    V_array =\
    (volatility_rates**2/(a_value**2)) * ( times_to_maturities + (2/a_value) * np.exp(-a_value*times_to_maturities)-(1/(2*a_value))* np.exp(-2*a_value*times_to_maturities)-3/(2*a_value))\
    +(volatility_inflation**2/(b_value**2)) * ( times_to_maturities + (2/b_value) * np.exp(-b_value*times_to_maturities)-(1/(2*b_value))* np.exp(-2*b_value*times_to_maturities)-3/(2*b_value))\
    +(2 * rho_value) *(volatility_rates*volatility_inflation)/(a_value*b_value)\
    *(
    times_to_maturities\
    +(np.exp(-a_value*times_to_maturities)-1)/a_value\
    +(np.exp(-b_value*times_to_maturities)-1)/b_value\
    +(np.exp(-(a_value+b_value)*times_to_maturities)-1)/(a_value+b_value)                                                                       \
    )
    
    price_array =np.exp(\
    -((1 - np.exp(-a_value*times_to_maturities))/a_value)*initial_rate+\
    -((1 - np.exp(-b_value*times_to_maturities))/b_value)*initial_inflation+\
    (1 / 2.0) * V_array)
                                                                         
    # some of squared errors for minimization
    sse=np.sum((market_bond_prices - price_array)**2)
    
    return sse

# def calibrate_two_factor_model(model_params):
    
#     initial_rate
#     initial_inflation
    
#     volatility_rates
#     volatility_inflation

#     a_value = abs(model_params[0]) #Absolute value is used to make this an unconstrained opt. problem.
#     b_value = abs(model_params[1]) #Absolute value is used to make this an unconstrained opt. problem.
#     rho_value = abs(model_params[2]) #Absolute value is used to make this an unconstrained opt. problem., has to be between -1 and +1

#     v_array = (volatility_rates**2/a_value**2)\
#     *(times_to_maturities + (2/a_value)*np.exp(-a_value*times_to_maturities)-(1/(2*a_value))*np.exp(-2*a_value*times_to_maturities)-3/(2*a_value))\
#     +(volatility_inflation**2/b_value**2)\
#     *(times_to_maturities + (2/b_value)*np.exp(-b_value*times_to_maturities)-(1/(2*b_value))*np.exp(-2*b_value*times_to_maturities)-3/(2*b_value))\
#     +2*rho_value*(volatility_rates*volatility_inflation)/(a_value*b_value)\
#     *(
#     times_to_maturities\
#     +(np.exp(-a_value*times_to_maturities)-1)/a_value\
#     +(np.exp(-b_value*times_to_maturities)-1)/b_value\
#     +(np.exp(-(a_value+b_value)*times_to_maturities)-1)/(a_value+b_value)
#     )
    
#     price_array = np.exp(
#     - (1-np.exp(-a_value*times_to_maturities))/a_value*initial_rate\
#     - (1-np.exp(-b_value*times_to_maturities))/b_value*initial_inflation\
#     + (1/2.0)*v_array)
    
#     # some of squared errors for minimization
#     sse=np.sum((market_bond_prices - price_array)**2)
    
#     return sse

vasicek_params_0=np.c_[0.01,0.01,0.01] #Starting values for k and theta
res = minimize(calibrate_two_factor_model,vasicek_params_0.flatten(),method='L-BFGS-B') # call the optimization function

optimal_a_value = abs(res.x[0])
optimal_b_value = abs(res.x[1])
optimal_rho_value = abs(res.x[2])

# def constraint_func1(model_params):
#     return [model_params[0] + model_params[1] - 1]

# def constraint_func2(model_params):
#     # a must be less than 1
#     return [1-model_params[0]]

# def constraint_func3(model_params):
#     # b must be less than 1
#     return [1.0-model_params[1]]


# def constraint_func4(model_params):
#     # rho  msut be less than 1
#     return [1.0-model_params[2]]

# def constraint_func5(model_params):
#     # b must be greater than 0
#     return [model_params[1]]

# vasicek_params_0=np.c_[1.0,0.01,0.01] #Starting values for parameters
# res = minimize(calibrate_two_factor_model,
#                vasicek_params_0.flatten(), 
#                method='SLSQP', 
#                constraints=
#                [
#                 {'type': 'eq', 'fun': constraint_func1},
#                 {'type': 'ineq', 'fun': constraint_func2},
#                 {'type': 'ineq', 'fun': constraint_func3},
#                 {'type': 'ineq', 'fun': constraint_func4},
#                 {'type': 'ineq', 'fun': constraint_func5}
#                ],

#               ) # call the optimization function

# optimal_a_value = abs(res.x[0])
# optimal_b_value = abs(res.x[1])
# optimal_rho_value = abs(res.x[2])

print(f'Optimal a value: {optimal_a_value}')
print(f'Optimal b value: {optimal_b_value}')
print(f'Optimal rho value: {optimal_rho_value}')
print(f'Two factor calibration errors given calibrated values: {round(calibrate_two_factor_model(np.c_[optimal_a_value, optimal_b_value,optimal_rho_value].flatten()),3)}')


def two_factor_price_function(a_value, b_value, rho_value,times_to_maturities):
    
    initial_rate
    initial_inflation
    
    volatility_rates
    volatility_inflation

    # get the V values we will need
    V_array =\
    (volatility_rates**2/(a_value**2)) * ( times_to_maturities + (2/a_value) * np.exp(-a_value*times_to_maturities)-(1/(2*a_value))* np.exp(-2*a_value*times_to_maturities)-3/(2*a_value))\
    +(volatility_inflation**2/(b_value**2)) * ( times_to_maturities + (2/b_value) * np.exp(-b_value*times_to_maturities)-(1/(2*b_value))* np.exp(-2*b_value*times_to_maturities)-3/(2*b_value))\
    +(2 * rho_value) *(volatility_rates*volatility_inflation)/(a_value*b_value)\
    *(
    times_to_maturities\
    +(np.exp(-a_value*times_to_maturities)-1)/a_value\
    +(np.exp(-b_value*times_to_maturities)-1)/b_value\
    +(np.exp(-(a_value+b_value)*times_to_maturities)-1)/(a_value+b_value)                                                                       \
    )
    
    price_array =np.exp(\
    -((1 - np.exp(-a_value*times_to_maturities))/a_value)*initial_rate+\
    -((1 - np.exp(-b_value*times_to_maturities))/b_value)*initial_inflation+\
    (1 / 2.0) * V_array)
    
    return price_array

# def two_factor_price_function(a_value, b_value, rho_value,times_to_maturities):
    
#     initial_rate
#     initial_inflation
    
#     volatility_rates
#     volatility_inflation

#     v_array = (volatility_rates**2/a_value**2)\
#     *(times_to_maturities + (2/a_value)*np.exp(-a_value*times_to_maturities)-(1/(2*a_value))*np.exp(-2*a_value*times_to_maturities)-3/(2*a_value))\
#     +(volatility_inflation**2/b_value**2)\
#     *(times_to_maturities + (2/b_value)*np.exp(-b_value*times_to_maturities)-(1/(2*b_value))*np.exp(-2*b_value*times_to_maturities)-3/(2*b_value))\
#     +2*rho_value*(volatility_rates*volatility_inflation)/(a_value*b_value)\
#     *(
#     times_to_maturities\
#     +(np.exp(-a_value*times_to_maturities)-1)/a_value\
#     +(np.exp(-b_value*times_to_maturities)-1)/b_value\
#     +(np.exp(-(a_value+b_value)*times_to_maturities)-1)/(a_value+b_value)
#     )
    
#     price_array = np.exp(
#     - (1-np.exp(-a_value*times_to_maturities))/a_value*initial_rate\
#     - (1-np.exp(-b_value*times_to_maturities))/b_value*initial_inflation\
#     + (1/2.0)*v_array)
    
#     return price_array

delta_graphing = 1.0 / 12 # a 1 month delta so we can find times to maturities closest ot it
starting_maturity = 3.0 / 12 # starting at 3 months

times_to_maturities_plotted=np.c_[np.arange(starting_maturity,years_used+delta_graphing/2,delta_graphing)] # we're using this array to plot the modeled term structure
market_prices = np.zeros((1,len(times_to_maturities_plotted)))

for index in range(0, len(times_to_maturities_plotted)):
    
    # pull the market prices of the bonds using the polynomial function
    market_prices[0, index] = bond_polynomial_function(times_to_maturities_plotted[index])

market_prices = market_prices.T # transpose it so we can use it for our operations

modeled_prices = two_factor_price_function(
    a_value=optimal_a_value,
    b_value=optimal_b_value,
    rho_value=optimal_rho_value,
    times_to_maturities=times_to_maturities_plotted)

# the yields of the prices assuming continuous compounding
modeled_yields = -1/times_to_maturities_plotted*np.log(modeled_prices)
market_yields =  -1/times_to_maturities_plotted*np.log(market_prices)

plt.title(f'Calibrated Two Factor Vasicek term structure of interest rates\nσ={round(volatility_rates,5)},η={round(volatility_inflation,5)}\na={round(optimal_a_value,3)}, b={round(optimal_b_value,3)}, ρ={round(optimal_rho_value,5)}')
plt.plot(times_to_maturities_plotted,market_yields,color='black')
plt.plot(times_to_maturities_plotted,modeled_yields, \
        linestyle='none',marker='o',fillstyle='none', color='blue')

plt.ylabel('Interest rate')
plt.xlabel('Time-to-maturity')
plt.gca().legend(('True rate','Calibrated rate'),loc='best')
plt.show()

plt.title(f'Calibrated Two Factor Vasicek Modeled Bond Prices\nσ={round(volatility_rates,5)},η={round(volatility_inflation,5)}\na={round(optimal_a_value,3)}, b={round(optimal_b_value,3)}, ρ={round(optimal_rho_value,5)}')

plt.plot(times_to_maturities_plotted, market_prices,color='black')
plt.plot(times_to_maturities_plotted, modeled_prices, \
        linestyle='none',marker='o',fillstyle='none', color='blue')

plt.ylabel('Discount Factor')
plt.xlabel('Time-to-maturity')
plt.gca().legend(('True price','Calibrated price'),loc='best')
plt.show()

plt.title('Real Rates and TIP Rates')
plt.plot(rate_data.index, rate_data.Rf,color='blue')
plt.plot(inflation_data.index, inflation_data.Rf,color='red')
plt.ylabel('Rate')
plt.xlabel('Time-to-maturity')
plt.gca().legend(('Real Rates','TIPS'),loc='best')
plt.show()



