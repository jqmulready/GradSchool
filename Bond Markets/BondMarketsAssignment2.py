# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:41:30 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import datetime as dt

# Specify the file path
csv_file_path = r"C:\Users\John Mulready\Desktop\John Mulready\Downloads\Spring 2024\Bond Markets\zeros_df.csv"
# Read the CSV file into a DataFrame
zeros_df = pd.read_csv(csv_file_path)

# Desired TTM values
desired_ttm_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# Calculate the absolute difference between each TTM value and the desired TTM values
for desired_ttm in desired_ttm_values:
    zeros_df['Abs_Diff_' + str(desired_ttm)] = abs(zeros_df['Ttm'] - desired_ttm)

# Find the row with the minimum absolute difference for each desired TTM value
closest_prices = {}
for desired_ttm in desired_ttm_values:
    closest_row_index = zeros_df['Abs_Diff_' + str(desired_ttm)].idxmin()
    closest_prices[desired_ttm] = zeros_df.loc[closest_row_index, 'Price']

print("Closest prices to TTM values:")
print(closest_prices)

#Store your matrix of ttm and zero-coupon bond prices as tab1.
#Table from slide 2
#                 ttm , P(i)
tab1=np.matrix(list(closest_prices.items()))


q=0.5 #R-N probability of up move in Ho and Lee model
delta=tab1[0,0] #Time step
r0=-(1/delta)*np.log(tab1[0,1]) #Starting delta-period interest rate
lv=0.5317 #Lognormal interest rate volatility
sigma=.0078 #Normal volatility
N=tab1.shape[0]
rTree=np.zeros((N,N)) #Empty rate tree
rTree[0,0]=r0

thetavec=np.zeros((tab1.shape[0],1))

def bondTree(theta):
    pTree=np.zeros((int(tab1[i,0]/delta+1),int(tab1[i,0]/delta+1)))
    pTree[:,pTree.shape[1]-1]=1 #-1 because python starts indexing at 0.
    K=pTree.shape[1]-1
    for k in range(K-1,0,-1): #loops backwards from K to 1.
        for m in range(0,k+1,1): #loops from row 0 (1st row) to row k
            if(m<k):
                r=rTree[m,k-1]+theta*delta+sigma*np.sqrt(delta) #ru in sub 1-period tree
                pTree[m,k]=np.exp(-r*delta)*(q*pTree[m,k+1]+(1-q)*pTree[m+1,k+1]) #p{i}=exp[-r*delta]*E{p(i+1)}
            if(m==k):
                r=rTree[m-1,k-1]+theta*delta-sigma*np.sqrt(delta) #rd in sub 1-period tree
            pTree[m,k]=np.exp(-r*delta)*(q*pTree[m,k+1]+(1-q)*pTree[m+1,k+1]) #p{i}=exp[-r*delta]*E{p(i+1)}
    pTree[0,0]=np.exp(-r0*delta)*(q*pTree[0,1]+(1-q)*pTree[1,1]) #model-implied p0
    esq=(tab1[i,1]-pTree[0,0])**2 #model squared error compared to bond price in table observed in market
    return esq

i=1
while(i<N): 
    theta0=0 #Starting value for theta
    res = minimize(bondTree,theta0) #minimization function
    thetavec[i,0]=res.x
    for m in range(0,i+1,1): #Fills in the newly found interest rates into the rate tree
        if(m<i):
            rTree[m,i]=rTree[m,i-1]+thetavec[i,0]*delta+sigma*np.sqrt(delta)
        if(m==i):
            rTree[m,i]=rTree[m,i-1]+thetavec[i,0]*delta+sigma*np.sqrt(delta)
    i=i+1
    
print('Calibrated interest rate tree:\n')
print('t')
print(tab1[:,0].T)
print('\nTheta:')
print(thetavec.T)
print('\nRate:')
print(rTree)
