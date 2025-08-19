
'''
# ASSIGNMENT 6

(1) Calibrate the Vasicek volatility parameter to the prior 10 years (2024-02-15 minus 10 years up to 2024-02-15) of daily data for the 3-month constant maturity treasury rate (DGS3MO in FRED). 

(2) Using the calibrated volatility parameter from (1), calibrate k and theta to the polynomial-fitted values for the bootstrapped term structure of discount factors for times-to-maturity ranging from 3 month to 10 years. 

(3) Plot the quoted term structure of interest rates and the calibrated term structure of interest rates (assume continuous compounding) for times-to-maturity ranging from 3/12 to 10 by 1/12 step size.  For r(0), assume the 3-month constant maturity treasury rate on 2024-02-15. 

(4) Plot the shifted Vasicek term structure of interest rates.

(5) Plot the term structure of shift factors.

'''

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


#------------------------------------------------------------------------------
# I. Bootstrapps the term structure of discount factors
#------------------------------------------------------------------------------

#Global variabels
_forPer=1.0 #1.0 for 1-year forecast, 0.25 for 3-mo forecast, etc.
_polyO=9
#------------------------------------------------------------------------------

#----------------------------------------------------------------------------#
#Reads in the treasuries data table and formats it.
#----------------------------------------------------------------------------#

f1=r'C:\Users\John Mulready\Desktop\John Mulready\Downloads\Spring 2024\Bond Markets\bills2024-02-15.txt'
f2=r'C:\Users\John Mulready\Desktop\John Mulready\Downloads\Spring 2024\Bond Markets\bonds2024-02-15.txt'
#url="http://www.wsj.com/mdc/public/page/2_3020-treasury.html" (old)
#data=pd.read_html(url) #Reads in the data tables. (old)
bills=pd.read_csv(f1,sep='\t') #Selects the Treasury Bills table.
bonds=pd.read_csv(f2,sep='\t') #Selects the Treasury Notes/Bonds table.
bills=bills.rename(columns={'MATURITY':'Maturity','BID':'Bid','ASKED':'Asked','CHG':'Chg',
                            'ASKED YIELD':'Askedyield'})
bonds=bonds.rename(columns={'MATURITY':'Maturity','COUPON':'Coupon','BID':'Bid',
                            'ASKED':'Asked','CHG':'Chg','ASKED YIELD':'Askedyield'})
#drops the rows where the bond or bill price has n.a.
i=0
count=0
while i<len(bills):
    if(bills.Asked[i]=='n.a.' or bills.Asked[i]=='n.a'):
        count=count+1
        if count>0:
            i=len(bills)
    i=i+1
if(count>0):
    bills.drop(bills.index[bills.loc[bills.Asked=='n.a.'].index], inplace=True)
    bills.drop(bills.index[bills.loc[bills.Asked=='n.a'].index], inplace=True)
i=0
count=0
while i<len(bonds):
    if(bonds.Asked[i]=='n.a.' or bonds.Asked[i]=='n.a'):
        count=count+1
        if count>0:
            i=len(bonds)
    i=i+1
if(count>0):
    bonds.drop(bonds.index[bonds.loc[bonds.Asked=='n.a.'].index], inplace=True)
    bonds.drop(bonds.index[bonds.loc[bonds.Asked=='n.a'].index], inplace=True)
#Adjusts the bond price since the new data has the decimal as the fraction of 32.
bonds.Asked=divmod(pd.to_numeric(bonds.Asked),1)[0]+divmod(pd.to_numeric(bonds.Asked),1)[1]*100/32
#bills.columns=bills.iloc[0] #Renames the bills header to be the names in row 0. (old)
#bonds.columns=bonds.iloc[0] #Renames the bonds header to be the names in row 0. (old)
#drops the redundant header in row 0.
#bills=bills.drop([0]) (old)
#bonds=bonds.drop([0]) (old)
#Identifies today's date.
#f = requests.get(url) #Reads in the url info (old)
#soup=bs.BeautifulSoup(f.text,'html.parser') #parses the url into text (old)
#l=soup.find("div",{"class":"tbltime"}) #Finds the location of the table date (old)
#for span in l.findAll('span'): #Keeps only the last obs in this class, which is the date (old)
#    date=span.text (old)
#dt=parse(date) #Converts the date format. (old)
dt=f1[len(f1)-14:len(f1)-4]
#today=pd.to_datetime(dt.strftime('%Y-%m-%d')) #Makes the date a datetime variable in the format we want. (old)
today=pd.to_datetime(dt)
#bills.to_csv("bills"+str(today.date())+".csv")  #Saves bills data to csv on computer
#bonds.to_csv("bonds"+str(today.date())+".csv") #Saves bonds data to csv on computer
yrlen=365 #The number of days assumed to be in 1 year.
#converts bonds and bills maturity dates to datetime values.
bonds.Maturity=pd.to_datetime(bonds.Maturity)
bills.Maturity=pd.to_datetime(bills.Maturity)
#Converts bonds and bills asked yields to numeric values
bonds.Askedyield=pd.to_numeric(bonds.Askedyield)
bills.Askedyield=pd.to_numeric(bills.Askedyield)
#Keeps only bonds from the bonds table with a maturity of >1-year from today.
bonds=bonds[(bonds.Maturity-datetime.timedelta(yrlen))>today]
 #Keeps only the first maturity date, when there are multiple obs with the same
 #maturity date.
bonds=bonds[bonds.Maturity != bonds.Maturity.shift(1)]
bills=bills[bills.Maturity != bills.Maturity.shift(1)]
bonds.index=np.arange(1,len(bonds)+1) #Resets the new obs. index to start with 1.
bills.index=np.arange(1,len(bills)+1) #Resets the new obs. index to start with 1.
#Calculates the time-to-maturity in years
bills['Ttm']=pd.to_numeric((bills.Maturity-today)/datetime.timedelta(yrlen))
bonds['Ttm']=pd.to_numeric((bonds.Maturity-today)/datetime.timedelta(yrlen))
bills['Price']=1./(1.+(bills.Askedyield/100)*bills.Ttm) #Treasury bill prices.

#-----------------------------------------------------------------------------#
#Bootstraps the zero-curve for bonds observations. Semi-annual coupon payments
#is used with the coupon payments on
#-----------------------------------------------------------------------------#

#Sets the ask price for the coupon bond, which all the coupons will be stripped
#from to attain the final zero price as the result.
bonds['ZeroPrice']=pd.to_numeric(bonds.Asked)/100 #sets the quoted price
bonds.Coupon=pd.to_numeric(bonds.Coupon) #Makes the coupons a numeric vlaue.
i=1 #Sets the bond index counting variable to the first obs.
while i<=len(bonds): #Iterates over all the bonds
    
    #Strips the coupons from the quoted bond Asks.
    s=np.floor(pd.to_numeric((bonds.Maturity[i]-today)/datetime.timedelta(yrlen))*2)
    while ((bonds.Maturity[i]-dateutil.relativedelta.relativedelta(months=s*6)>today) & (bonds.Maturity[i]-dateutil.relativedelta.relativedelta(months=s*6)<bonds.Maturity[i])):
        #Calculates the coupon date.
        cpndate=bonds.Maturity[i]-dateutil.relativedelta.relativedelta(months=s*6)
        #calculates the absolute difference between the coupon date and all 
        #available zero-coupon bills maturity dates.
        if pd.to_numeric((cpndate-today)/datetime.timedelta(yrlen))<1:
            absdif=abs(bills.Maturity-cpndate)
            df=bills.Price[absdif.idxmin()]
        else:
            absdif=abs(bonds.Maturity-cpndate)
            df=bonds.ZeroPrice[absdif.idxmin()]
        #Strips the coupon, using the bill with the closest maturity date to
        #the coupon date
        if s==np.floor(pd.to_numeric((bonds.Maturity[i]-today)/datetime.timedelta(yrlen))*2):
            #Adds accrued interest to the published "clean" price.
            bonds.ZeroPrice[i]=bonds.ZeroPrice[i]+((bonds.Coupon[i]/100)/2)*(1-pd.to_numeric((cpndate-today)/datetime.timedelta(30*6)))
        bonds.ZeroPrice[i]=bonds.ZeroPrice[i]-((bonds.Coupon[i]/100)/2)*df
        s=s-1
    bonds.ZeroPrice[i]=bonds.ZeroPrice[i]/(1+((bonds.Coupon[i]/100)/2))
    #This if statement corrects for numerical errors resulting in large jumps
    #in the zerio yield.
    if i>1 and (bonds.ZeroPrice[i]/bonds.ZeroPrice[i-1]-1)>0.01:
        bonds.ZeroPrice[i]=1/((1+1/(bonds.ZeroPrice[i-1]**(1/bonds.Ttm[i-1]))-1)**bonds.Ttm[i])
    i=i+1
#Calculates the yield implied by the coupon bond's bootstrapped zero-coupon
#price.
bonds['ZeroYield']=(1/(bonds.ZeroPrice**(1/bonds.Ttm))-1)*100

#-----------------------------------------------------------------------------#
#Appends the term-structure (using coupon bonds).
#-----------------------------------------------------------------------------#
term=pd.DataFrame((bills.Askedyield)._append(bonds.Askedyield))
term['Maturity']=(bills.Maturity)._append(bonds.Maturity)
term.index=np.arange(1,len(term)+1)

#-----------------------------------------------------------------------------#
#Appends the zero curves
#-----------------------------------------------------------------------------#

zeros=pd.DataFrame((bills.Askedyield)._append(bonds.ZeroYield))
zeros.columns=['Yield']
zeros['Price']=(bills.Price)._append(bonds.ZeroPrice)
zeros['Maturity']=(bills.Maturity)._append(bonds.Maturity)
zeros.index=np.arange(1,len(zeros)+1)
#Constructs a 12-month rolling centered moving average of yields.
zeros['MA']=zeros.Yield.rolling(window=12,center=True,min_periods=0).mean()


#-----------------------------------------------------------------------------#
#Forward curve. F(t;T,T+3 months)
#Linearly interpolated
#-----------------------------------------------------------------------------#
zeros["Fwrd"]=zeros.Yield
i=1
while(i<=len(zeros)-1):
    ft=zeros.Maturity[i]
    fs=zeros.Maturity[i]+dateutil.relativedelta.relativedelta(months=3)
    tau=pd.to_numeric((fs-ft)/datetime.timedelta(yrlen))
    dif=pd.to_numeric(zeros.Maturity-fs)
    absdifs=abs(zeros.Maturity-fs)
    sgn=np.sign(dif[absdifs.idxmin()])
    if sgn==-1:
        ps=zeros.Price[absdifs.idxmin()]+(fs-zeros.Maturity[absdifs.idxmin()])/(zeros.Maturity[absdifs.idxmin()+1]-zeros.Maturity[absdifs.idxmin()])*(zeros.Price[absdifs.idxmin()+1]-zeros.Price[absdifs.idxmin()])
    if sgn==1:
        ps=zeros.Price[absdifs.idxmin()-1]+(fs-zeros.Maturity[absdifs.idxmin()-1])/(zeros.Maturity[absdifs.idxmin()]-zeros.Maturity[absdifs.idxmin()-1])*(zeros.Price[absdifs.idxmin()]-zeros.Price[absdifs.idxmin()-1])
    if sgn==0:
        ps=zeros.Price[absdifs.idxmin()]
    zeros.Fwrd[i]=(1/tau)*(zeros.Price[i]/ps-1)*100
    if i==len(zeros)-1:
        zeros.Fwrd[i+1]=zeros.Fwrd[i]
    i=i+1

zeros['FwrdMA']=zeros.Fwrd.rolling(window=6,center=True,min_periods=0).mean() #Rolling centered moving average
zeros['Ttm']=pd.to_numeric((zeros.Maturity-today)/datetime.timedelta(yrlen)) #Time to maturity in years
PlyFit=np.polyfit(zeros.Ttm,zeros.Fwrd,9) #Polynomial deg(9) fit to TTM.
zeros['PlyFit']=np.polyval(PlyFit,zeros.Ttm)

#Fits the discount factor
yy=zeros['Price']
xx=zeros['Ttm']**0
for i in range(1,_polyO+1):
    xx=np.c_[xx,zeros['Ttm']**i]
beta=np.linalg.inv(xx.T@xx)@xx.T@yy

betaM=beta*1000.0



#Defines discount factor price function
def bond_polynomial_function(T):
    Tvec=1
    for i in range(1,_polyO+1):
        Tvec=np.c_[Tvec,T**i]
    ds=abs(zeros.Ttm-T)
    if(T<2.0):
        ret=zeros.Price[ds.idxmin()]
    else:
        ret=(Tvec@beta)[0]
    return ret

# plt.plot(term.Maturity,term.Askedyield,color='blue')
# plt.plot(zeros.Maturity,zeros.Yield,color="red")
# plt.plot(zeros.Maturity,np.polyval(PlyFit,zeros.Ttm),color="green")
# plt.plot(zeros.Maturity,zeros.FwrdMA,color="orange",linewidth=0.5)
# plt.ylim(-3.,10.)
# plt.title("Forward rate curve"+" ("+str(today.date())+")")
# plt.xlabel("Maturity date")
# plt.ylabel("Interest rate")
# plt.gca().legend(('coupon term structure','bootstrapped zero rates','P[F(t:T,T+3mo)], deg(P)=9',\
#                   'F(t,T,T+3mo), centered MA(6)'),loc='best')
# plt.show()

# Create a data range to pull our interest rate data so we can determine the volatility

daily_rate_delta = 1 / 12 # trading days in the year approximetly
years_used = 10 # ten years of data as specified in the assignment
treasury_series = 'REAINTRATREARAT10Y'

end = datetime.datetime(2024,4,25)
start=end-relativedelta(years=years_used)

print('Rate data dates:')
print(f'Start date as Datetime: {start}')
print(f'End date as Datetime: {end}')

rate_data = web.DataReader(treasury_series,'fred',start,end).rename(columns={treasury_series:'Rf'})
rate_data = rate_data.copy().ffill()
rate_data['Rf'] = rate_data['Rf'] / 100.0
if(rate_data['Rf'].isnull().iloc[0]):
    rate_data['Rf'].iloc[0]=rate_data['Rf'].iloc[1]
rate_data['U1']=1
rate_data

initial_rate = rate_data.iloc[-1]['Rf'] # initial rate used in outr models
print(f'{treasury_series} Rate on 04/25/2024: {round(initial_rate*100,3)}%')

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

print(f'K Value from linear regression: {k_value}')
print(f'Theta value from linear regression: {theta_value}')
print(f'Volatility from linear regression: {volatility}')


delta = 3.0 / 12 # a 3 month delta so we can find times to maturities closest ot it

total_periods = int(years_used/delta)
times_to_maturities = np.zeros((total_periods, 1))
market_bond_prices = np.zeros((total_periods, 1))

maturity = delta # set the first maturity equal to the delta since we're starting after 1 month

# make the first row the price of a zero bound expiring at the current date

for index in range(0, total_periods):

    times_to_maturities[index, 0] = maturity
    market_bond_prices[index, 0] = bond_polynomial_function(maturity)
    maturity += delta # increment the maturity
    
def calibrate_vasicek_func(vasicek_params):
    
    k_value = abs(vasicek_params[0]) #Absolute value is used to make this an unconstrained opt. problem.
    theta_value = abs(vasicek_params[1]) #Absolute value is used to make this an unconstrained opt. problem.
    
    # create an array of our B values
    B_array = 1 / k_value * (1-np.exp(-k_value * times_to_maturities))
    
    # create an array of our A values
    A_array=np.exp((theta_value-volatility**2/(2*k_value**2))*(B_array-times_to_maturities)-volatility**2/(4*k_value)*B_array**2)
    
    # create an array of our prices based on the parameters of the model
    price_array=A_array * np.exp(-B_array*initial_rate)
    
    # some of squared errors for minimization
    sse=np.sum((market_bond_prices - price_array)**2)
    
    return sse

vasicek_params_0=np.c_[0.01,0.01] #Starting values for k and theta
res = minimize(calibrate_vasicek_func,vasicek_params_0.flatten(),method='L-BFGS-B') # call the optimization function

optimal_k_value = abs(res.x[0])
optimal_theta_value = abs(res.x[1])

print(f'Optimal k value: {optimal_k_value}')
print(f'Optimal theta value: {optimal_theta_value}')
print(f'Vasicek function value given k and theta values: {calibrate_vasicek_func(np.c_[optimal_k_value, optimal_theta_value].flatten())}')

def calc_prices_vasicek(initial_rate, k_value, theta_value, volatility, times_to_maturities):
    '''
    Returns and array of calculated CIR Model Zero Bond Prices
    
    '''

    # create an array of our B values
    B_array = 1 / k_value * (1-np.exp(-k_value * times_to_maturities))
    
    # create an array of our A values
    A_array=np.exp((theta_value-volatility**2/(2*k_value**2))*(B_array-times_to_maturities)-volatility**2/(4*k_value)*B_array**2)
    
    # create an array of our prices based on the parameters of the model
    price_array=A_array * np.exp(-B_array*initial_rate)
    
    return price_array

delta_graphing = 1.0 / 12 # a 1 month delta so we can find times to maturities closest ot it
starting_maturity = 3.0 / 12 # starting at 3 months

times_to_maturities_plotted=np.c_[np.arange(starting_maturity,years_used+delta_graphing/2,delta_graphing)] # we're using this array to plot the modeled term structure
market_prices = np.zeros((1,len(times_to_maturities_plotted)))

for index in range(0, len(times_to_maturities_plotted)):
    
    # pull the market prices of the bonds using the polynomial function
    market_prices[0, index] = bond_polynomial_function(times_to_maturities_plotted[index])

market_prices = market_prices.T # transpose it so we can use it for our operations

modeled_prices = calc_prices_vasicek(
    initial_rate=initial_rate, 
    k_value=optimal_k_value, 
    theta_value=optimal_theta_value, 
    volatility=volatility, 
    times_to_maturities=times_to_maturities_plotted)

# the yields of the prices assuming continuous compounding
modeled_yields = -1/times_to_maturities_plotted*np.log(modeled_prices)
market_yields =  -1/times_to_maturities_plotted*np.log(market_prices)
