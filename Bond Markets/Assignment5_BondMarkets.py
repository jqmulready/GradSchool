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

f1='C:/Users/John Mulready/Desktop/John Mulready/Downloads/Spring 2024/Bond Markets/bills2024-03-05.txt'
f2='C:/Users/John Mulready/Desktop/John Mulready/Downloads/Spring 2024/Bond Markets/bonds2024-03-05.txt'
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


daily_rate_delta = 1 / 252 # trading days in the year approximetly
years_used = 10 # ten years of data as specified in the assignment
treasury_series = 'DGS3MO'

end = datetime.datetime(2024,3,5)
start=end-relativedelta(years=years_used)

print(f'Start Date as Datetime: {start}')
print(f'End Date as Datetime: {end}')

try:
    rate_data = web.DataReader(treasury_series,'fred',start,end).rename(columns={treasury_series:'Rf'})
    rate_data = rate_data.copy().ffill()
    if(rate_data['Rf'].isnull().iloc[0]):
        rate_data['Rf'].iloc[0]=rate_data['Rf'].iloc[1]
    rate_data['U1']=1
    rate_data
except:
    rate_data = pd.read_csv('rate_data.csv', index_col='DATE')

rate_data = rate_data.copy().ffill()
if(rate_data['Rf'].isnull().iloc[0]):
    rate_data['Rf'].iloc[0]=rate_data['Rf'].iloc[1]
rate_data['U1']=1
rate_data

initial_rate = rate_data.iloc[-1]['Rf'] / 100 # initial rate used in outr models
print(f'{treasury_series} on {rate_data.index[-1]} : {round(initial_rate*100,3)}%')

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
sigma = float(errs.std() / (daily_rate_delta ** 0.5))

k_value = float(( 1 - beta_values[1][0] ) / daily_rate_delta)
theta_value = float(beta_values[0][0] / ( k_value * daily_rate_delta ))

print(f'K value from linear regression: {k_value}')
print(f'Theta value from linear regression: {theta_value}')
print(f'Sigma value from linear regression: {sigma}')

delta = 1.0 / 12 # a 1 month delta so we can find times to maturities closest ot it

total_periods = int(years_used/delta)+1
zero_bond_matrix = np.zeros((total_periods, 2))
times_to_maturities = zero_bond_matrix[:,0]

maturity = delta

# make the first row the price of a zero bound expiring at the current date
zero_bond_matrix[0, 0] = 0
zero_bond_matrix[0, 1] = 1

for index in range(1, total_periods):
    
    zero_bond_matrix[index, 0] = maturity
    zero_bond_matrix[index, 1] = bond_polynomial_function(maturity)
    maturity += delta # increment the maturity
    

def calc_prices_hw(initial_rate, a_value, sigma, times_to_maturities):
    
    # we will treat the forward rate as the current 3 month rate

    forward_rate = initial_rate
    bond_prices = np.zeros(len(times_to_maturities))
    
    for index in range(0, len(times_to_maturities)):
        bond_prices[index] = bond_polynomial_function(times_to_maturities[index])
        
    # we will treat the forward rate as the current 3 month rate
    intitial_bond_price = zero_bond_matrix[0,1]

    # create an array of our B values
    B_array = (1 / a_value) * (1 - np.exp( - a_value * times_to_maturities ))

    # create an array of the zero coupong bond prices
    A_array = (bond_prices / intitial_bond_price) * np.exp(B_array * forward_rate - (sigma**2)/(4*a_value) * (1-np.exp(-2*a_value*times_to_maturities))*B_array**2) 
    
    # call our price function
    price_array=A_array * np.exp(-B_array * initial_rate)
    
    return price_array

def calibrate_hw_func(hw_params):
    
    a_value = abs(hw_params[0]) #Absolute value is used to make this an unconstrained opt. problem.
    
    # we will treat the forward rate as the current 3 month rate
    forward_rate = initial_rate

    # in our model, we want to start calibrating after the first 3 months
    times_to_maturities = zero_bond_matrix[int(delta*12*3):,0]
    bond_prices = zero_bond_matrix[int(delta*12*3):,1]
    intitial_bond_price = zero_bond_matrix[0,1]

    # create an array of our B values
    B_array = (1 / a_value) * (1 - np.exp( - a_value * times_to_maturities ))

    # create an array of the zero coupong bond prices
    A_array = (bond_prices / intitial_bond_price) * np.exp(B_array * forward_rate - (sigma**2)/(4*a_value) * (1-np.exp(-2*a_value*times_to_maturities))*B_array**2) 
    
    # call our price function
    price_array=A_array * np.exp(-B_array * initial_rate)
    
    # some of squared errors for minimization
    sse=np.sum((bond_prices-price_array)**2)
        
    return sse

hw_params_0=np.c_[k_value] #Starting values for a
res = minimize(calibrate_hw_func,hw_params_0.flatten(),method='L-BFGS-B') # call the optimization function
optimal_a_value = abs(res.x[0])

print(f'Optimal a value: {optimal_a_value}')
print(f'HW function value given a value: {calibrate_hw_func(np.c_[optimal_a_value].flatten())}')

optimal_a_value

delta = 1.0 / 12.0 # use a 1 month delta

# times to maturity and forward rates over a range of the yeas we're using
times_to_maturities_zeta=np.arange(0, years_used+delta/2.0, delta)
fitted_forward_rates=np.arange(0, years_used+delta/2.0, delta)

maturity = 0
for index in range(0,len(times_to_maturities_zeta)):
    times_to_maturities_zeta[index] = maturity
    # get the forward rates from the polynomial fitted curve of the forward rates
    maturity += delta
    
fitted_forward_rates=np.polyval(PlyFit,times_to_maturities_zeta)
fitted_forward_rates = fitted_forward_rates / 100.0
changes_in_forward_rates = np.zeros(len(fitted_forward_rates)-1)

for index in range(1, len(changes_in_forward_rates)):
    changes_in_forward_rates[index-1] = fitted_forward_rates[index] - fitted_forward_rates[index-1]

zeta_values = changes_in_forward_rates/delta + optimal_a_value * fitted_forward_rates[1:] + sigma ** 2 / (2 * optimal_a_value) * (1 - np.exp(-2 * optimal_a_value * times_to_maturities_zeta[1:]))

plt.title('Zeta Values divided by optimal a value')
plt.plot(times_to_maturities_zeta[1:],zeta_values/optimal_a_value,color='black')


plt.ylabel('Zeta/a')
plt.xlabel('Time-to-maturity')
plt.show()

plt.title('Calibrated forward rates')
plt.plot(times_to_maturities_zeta,fitted_forward_rates,color='black')

plt.ylabel('Forward rate')
plt.xlabel('Time-to-maturity')
plt.show()

delta_graphing = 1.0 / 12 # a 3 month delta so we can find times to maturities closest ot it

times_to_maturities_plotted=np.arange(0, years_used+delta_graphing, delta_graphing) # we're using this array to plot the modeled term structure

modeled_prices = calc_prices_hw(
    initial_rate=initial_rate, 
    a_value=optimal_a_value, 
    sigma=sigma, 
    times_to_maturities=times_to_maturities_plotted)

# the yields of the modeled prices assuming continuous compounding
modeled_yields = -1/times_to_maturities_plotted*np.log(modeled_prices)

graph_bond_matrix = np.zeros((len(times_to_maturities_plotted), 2))
maturity = delta_graphing

# # make the first row the price of a zero bound expiring at the current date
# graph_bond_matrix[0, 0] = 0
# graph_bond_matrix[0, 1] = 1

for index in range(0, len(graph_bond_matrix)):
    graph_bond_matrix[index, 0] = times_to_maturities_plotted[index]
    graph_bond_matrix[index, 1] = bond_polynomial_function(graph_bond_matrix[index, 0])
    maturity += delta_graphing # increment the maturity

# the yields of the actual bond prices assuming continuous compounding
graphed_times_to_maturities = graph_bond_matrix[:,0]

graphed_yields=-1/graphed_times_to_maturities*np.log(graph_bond_matrix[:, 1])


plt.title('Calibrated HW term structure of interest rates')
plt.plot(graphed_times_to_maturities,graphed_yields,color='black')
plt.plot(times_to_maturities_plotted,modeled_yields, \
        linestyle='none',marker='o',fillstyle='none', color='blue')

plt.ylabel('Interest rate')
plt.xlabel('Time-to-maturity')
plt.gca().legend(('True rate','Calibrated rate'),loc='best')
plt.show()

plt.title('Calibrated HW Modeled Bond Prices')
plt.plot(graph_bond_matrix[:, 0],graph_bond_matrix[:, 1],color='black')
plt.plot(times_to_maturities_plotted,modeled_prices, \
        linestyle='none',marker='o',fillstyle='none', color='blue')

plt.ylabel('Discount Factor')
plt.xlabel('Time-to-maturity')
plt.gca().legend(('True price','Calibrated price'),loc='best')
plt.show()