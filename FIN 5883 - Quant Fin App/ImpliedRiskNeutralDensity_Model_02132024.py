#-----------------------------------------------------------------------------#
#Louis R. Piccotti
#Assistant Professor of Finance
#Director, M.S. Quantitative Financial Economics
#Spears School of Business
#Oklahoma State University
#This version: 03.04.2019
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#RND for the November 15, 2018 options maturity date.
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#Notes:1.Obtain the zero rate and the zero coupon bond price from running the
#        zero curve bootstrappig code.
#1.559000  0.997359 0.17260273972602739726027397260274
#0.0184 3221
#-----------------------------------------------------------------------------#

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
from scipy.stats import norm
import yfinance as yf

#----------------------------------------------------------------------------#
#Calculates zero curve, if selected
#----------------------------------------------------------------------------#

curve=input('Calculate zero-rate curve (y,n)? ')
if(curve=='y'):
    f1='H:/TreasuriesData/'+input('Bills file name:')+'.txt'
    f2='H:/TreasuriesData/'+input('Bonds file name:')+'.txt'
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
        if(bills.Asked[i]=='n.a.'):
            count=count+1
            if count>0:
                i=len(bills)
        i=i+1
    if(count>0):
        bills.drop(bills.index[bills.loc[bills.Asked=='n.a.'].index], inplace=True)
    i=0
    count=0
    while i<len(bonds):
        if(bonds.Asked[i]=='n.a.'):
            count=count+1
            if count>0:
                i=len(bonds)
        i=i+1
    if(count>0):
        bonds.drop(bonds.index[bonds.loc[bonds.Asked=='n.a.'].index], inplace=True)
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
    #Plots the bills zero-curve
    #-----------------------------------------------------------------------------#
    
    plt.plot(term.Maturity,term.Askedyield,color='blue')
    plt.plot(zeros.Maturity,zeros.Yield,color="red")
    plt.ylim(0.,4.)
    plt.title("Yield curve"+" ("+str(today.date())+")")
    plt.xlabel("Maturity date")
    plt.ylabel("Interest rate")
    plt.gca().legend(('coupon','bootstrapped zero'),loc='lower right')
    plt.show()
    
    plt.plot(bills.Maturity[1:len(bills)],bills.Price[1:len(bills)],
            bonds.Maturity[1:len(bonds)],bonds.ZeroPrice[1:len(bonds)],
            color="black")
    plt.title("Term structure of discount factors"+" ("+str(today.date())+")")
    plt.xlabel("Maturity date")
    plt.ylabel("Discount factor")
    plt.show()
    
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
    
    plt.plot(term.Maturity,term.Askedyield,color='blue')
    plt.plot(zeros.Maturity,zeros.Yield,color="red")
    plt.plot(zeros.Maturity,np.polyval(PlyFit,zeros.Ttm),color="green")
    plt.plot(zeros.Maturity,zeros.Fwrd,color="orange",linewidth=0.5)
    plt.ylim(0.,6.)
    plt.title("Yield curve"+" ("+str(today.date())+")")
    plt.xlabel("Maturity date")
    plt.ylabel("Interest rate")
    plt.gca().legend(('coupon term structure','bootstrapped zero rates',"P[F(t:T,T+3mo)], deg(P)=9","F(t,T,T+3mo)"),loc='upper left')
    plt.show()
	
    pd.options.display.max_rows=250
    print(zeros)


#----------------------------------------------------------------------------#
#Automated inputs.
#----------------------------------------------------------------------------#

today=datetime.datetime.today()
sym0=input('Ticker symbol (all caps):')
sym=yf.Ticker(sym0)
print(pd.DataFrame(sym.options,columns=['Mat']))
mat=pd.to_numeric(input('Maturity date (insert index number):'))
opmat=sym.options[mat]
indexlevel=((yf.Ticker(sym0)).history(period='1d'))['Close'][0]
ttm=((datetime.datetime.strptime(sym.options[mat],'%Y-%m-%d')-today).days+1)/365
r=zeros.Yield[abs(datetime.datetime.strptime(sym.options[mat],'%Y-%m-%d')\
                  -zeros.Maturity).idxmin()]/100
zb=zeros.Price[abs(datetime.datetime.strptime(sym.options[mat],'%Y-%m-%d')\
                       -zeros.Maturity).idxmin()]
if(sym.ticker!='^SPX'):
    divyld=sym.info['dividendYield']
if(sym.ticker=='^SPX'):
    divyld=yf.Ticker('SPY').info['yield']
if(sym.ticker=='^NDX'):
    divyld=yf.Ticker('QQQ').info['yield']

#----------------------------------------------------------------------------#
#Reads in the options data table and formats it.
#----------------------------------------------------------------------------#
    
calls=sym.option_chain(sym.options[mat]).calls #Selects the table of call options
calls['Moneyness']=calls.strike/indexlevel #Formats strikes in terms of moneyness
#Converts IV object to a IV numeric in decimal form.
calls['IV']=calls.impliedVolatility
calls=calls[calls['Moneyness']>0.20]
calls=calls[calls['IV']>0.10].reset_index()

plt.plot(calls.Moneyness[1:len(calls)],calls.IV[1:len(calls)],color='black',marker='o',fillstyle='none',linestyle='none')
plt.title("Raw Implied volatility ("+opmat+" call options)"+" ("+str(datetime.datetime.date(today))+")")
plt.xlabel("K/S")
plt.ylabel("Implied volatility")
plt.show()



#----------------------------------------------------------------------------#
#Plots call option implied volatilities
#----------------------------------------------------------------------------#

plt.plot(calls.Moneyness[1:len(calls)],calls.IV[1:len(calls)],color='black',marker='o',fillstyle='none',linestyle='none')
plt.title("Implied volatility ("+opmat+" call options)"+" ("+str(datetime.datetime.date(today))+")")
plt.xlabel("K/S")
plt.ylabel("Implied volatility")
plt.show()

#----------------------------------------------------------------------------#
#Fits spline regression to IV
#----------------------------------------------------------------------------#

yy=np.matrix(calls.IV).T
#3rd degree polynomial spline with 6 knots: 0.70,0.80,0.90,0.95,1.05,1.10
xx=np.concatenate((np.matrix(np.ones(len(calls))),np.matrix(calls.Moneyness),
    np.matrix(calls.Moneyness**2),np.matrix(calls.Moneyness**3),
    np.matrix(((((calls.Moneyness-0.90)+0+abs(0-(calls.Moneyness-0.90)))/2)**3)),
    np.matrix(((((calls.Moneyness-0.95)+0+abs(0-(calls.Moneyness-0.95)))/2)**3)),
    np.matrix(((((calls.Moneyness-1.05)+0+abs(0-(calls.Moneyness-1.05)))/2)**3)),
    np.matrix(((((calls.Moneyness-1.10)+0+abs(0-(calls.Moneyness-1.10)))/2)**3))),
    axis=0).T
beta=(xx.T*xx)**(-1)*xx.T*yy #OLS estimates of the coefficients
calls['IVfit']=xx*beta #Model-fit IV.

#Plots the IV model fit
plt.plot(calls.Moneyness,calls.IV,color='black',marker='o',fillstyle='none',linestyle='none')
plt.plot(calls.Moneyness,calls.IVfit,color='black')
plt.title("Implied volatility fit ("+opmat+" call options)"+" ("+str(datetime.datetime.date(today))+")")
plt.xlabel("K/S")
plt.ylabel("Implied volatility")
plt.show()

#Corrects for erroneous IVs
for i in range(1,len(calls['IV'])-1):
    if(i==1):
        if(calls['IV'][0]<calls['IV'][1]):
            calls['IV'][0]=calls['IV'][1]
    fh=calls['Moneyness'][i]-calls['Moneyness'][i-1]
    dIV=(calls['IVfit'][i+1]-calls['IVfit'][i-1])/fh
    if(abs(calls['IV'][i+1]-calls['IV'][i-1])<5):
        calls['IV'][i]=calls['IV'][i-1]+.5*(dIV*fh+(calls['IV'][i+1]-calls['IV'][i-1])/(2*fh)*fh)
    else:
        calls['IV'][i]=calls['IV'][i-1]+dIV*fh

yy=np.matrix(calls.IV).T
#3rd degree polynomial spline with 6 knots: 0.70,0.80,0.90,0.95,1.05,1.10
xx=np.concatenate((np.matrix(np.ones(len(calls))),np.matrix(calls.Moneyness),
    np.matrix(calls.Moneyness**2),np.matrix(calls.Moneyness**3),
    np.matrix(((((calls.Moneyness-0.90)+0+abs(0-(calls.Moneyness-0.90)))/2)**3)),
    np.matrix(((((calls.Moneyness-0.95)+0+abs(0-(calls.Moneyness-0.95)))/2)**3)),
    np.matrix(((((calls.Moneyness-1.05)+0+abs(0-(calls.Moneyness-1.05)))/2)**3)),
    np.matrix(((((calls.Moneyness-1.10)+0+abs(0-(calls.Moneyness-1.10)))/2)**3))),
    axis=0).T
beta=(xx.T*xx)**(-1)*xx.T*yy #OLS estimates of the coefficients
calls['IVfit']=xx*beta #Model-fit IV.

#Plots the IV model fit
plt.plot(calls.Moneyness,calls.IV,color='black',marker='o',fillstyle='none',linestyle='none')
plt.plot(calls.Moneyness,calls.IVfit,color='black')
plt.title("Implied volatility fit ("+opmat+" call options)"+" ("+str(datetime.datetime.date(today))+")")
plt.xlabel("K/S")
plt.ylabel("Implied volatility")
plt.show()

#----------------------------------------------------------------------------#
#Estimates the RND with the fitted IV spline model from above.
#----------------------------------------------------------------------------#

Mongrid=np.matrix(np.arange(np.min(calls.Moneyness),np.max(calls.Moneyness),0.0005)).T
Ivgrid=np.matrix(np.zeros(len(Mongrid))).T
#This loop fills in a grid of IVs implied by the spline model and moneyness grid
i=0
while i<len(Mongrid):
    Ivgrid[i,0]=beta[0,0]+Mongrid[i,0]*beta[1,0]+Mongrid[i,0]**2*beta[2,0]+Mongrid[i,0]**3*beta[3,0]+(((Mongrid[i,0]-0.90)+0+abs(0-(Mongrid[i,0]-0.90)))/2)**3*beta[4,0]+(((Mongrid[i,0]-0.95)+0+abs(0-(Mongrid[i,0]-0.95)))/2)**3*beta[5,0]+(((Mongrid[i,0]-1.05)+0+abs(0-(Mongrid[i,0]-1.05)))/2)**3*beta[6,0]+(((Mongrid[i,0]-1.10)+0+abs(0-(Mongrid[i,0]-1.10)))/2)**3*beta[7,0]
    i=i+1
bsd1=np.matrix(np.zeros(len(Mongrid))).T
bsd2=np.matrix(np.zeros(len(Mongrid))).T
#This loop calculates d1 for each moneyness
i=0
while i<len(Mongrid):
    bsd1[i,0]=(1/(Ivgrid[i,0]*np.sqrt(ttm)))*(np.log(indexlevel/(Mongrid[i,0]*indexlevel))+(r-divyld+Ivgrid[i,0]**2/2)*ttm)
    i=i+1
bsd2=bsd1-Ivgrid*np.sqrt(ttm) #d2
fwd=indexlevel/zb*np.exp((-divyld)*ttm) #Forward index level
#Black-Scholes option prices with continuously paid dividend yield
bsc=zb*(fwd*norm.cdf(bsd1)-np.multiply(norm.cdf(bsd2),Mongrid*indexlevel))
bsctab=np.concatenate((Mongrid*indexlevel,bsc),axis=1)

fdif=np.matrix(np.zeros(len(Mongrid))).T
sdif=np.matrix(np.zeros(len(Mongrid))).T
#This loop computes the 1st and 2nd derivatives via finite differences
i=1
while i<len(Mongrid):
    h=indexlevel*(Mongrid[i,0]-Mongrid[i-1,0])
    fdif[i,0]=(bsc[i,0]-bsc[i-1,0])/h
    if i>=2:
        sdif[i,0]=max((fdif[i,0]-fdif[i-1,0])/h,0)#Second derivative can't be negative
    i=i+1
#Estimates the RND and divides the by sum of all estimated probabilities so that
#the density sums to 1.
rnd=1/zb*sdif

#BS closed-form RND
#rnd=1/zb*norm.pdf(bsd2)/(np.sqrt(ttm)*np.multiply(Ivgrid,Mongrid*indexlevel))

#----------------------------------------------------------------------------#
#Numeric integration of the RND.

#Note:1.This should be approximately 1.  It will vary a bit due to numerical
#       errors and grid discreteness.
#     2.Alternatively, a monte-carlo integration could be used.
#----------------------------------------------------------------------------#

#Numeric integral (composite rule)
rndint=np.sum(rnd)*(np.max(Mongrid)-np.min(Mongrid))/(len(Mongrid)-1)*indexlevel
#Cumulative RND ( int_{min(Strike)}^{x} RND(u)du )
rncd=np.cumsum(rnd).T*(np.max(Mongrid)-np.min(Mongrid))/(len(Mongrid)-1)*indexlevel
np.set_printoptions(suppress=True)#Surpresses scientific notation
cdftab=np.concatenate((Mongrid*indexlevel,Mongrid,rncd),axis=1)
#This 2nd time through these variables adjust the PDF and CDF so that the PDF
#integrates to 1.
rnd=1/zb*sdif*(1/cdftab[len(cdftab)-1,2])
rndint=np.sum(rnd)*(np.max(Mongrid)-np.min(Mongrid))/(len(Mongrid)-1)*indexlevel
rncd=np.cumsum(rnd).T*(np.max(Mongrid)-np.min(Mongrid))/(len(Mongrid)-1)*indexlevel
np.set_printoptions(suppress=True)#Surpresses scientific notation
cdftab=np.concatenate((Mongrid*indexlevel,Mongrid,Ivgrid,rncd),axis=1) 

#----------------------------------------------------------------------------#
#Plots call option implied probability and cumulative density functions
#----------------------------------------------------------------------------#

plt.plot(Mongrid,rnd/rnd.sum(),color='black')
plt.title("Implied Risk-neutral pdf ("+opmat+" call options)"+" ("+str(datetime.datetime.date(today))+")")
plt.xlabel("K/S")
plt.ylabel("Density")
plt.show()

plt.plot(Mongrid,rncd,color='black')
plt.title("Implied Risk-neutral cdf ("+opmat+" call options)"+" ("+str(datetime.datetime.date(today))+")")
plt.xlabel("K/S")
plt.ylabel("Density")
plt.show()

import sys
np.set_printoptions(threshold=sys.maxsize)
print(cdftab)

mu=np.matrix(np.append(0,np.diff(rncd,n=1,axis=0)))@(Mongrid)
var=np.multiply(np.matrix(np.append(0,np.diff(rncd,n=1,axis=0))).T,np.multiply(Mongrid-mu,Mongrid-mu)).sum()
erLog=(var*zb)*365/(pd.to_datetime(opmat)-today).days
erLB=(2*var*zb)*365/(pd.to_datetime(opmat)-today).days
erUB=(4*var*zb)*365/(pd.to_datetime(opmat)-today).days
erUUB=(10*var*zb)*365/(pd.to_datetime(opmat)-today).days
print('\n\n--------------------------------'\
      '\nE{R-Rf}=RA*V{RND}*P(0,T)*(365/T)'\
      '\n--------------------------------')
print('\nLog utility: U=ln[W] (RA=1)'\
      '\n(set W=1)'\
      '\n---------------------------')
print('E{R-Rf} (RA=1): '+str(np.round(erLog,4)))
print('\nNegative exponential utility: U=exp{-RA*W}, where RA is risk-aversion'
      '\nparam. and W is wealth (set W=1)'\
      '\n---------------------------------------------------------------------')
print('\nE{R-Rf} (RA=2): '+str(np.round(erLB,4)))
print('E{R-Rf} (RA=4): '+str(np.round(erUB,4)))
print('E{R-Rf} (RA=10): '+str(np.round(erUUB,4)))
print('\nModel weighted average (0.10*Log+0.40*LB+0.40*UB+0.10*UUB)'\
      '\n----------------------------------------------------------'\
      '\n\nE{R-Rf}: '+\
      str(np.round(0.10*erLog+0.40*erLB+0.40*erUB+0.10*erUUB,4)))

k=input("press close to exit") 
