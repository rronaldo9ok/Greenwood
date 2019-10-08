# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:47:28 2019

@author: Rony
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import statsmodels.formula.api as smf
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import YearEnd
F_score = 8

stocks = pd.read_csv('computestat_infomation.csv')
stocklist = list(stocks['tic'])
stocks.set_index(['tic'], inplace=True)
stocks['TICKER'] = stocks.index
stocklist = list(dict.fromkeys(stocklist))
market = pd.read_csv('stocks_information.csv')

List_14 = [];
List_15 = [];
List_16 = [];
List_17 = [];
List_18 = [];


for x in stocklist:
    temp = stocks.loc[x]
    temp['IB_score'] = np.where(temp['IB']>0, 1, 0) #first score (IB)
    temp['CFO_score'] = np.where(temp['CFO']>0,1,0) #second score  (CFO)      
    temp['IB_change_score'] = np.where(temp['IB'].pct_change()>0, 1, 0)  #third score (Percentage change of IB)
    temp['Manipulation_score'] = np.where(temp['CFO'] - temp['IB'] > 0 , 1, 0) #fourth score (Manipulation)
    temp['Leverage_ratio_score'] = np.where((temp['TD']/temp['AT']).pct_change() < 0, 1, 0) #fifth score(percentage change of leverage ratio)
    temp['current_ratio_score'] = np.where((temp['ACT']/temp['CL']).pct_change() < 0, 1, 0) #sixth score(percentage change of current ratio)
    temp['Share_change_score'] = np.where(temp['SHARE'].pct_change()>0, 1, 0)  #seventh score (Percentage change of outstanding share)
    temp['Gross_Margin_change_score'] = np.where((temp['REVT']-temp['COGS']/temp['REVT']).pct_change()>0, 1, 0)  #third score (Percentage change of IB)
    temp['Asset_Turnover_score'] = np.where((temp['REVT']/temp['AT']).pct_change()>0, 1, 0)  #seventh score (Percentage change of outstanding share)
    temp['Total_score'] = temp['IB_score'] + temp['CFO_score']+temp['IB_change_score']+temp['Manipulation_score']+temp['Leverage_ratio_score']+temp['current_ratio_score']+temp['Share_change_score']+ temp['Gross_Margin_change_score'] + temp['Asset_Turnover_score']
    temp.set_index(['fyear'], inplace = True)
    try:
        if temp.loc[2013,'Total_score'] >= F_score:
            List_14.append(x) 
        if temp.loc[2014, 'Total_score'] >= F_score:
            List_15.append(x)
        if temp.loc[2015,'Total_score'] >=  F_score:
            List_16.append(x) 
        if temp.loc[2016, 'Total_score'] >= F_score:
            List_17.append(x)
        if temp.loc[2017, 'Total_score'] >= F_score:
            List_18.append(x)
    except:
        pass
 
    
#The following is doing the regression considering the industry    
#stocks = stocks[stocks.fyear < 2014] The following step we add one more row as 2018 to
#each stock so that when we pad the stat by one year, the stat from 2017 will flows to 2018
def add_row(x):
             last_row = x.iloc[-1]
             last_row['fyear'] = 2018
             return x.append(last_row)
             
stocks = stocks.groupby(['TICKER']).apply(add_row).reset_index(drop=True)        
    
    
    
stocks['ROE'] = stocks['NI']/(stocks['AT'] - stocks['TD'])
stocks['AT'] = stocks.groupby(['TICKER'])['AT'].shift(1)
stocks['TD'] = stocks.groupby(['TICKER'])['TD'].shift(1)
stocks['Leverage_ratio_pad'] = stocks['TD']/stocks['AT']
stocks['NI'] = stocks.groupby(['TICKER'])['NI'].shift(1)
stocks['REVT'] = stocks.groupby(['TICKER'])['REVT'].shift(1)
stocks['Profit_Margin_pad'] = stocks['NI']/stocks['REVT']
stocks['Asset_Turnover_pad'] = stocks['NI']/stocks['AT']
stocks['C&E'] = stocks.groupby(['TICKER'])['C&E'].shift(1)
stocks['CL'] = stocks.groupby(['TICKER'])['CL'].shift(1)
stocks['Cash_ratio_pad'] = stocks['C&E']/stocks['CL']
stocks = stocks.drop(['ACT','AT','C&E','COGS','SHARE','TD','IB','CL','NI','CFO','REVT'],axis = 1)
stocks = stocks.dropna(axis=0)
stocks.set_index('TICKER', inplace = True)

#dt used for the data for regression model build up
#pd used for the data for prediction
dt_14 = stocks.loc[List_14]
pd_14 = dt_14[dt_14.fyear == 2014]
dt_14 = dt_14[dt_14.fyear < 2014]

dt_15 = stocks.loc[List_15]
pd_15 = dt_15[dt_15.fyear == 2015]
dt_15 = dt_15[dt_15.fyear < 2015]

dt_16 = stocks.loc[List_16]
pd_16 = dt_16[dt_16.fyear == 2016]
dt_16 = dt_16[dt_16.fyear < 2016]

dt_17 = stocks.loc[List_17]
pd_17 = dt_17[dt_17.fyear == 2017]
dt_17 = dt_17[dt_17.fyear < 2017]


dt_18 = stocks.loc[List_18]
pd_18 = dt_18[dt_18.fyear == 2018]
dt_18 = dt_18[dt_18.fyear < 2018]





def regfun(df, mod):
    results = smf.ols(mod, data=df).fit()
    return results.params
    
regfun(dt_14, 'ROE ~ Leverage_ratio_pad + Profit_Margin_pad + Asset_Turnover_pad + Cash_ratio_pad')

allparams14 = dt_14.groupby('fyear').apply(regfun, 'ROE ~ Leverage_ratio_pad + Profit_Margin_pad + Asset_Turnover_pad + Cash_ratio_pad')
allparams15 = dt_15.groupby('fyear').apply(regfun, 'ROE ~ Leverage_ratio_pad + Profit_Margin_pad + Asset_Turnover_pad + Cash_ratio_pad')
allparams16 = dt_16.groupby('fyear').apply(regfun, 'ROE ~ Leverage_ratio_pad + Profit_Margin_pad + Asset_Turnover_pad + Cash_ratio_pad')
allparams17 = dt_17.groupby('fyear').apply(regfun, 'ROE ~ Leverage_ratio_pad + Profit_Margin_pad + Asset_Turnover_pad + Cash_ratio_pad')
allparams18 = dt_18.groupby('fyear').apply(regfun, 'ROE ~ Leverage_ratio_pad + Profit_Margin_pad + Asset_Turnover_pad + Cash_ratio_pad')

#stats = allparams.describe()
#stats.loc['stderr'] = stats.loc['std'] / np.sqrt(stats.loc['count'])
#stats.loc['tstat']  = stats.loc['mean'] / stats.loc['stderr']

allparams14 = allparams14.mean()
allparams15 = allparams15.mean()
allparams16 = allparams16.mean()
allparams17 = allparams17.mean()
allparams18 = allparams18.mean()

pd_14['ROE_predict'] = allparams14.Intercept + allparams14.Leverage_ratio_pad * pd_14['Leverage_ratio_pad'] + allparams14.Profit_Margin_pad * pd_14['Profit_Margin_pad'] + allparams14.Asset_Turnover_pad * pd_14['Asset_Turnover_pad'] + allparams14.Cash_ratio_pad * pd_14['Cash_ratio_pad']
pd_15['ROE_predict'] = allparams15.Intercept + allparams15.Leverage_ratio_pad * pd_15['Leverage_ratio_pad'] + allparams15.Profit_Margin_pad * pd_15['Profit_Margin_pad'] + allparams15.Asset_Turnover_pad * pd_15['Asset_Turnover_pad'] + allparams15.Cash_ratio_pad * pd_15['Cash_ratio_pad']
pd_16['ROE_predict'] = allparams16.Intercept + allparams16.Leverage_ratio_pad * pd_16['Leverage_ratio_pad'] + allparams16.Profit_Margin_pad * pd_16['Profit_Margin_pad'] + allparams16.Asset_Turnover_pad * pd_16['Asset_Turnover_pad'] + allparams16.Cash_ratio_pad * pd_16['Cash_ratio_pad']
pd_17['ROE_predict'] = allparams17.Intercept + allparams17.Leverage_ratio_pad * pd_17['Leverage_ratio_pad'] + allparams17.Profit_Margin_pad * pd_17['Profit_Margin_pad'] + allparams17.Asset_Turnover_pad * pd_17['Asset_Turnover_pad'] + allparams17.Cash_ratio_pad * pd_17['Cash_ratio_pad']
pd_18['ROE_predict'] = allparams18.Intercept + allparams18.Leverage_ratio_pad * pd_18['Leverage_ratio_pad'] + allparams18.Profit_Margin_pad * pd_18['Profit_Margin_pad'] + allparams18.Asset_Turnover_pad * pd_18['Asset_Turnover_pad'] + allparams18.Cash_ratio_pad * pd_18['Cash_ratio_pad']


pd_14['QUARTILE'] = pd.qcut(pd_14['ROE_predict'], 4, labels=False)
pd_15['QUARTILE'] = pd.qcut(pd_15['ROE_predict'], 4, labels=False)
pd_16['QUARTILE'] = pd.qcut(pd_16['ROE_predict'], 4, labels=False)
pd_17['QUARTILE'] = pd.qcut(pd_17['ROE_predict'], 4, labels=False)
pd_18['QUARTILE'] = pd.qcut(pd_18['ROE_predict'], 4, labels=False)

invest_list_14 = pd_14.index[(pd_14.QUARTILE == 3) ]
invest_list_15 = pd_15.index[(pd_15.QUARTILE == 3)  ]
invest_list_16 = pd_16.index[(pd_16.QUARTILE == 3)  ]
invest_list_17 = pd_17.index[(pd_17.QUARTILE == 3)  ]
invest_list_18 = pd_18.index[(pd_18.QUARTILE == 3)  ]




market.set_index(['Ticker'], inplace = True)
market['Returns'] = market['Returns'].convert_objects(convert_numeric = True)
market['Gro'] = market['Returns'] + 1


def calculate_return(invest_list, market, year):
    mk = market.loc[invest_list]
    mk['Date'] = mk['Date']/100
    mk = mk[mk['Date']/100 > year]
    mk = mk[mk['Date']/100 < (year + 1)]
    return (mk.groupby('Ticker')['Gro'].prod()).mean() - 1


Performance = pd.DataFrame(columns = ['Returns'])

F_score_only_Performance = pd.DataFrame(columns = ['Returns'])
F_score_only_Performance.loc['2014'] = calculate_return(List_14, market, 2014);
F_score_only_Performance.loc['2015'] = calculate_return(List_14, market, 2015);
F_score_only_Performance.loc['2016'] = calculate_return(List_14, market, 2016);
F_score_only_Performance.loc['2017'] = calculate_return(List_14, market, 2017);
F_score_only_Performance.loc['2018'] = calculate_return(List_14, market, 2018);



Performance.loc['2014'] = calculate_return(invest_list_14, market, 2014);
Performance.loc['2015'] = calculate_return(invest_list_15, market, 2015);
Performance.loc['2016'] = calculate_return(invest_list_16, market, 2016);
Performance.loc['2017'] = calculate_return(invest_list_17, market, 2017);
Performance.loc['2018'] = calculate_return(invest_list_18, market, 2018);



Performance['Gro'] = Performance['Returns'] + 1
five_year_gro = Performance['Gro'].prod()  - 1


Performance['Cumulative'] = 0
for i in range(len(Performance.index)):
    Performance.iloc[i,2]= Performance[0:i+1]['Gro'].prod() - 1
    
#input the data of spx return
Performance['SPX_return'] = 0
Performance.iloc[0,3] = 0.1139;
Performance.iloc[1,3] = -0.73/100;
Performance.iloc[2,3] = 0.0954
Performance.iloc[3,3] = 0.1942
Performance.iloc[4,3] = -0.0624
Performance['outperformance'] = Performance['Returns'] - Performance['SPX_return']
F_score_only_Performance['SPX_return'] = Performance['SPX_return']


X = Performance[['Returns']]
Y = Performance['SPX_return']
rho = X.corrwith(Y,axis = 0)
gap = Performance['outperformance'].mean()

#If I invest 100million dollar following this strategy, the portfolio will be worth as following after 5 year 
print("if you invest 100 million at the begining following the strategy, you will get",100*(1+five_year_gro), "million")


plt.plot(Performance.index, Performance['Returns'] )
plt.plot(Performance.index, Performance['SPX_return'] )
plt.legend()
plt.show()

plt.plot(Performance.index, Performance['outperformance'])
plt.show()    
    
arithmetic_return = Performance['Returns'].mean()
geometric_return = (1 + five_year_gro)**(0.2) - 1

print("The arithmetic return is ",arithmetic_return)
print("The geometric return is",geometric_return)









    
    
'''    
#The following is doing the regression for each firm independently    
ROE_List_14 = pd.DataFrame({'Stock':List_14.copy(), 'ROE_prediction':0.});
ROE_List_14.set_index(['Stock'], inplace = True)   
for x in List_14:
    temp = stocks.loc[x]    
    temp['Leverage_ratio'] = temp['TD']/temp['AT'].shift(-1)
    temp['Profit_Margin'] = temp['NI']/temp['REVT'].shift(-1)
    temp['Asset_Turnover'] = temp['REVT']/temp['AT'].shift(-1)
    temp['Cash_Ratio'] = temp['C&E']/temp['CL'].shift(-1)
    temp['ROE'] = temp['NI']/(temp['AT'] - temp['TD'])
#The following is the training data
    X_train = temp.filter(['fyear','Leverage_ratio','Profit_Margin','Asset_Turnover','Cash_Ratio'], axis=1)
    X_train = X_train[X_train.fyear <2014] 
    X_train.set_index(['fyear'],inplace = True)
    y_train = temp.filter(['fyear','ROE'],axis = 1)
    y_train = y_train[y_train.fyear <2014]
    y_train.set_index(['fyear'], inplace = True)
#The following is the testing data 
    X_test = temp.filter(['fyear','Leverage_ratio','Profit_Margin','Asset_Turnover','Cash_Ratio'], axis=1);
    X_test = X_test[X_test.fyear  == 2014]
    X_test.set_index(['fyear'], inplace = True)
    regressor = LinearRegression()
    try:
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test) 
        #print(y_pred)
        ROE_List_14.loc[x,'ROE_prediction'] = y_pred
    except:
        pass

'''


