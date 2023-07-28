# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:01:38 2022

@author: naouf
"""

import os
import pandas as pd
folder= r'\Users\naouf\OneDrive\Desktop\thesis\Meteorological Data\Counties'
df_meteo_counties=pd.DataFrame()
files=os.listdir(folder)
files
for file in files : 
    if file.endswith('.xlsx'):
        excel_file=pd.ExcelFile(f'{folder}/{file}')
        sheets=excel_file.sheet_names
        for sheet in sheets:
            df=excel_file.parse(sheet_name=sheet)
            df_meteo_counties=df_meteo_counties.append(df)

df_meteo_counties.to_excel(f'{folder}/Meteo_df.xlsx')

#### library
import pandas as pd 
import numpy as np 
import datetime as dt
            
### Corn  and soybean useful data - growing season months => 5/6/7/8 #


df= pd.read_excel(r'C:\Users\naouf\OneDrive\Desktop\thesis\Meteorological Data\Counties\MT_DATA\Meteo_df.xlsx')


df['Date']=pd.to_datetime(df['Date'])

corndf_MT=df.loc[(df['Date'].dt.month>=5) & (df['Date'].dt.month<9) & (df['Date'].dt.year<2021)]


data_5=corndf_MT.loc[(corndf_MT['Date'].dt.month==5)]
data_6=corndf_MT.loc[(corndf_MT['Date'].dt.month==6)]
data_7=corndf_MT.loc[(corndf_MT['Date'].dt.month==7)]
data_8=corndf_MT.loc[(corndf_MT['Date'].dt.month==8)]

data_5['Date']=pd.to_datetime(data_5['Date']).dt.strftime('%Y')       
data_5['Date']=data_5['Date'].astype(int)
data_5['County']=data_5['County'].str.upper()

data_5=data_5.rename(columns={'ppt (mm)':'ppt_5','tmin (degrees C)':'tmin_5','tmean (degrees C)':'tmean_5','tmax (degrees C)':'tmax_5','tdmean (degrees C)':'tdmean_5',
                       'vpdmin (hPa)':'vpdmin_5','vpdmax (hPa)':'vpdmax_5'})

data_6=data_6.rename(columns={'ppt (mm)':'ppt_6','tmin (degrees C)':'tmin_6','tmean (degrees C)':'tmean_6','tmax (degrees C)':'tmax_6','tdmean (degrees C)':'tdmean_6',
                       'vpdmin (hPa)':'vpdmin_6','vpdmax (hPa)':'vpdmax_6'})

data_7=data_7.rename(columns={'ppt (mm)':'ppt_7','tmin (degrees C)':'tmin_7','tmean (degrees C)':'tmean_7','tmax (degrees C)':'tmax_7','tdmean (degrees C)':'tdmean_7',
                       'vpdmin (hPa)':'vpdmin_7','vpdmax (hPa)':'vpdmax_7'})

data_8=data_8.rename(columns={'ppt (mm)':'ppt_8','tmin (degrees C)':'tmin_8','tmean (degrees C)':'tmean_8','tmax (degrees C)':'tmax_8','tdmean (degrees C)':'tdmean_8',
                       'vpdmin (hPa)':'vpdmin_8','vpdmax (hPa)':'vpdmax_8'}) 

data_6=data_6.drop(columns=['Date','County'])
data_7=data_7.drop(columns=['Date','County'])
data_8=data_8.drop(columns=['Date','County'])

#### reset the index
data_5=data_5.reset_index(drop=True)
data_6=data_6.reset_index(drop=True)
data_7=data_7.reset_index(drop=True)
data_8=data_8.reset_index(drop=True)
data=pd.concat([data_5,data_6,data_7,data_8],axis=1)

# wheat useful data - growing season => 10-11- ... -05-06

wheat_MT= df.loc[(df['Date'].dt.month>=10)&(df['Date'].dt.year<= 2020)]
wheat_MTi=df.loc[(df['Date'].dt.year>=1981)&(df['Date'].dt.month<7)&(df['Date'].dt.year<=2021)]


wheatdf_MT=pd.concat([wheat_MT,wheat_MTi])
wheatdf_MT= wheatdf_MT.sort_values(['County','Date'])


dt_10=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==10)]                        
dt_11=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==11)]       
dt_12=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==12)]
dt_1=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==1)]
dt_2=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==2)]
dt_3=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==3)]
dt_4=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==4)]
dt_5=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==5)]
dt_6=wheatdf_MT.loc[(wheatdf_MT['Date'].dt.month==6)]

dt_10['Date']=pd.to_datetime(dt_10['Date']).dt.strftime('%Y')       
dt_10['Date']=dt_10['Date'].astype(int)
dt_10['County']=dt_10['County'].str.upper() 

dt_10=dt_10.rename(columns={'ppt (mm)':'ppt_10','tmin (degrees C)':'tmin_10','tmean (degrees C)':'tmean_10','tmax (degrees C)':'tmax_10','tdmean (degrees C)':'tdmean_10',
                       'vpdmin (hPa)':'vpdmin_10','vpdmax (hPa)':'vpdmax_10'})

dt_11=dt_11.rename(columns={'ppt (mm)':'ppt_11','tmin (degrees C)':'tmin_11','tmean (degrees C)':'tmean_11','tmax (degrees C)':'tmax_11','tdmean (degrees C)':'tdmean_11',
                       'vpdmin (hPa)':'vpdmin_11','vpdmax (hPa)':'vpdmax_11'})

dt_12=dt_12.rename(columns={'ppt (mm)':'ppt_12','tmin (degrees C)':'tmin_12','tmean (degrees C)':'tmean_12','tmax (degrees C)':'tmax_12','tdmean (degrees C)':'tdmean_12',
                       'vpdmin (hPa)':'vpdmin_12','vpdmax (hPa)':'vpdmax_12'})

dt_1=dt_1.rename(columns={'ppt (mm)':'ppt_1','tmin (degrees C)':'tmin_1','tmean (degrees C)':'tmean_1','tmax (degrees C)':'tmax_1','tdmean (degrees C)':'tdmean_1',
                       'vpdmin (hPa)':'vpdmin_1','vpdmax (hPa)':'vpdmax_1'})

dt_2=dt_2.rename(columns={'ppt (mm)':'ppt_2','tmin (degrees C)':'tmin_2','tmean (degrees C)':'tmean_2','tmax (degrees C)':'tmax_2','tdmean (degrees C)':'tdmean_2',
                       'vpdmin (hPa)':'vpdmin_2','vpdmax (hPa)':'vpdmax_2'})

dt_3=dt_3.rename(columns={'ppt (mm)':'ppt_3','tmin (degrees C)':'tmin_3','tmean (degrees C)':'tmean_3','tmax (degrees C)':'tmax_3','tdmean (degrees C)':'tdmean_3',
                       'vpdmin (hPa)':'vpdmin_3','vpdmax (hPa)':'vpdmax_3'})

dt_4=dt_4.rename(columns={'ppt (mm)':'ppt_4','tmin (degrees C)':'tmin_4','tmean (degrees C)':'tmean_4','tmax (degrees C)':'tmax_4','tdmean (degrees C)':'tdmean_4',
                       'vpdmin (hPa)':'vpdmin_4','vpdmax (hPa)':'vpdmax_4'})

dt_5=dt_5.rename(columns={'ppt (mm)':'ppt_5','tmin (degrees C)':'tmin_5','tmean (degrees C)':'tmean_5','tmax (degrees C)':'tmax_5','tdmean (degrees C)':'tdmean_5',
                       'vpdmin (hPa)':'vpdmin_5','vpdmax (hPa)':'vpdmax_5'})

dt_6=dt_6.rename(columns={'ppt (mm)':'ppt_6','tmin (degrees C)':'tmin_6','tmean (degrees C)':'tmean_6','tmax (degrees C)':'tmax_6','tdmean (degrees C)':'tdmean_6',
                       'vpdmin (hPa)':'vpdmin_6','vpdmax (hPa)':'vpdmax_6'})

dt_11=dt_11.drop(columns=['Date','County'])
dt_12=dt_12.drop(columns=['Date','County'])
dt_1=dt_1.drop(columns=['Date','County'])
dt_2=dt_2.drop(columns=['Date','County'])
dt_3=dt_3.drop(columns=['Date','County'])
dt_4=dt_4.drop(columns=['Date','County'])
dt_5=dt_5.drop(columns=['Date','County'])
dt_6=dt_6.drop(columns=['Date','County'])

dt_10=dt_10.reset_index(drop=True)
dt_11=dt_11.reset_index(drop=True)
dt_12=dt_12.reset_index(drop=True)
dt_1=dt_1.reset_index(drop=True)
dt_2=dt_2.reset_index(drop=True)
dt_3=dt_3.reset_index(drop=True)
dt_4=dt_4.reset_index(drop=True)
dt_5=dt_5.reset_index(drop=True)
dt_6=dt_6.reset_index(drop=True)

data_=pd.concat([dt_10,dt_11,dt_12,dt_1,dt_2,dt_3,dt_4,dt_5,dt_6],axis=1)

####### Yield  
df_corn_yield=pd.read_excel(r'C:\Users\naouf\OneDrive\Desktop\thesis\Crop Yield Data\Corn yield.xlsx')
df_sb_yield=pd.read_excel(r'C:\Users\naouf\OneDrive\Desktop\thesis\Crop Yield Data\Soybean yield.xlsx')
df_wheat_yield=pd.read_excel(r'C:\Users\naouf\OneDrive\Desktop\thesis\Crop Yield Data\Wheat yield.xlsx')    

 

#### filtring DATA
Corn_yield=df_corn_yield.loc[(df_corn_yield['Date']>=1980)&(df_corn_yield['County']!='OTHER (COMBINED) COUNTIES') & (df_corn_yield['County']!='OTHER COUNTIES')]
sb_yield=df_sb_yield.loc[(df_sb_yield['Date']>=1980)&(df_sb_yield['County']!='OTHER (COMBINED) COUNTIES') & (df_sb_yield['County']!='OTHER COUNTIES')]
wt_yield=df_wheat_yield.loc[(df_wheat_yield['Date']>=1980)&(df_wheat_yield['County']!='OTHER (COMBINED) COUNTIES') & (df_wheat_yield['County']!='OTHER COUNTIES')&(df_wheat_yield['Date']<2021)]


#### Creating a dict of data frames and droping duplicated elements in each year  

my_df={}

for i in range(1980,2021):
    X=pd.DataFrame()
    X=Corn_yield.loc[(Corn_yield['Date']==i)]
    my_df[i]=X.drop_duplicates(subset=['County'])

A=pd.concat([my_df[1980],my_df[1981]]) 
X1=A

for i in range(1982,2021):
   b=my_df[i]
   X1=pd.concat([X1,b])
   
X1=X1.sort_values(['County','Date'])   


my_df={}

for i in range(1980,2021):
    X=pd.DataFrame()
    X=sb_yield.loc[(sb_yield['Date']==i)]
    my_df[i]=X.drop_duplicates(subset=['County'])

A=pd.concat([my_df[1980],my_df[1981]]) 
X2=A

for i in range(1982,2021):
   b=my_df[i]
   X2=pd.concat([X2,b])
   
X2=X2.sort_values(['County','Date'])  

my_df={}

for i in range(1980,2021):
    X=pd.DataFrame()
    X=wt_yield.loc[(wt_yield['Date']==i)]
    my_df[i]=X.drop_duplicates(subset=['County'])

A=pd.concat([my_df[1980],my_df[1981]]) 
X3=A

for i in range(1982,2021):
   b=my_df[i]
   X3=pd.concat([X3,b])
   
X3=X3.sort_values(['County','Date'])  


##### using merge for the data frames to obtain the right level

df_corn=pd.merge(X1,data,how='left',on=['County','Date'])
df_sb=pd.merge(X2,data,how='left',on=['County','Date'])
df_wt=pd.merge(X3,data_,how='left',on=['County','Date'])

#### checking results 
Count1=df_corn.groupby('County').count()
Count2=X1.groupby('County').count()
df_corn.isnull().sum().sum()

Count3=df_sb.groupby('County').count()
Count4=X2.groupby('County').count()
df_sb.isnull().sum().sum()

Count5=df_wt.groupby('County').count()
Count6=X3.groupby('County').count()
df_wt.isnull().sum().sum()


df_corn=df_corn.drop(['tmin_5','tmax_5','tmin_6','tmax_6','tmin_7','tmax_7','tmin_8','tmax_8'],axis=1)
df_sb=df_sb.drop(['tmin_5','tmax_5','tmin_6','tmax_6','tmin_7','tmax_7','tmin_8','tmax_8'],axis=1)
df_wt=df_wt.drop(['tmin_10','tmax_10','tmin_11','tmax_11','tmin_12','tmax_12','tmin_1','tmax_1','tmin_2','tmax_2','tmin_3',
                  'tmax_3','tmin_4','tmax_4','tmin_5','tmax_5','tmin_6','tmax_6'],axis=1)

####### linear regression Corn
#### define the dependent y and independent var X

X_c=df_corn[['ppt_5','tmean_5','tdmean_5','vpdmin_5','vpdmax_5','ppt_6','tmean_6','tdmean_6','vpdmin_6','vpdmax_6',
             'ppt_7','tmean_7','tdmean_7','vpdmin_7','vpdmax_7','ppt_8','tmean_8','tdmean_8','vpdmin_8','vpdmax_8']].values

X_c.shape[1]
X_sb=df_sb[['ppt_5','tmean_5','tdmean_5','vpdmin_5','vpdmax_5','ppt_6','tmean_6','tdmean_6','vpdmin_6','vpdmax_6',
             'ppt_7','tmean_7','tdmean_7','vpdmin_7','vpdmax_7','ppt_8','tmean_8','tdmean_8','vpdmin_8','vpdmax_8']]

X_wt=df_wt[['ppt_10','tmean_10','tdmean_10','vpdmin_10','vpdmax_10','ppt_11','tmean_11','tdmean_11','vpdmin_11','vpdmax_11','ppt_12','tmean_12','tdmean_12','vpdmin_12','vpdmax_12','ppt_1','tmean_1','tdmean_1','vpdmin_1','vpdmax_1',
            'ppt_2','tmean_2','tdmean_2','vpdmin_2','vpdmax_2','ppt_3','tmean_3','tdmean_3','vpdmin_3','vpdmax_3','ppt_4','tmean_4','tdmean_4','vpdmin_4','vpdmax_4','ppt_5','tmean_5','tdmean_5','vpdmin_5','vpdmax_5','ppt_6','tmean_6','tdmean_6','vpdmin_6','vpdmax_6']].values

X_c_train.pop('const')
y_c=df_corn['Value'].values
y_c=y_c.reshape(-1,1)
y_sb=df_sb['Value'].values.reshape(-1,1)
y_wt=df_wt['Value'].values.reshape(-1,1)
#### split to test and train data
from sklearn.model_selection import train_test_split
X_c_train,X_c_test,y_c_train,y_c_test=train_test_split(X_c,y_c,test_size=0.25,random_state=42)
                                                     

X_sb_train,X_sb_test,y_sb_train,y_sb_test=train_test_split(X_sb,y_sb,test_size=0.25,random_state=42)
X_wt_train,X_wt_test,y_wt_train,y_wt_test=train_test_split(X_wt,y_wt,test_size=0.25,random_state=42)
#### Train the model with training set

from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
res=mlr.fit(X_c_train,y_c_train)
ols_wt=sm.OLS(y_wt_train,X_wt_train).fit()
res.score(X_c_train,y_c_train)


train_predict= ols_wt.predict(X_wt_train)
test_predict=ols_.predict(X_wt_test)
insample_mape= mean_absolute_percentage_error(y_wt_train, train_predict)
outsample_mape= mean_absolute_percentage_error(y_wt_test, test_predict)
print(1-insample_mape,1-outsample_mape)


print(ols_wt.summary())





### test model

from sklearn.metrics import mean_squared_error
mean_squared_error(y_c_test, y_c_predict)
mean_squared_error(y_sb_test, y_sb_predict)

from sklearn.metrics import r2_score
r2_score(y_c_test, y_c_predict)
r2_score(y_sb_test, y_sb_predict)

crl_c=X_c.corr()
crl_sb=X_sb.corr()
crl_wt=X_wt.corr()


print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(X_c, mlr.coef_))

mlr.score(X_c,y_c)
X_c_test.pop('const')
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X_wt_train)
X_polyt=poly_reg.fit_transform(X_wt_test)
poly_wt=sm.OLS(y_wt_train,X_poly).fit()
print(poly_wt.summary())








print(MAPE_quad)
ypred =ols_corn.predict(X_c)
import matplotlib.pyplot as plt
plt.scatter(X_c,y_c)
plt.plot(y_c,ypred)

r2_score(y_c, ypred)
mean_squared_error(y_c, ypred)


import statsmodels.formula.api as smf
ols_corn=smf.ols(formula='Value~ppt_5+tmean_5+tdmean_5+vpdmin_5+vpdmax_5+ppt_6+tmean_6+tdmean_6+vpdmin_6+vpdmax_6+ppt_7+tmean_7+tdmean_7+vpdmin_7+vpdmax_7+ppt_8+tmean_8+tdmean_8+vpdmin_8+vpdmax_8',
              data=df_corn).fit()
print(ols_corn.summary())

MAPE_ols = 100*mean_absolute_percentage_error(y_c, ypred)
MAPE_ols
100-MAPE_ols

ols_sb=smf.ols(formula='Value~ppt_5+tmean_5+tdmean_5+vpdmin_5+vpdmax_5+ppt_6+tmean_6+tdmean_6+vpdmin_6+vpdmax_6+ppt_7+tmean_7+tdmean_7+vpdmin_7+vpdmax_7+ppt_8+tmean_8+tdmean_8+vpdmin_8+vpdmax_8',
              data=df_sb).fit()
print(ols_sb.summary())

# iterating the columns
for col in df_wt.columns:
    print(col)



ols_wt=smf.ols(formula='Value~ppt_10+tmean_10+tdmean_10+vpdmin_10+vpdmax_10+ppt_11+tmean_11+tdmean_11+vpdmin_11+vpdmax_11+ppt_12+tmean_12+tdmean_12+vpdmin_12+vpdmax_12+ppt_1+tmean_1+tdmean_1+vpdmin_1+vpdmax_1+ppt_2+tmean_2+tdmean_2+vpdmin_2+vpdmax_2+ppt_3+tmean_3+tdmean_3+vpdmin_3+vpdmax_3+ppt_4+tmean_4+tdmean_4+vpdmin_4+vpdmax_4+ppt_5+tmean_5+tdmean_5+vpdmin_5+vpdmax_5+ppt_6+tmean_6+tdmean_6+vpdmin_6+vpdmax_6',
              data=df_wt).fit()

print(ols_wt.summary())

import matplotlib.pyplot as plt
plt.plot(X_c,'c.')
plt.xlabel("ppt mai/august")
plt.ylabel("Frequency")
plt.show()
plt.hist(y_c,bins= 35,)
plt.xlabel("corn yield")
plt.ylabel("Frequency")
plt.hist(X_c[['ppt_5','ppt_6','ppt_7','ppt_8']],bins= 20)
plt.xlabel("ppt mai/august")
plt.ylabel("Frequency")
plt.hist(X_c[['tmean_5','tmean_6','tmean_7','tmean_8']],bins= 25)
plt.xlabel("tmean mai/august")
plt.ylabel("Frequency")
plt.hist(X_c[['tdmean_5','tdmean_6','tdmean_7','tdmean_8']],bins= 25)
plt.xlabel("tdmean mai/august")
plt.ylabel("Frequency")
plt.hist(X_c[['vpdmin_5','vpdmin_6','vpdmin_7','vpdmin_8']],bins= 25)
plt.xlabel("vpdmin mai/august")
plt.ylabel("Frequency")
plt.hist(X_c[['vpdmax_5','vpdmax_6','vpdmax_7','vpdmax_8']],bins= 25)
plt.xlabel("vpdmax mai/august")
plt.ylabel("Frequency")
plt.show()

import  statistics as stt
stt.mean(y_c)
stt.median(y_c)
stt.stdev(y_c)**2
stt.variance(y_c)
#########
from sklearn.linear_model import LinearRegression
m=LinearRegression()
m.fit(X_c_train,y_c_train)



mod=smf.ols(y_c,X_c)

import statsmodels.api as sm
X_c=sm.add_constant(X_c)
X_ct=sm.add_constant(X_wt)
cst= X_c.pop('const')
  
# insert column using insert(position,column_name,
# first_column) function
X_c.insert(0, 'const', cst)


mod=sm.OLS(y_c,X_poly)
resul=mod.fit()
resul.summary()

Bin_c=sm.GLM(y_c,X_c,family=sm.families.Binomial())
r1=Bin_c.fit()
r1.summary()
r1.deviance/r1.null_deviance
ypred = ols_corn.predict(X_c)


glm_gamma = sm.GLM(y_c, X_ct, family=sm.families.Gamma(link=sm.families.links.log())).fit()
print(glm_gamma.summary())

MAPE_gam = mean_absolute_percentage_error(y_c, glm_gamma.predict(X_ct))
MAPE_gam
glm_Poisson = sm.GLM(y_c, X_ct, family=sm.families.Poisson(link=sm.families.links.log())).fit()
print(glm_Poisson.summary())
MAPE_poi = mean_absolute_percentage_error(y_c, glm_Poisson.predict(X_ct))
100-MAPE_poi*100
glm_Gauss = sm.GLM(y_c, X_c, family=sm.families.Gaussian()).fit()
print(glm_Gauss.summary())
MAPE_gau = mean_absolute_percentage_error(y_c, glm_Gauss.predict(X_c))

y_pr=glm_Gauss.predict(X_c)


import seaborn as sns
sns.displot(y_c,bins=15,kde=True,rug=True)


from sklearn.metrics import mean_absolute_percentage_error
train_predict= poly_wt.predict(X_poly)
test_predict=poly_wt.predict(X_polyt)
insample_mape= mean_absolute_percentage_error(y_wt_train, train_predict)
outsample_mape= mean_absolute_percentage_error(y_wt_test, test_predict)
print(1-insample_mape,1-outsample_mape)

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs(((Y_actual - Y_Predicted)/Y_actual))*100)
    return mape

#########################Random forest ##########################################
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators=800,random_state = 42,max_features=5,min_samples_leaf=2,min_samples_split=3,oob_score=True)
random_forest.fit(X_wt_train, y_wt_train)
train_predict=random_forest.predict(X_wt_train)
test_predict=random_forest.predict(X_wt_test)
insample_mape= mean_absolute_percentage_error(y_wt_train, train_predict)
outsample_mape= mean_absolute_percentage_error(y_wt_test, test_predict)
print(1-insample_mape,1-outsample_mape)
r2=r2_score(y_c_test, RF_ts)
from sklearn.model_selection import GridSearchCV
RF = RandomForestRegressor(random_state = 42)
params = {
   'n_estimators': [200,400,600,800,1000],
   'max_features': [3,4,5],
   'max_depth' : [None,5,10,15],
   'min_samples_leaf' :[2,3,4],
   'min_samples_split' :[3,5,7]
}
GD_search = GridSearchCV(estimator=RF,cv=3, param_grid=params,verbose=2,n_jobs=-1)
GD_search.fit(X_wt_train, y_wt_train)
GD_search.best_params_

from sklearn.metrics import accuracy_score
c=y_c_test.values.reshape(-1,1)
d=test_predict.reshape(-1,1)
accuracy_score(c,d,normalize=False)
random_forest_out_of_bag = RandomForestRegressor(oob_score=True)
random_forest_out_of_bag.fit(X_c_train, y_c_train)
print(random_forest.oob_score_) 

















accuracy_ols=100-mape_ols

mape_ols=MAPE(y_c,ols_corn.predict(X_c))
print(100-mape_ols)

mape_gamma=mape(y_c,glm_gamma.predict(X_ct))
print(100-mape_gamma)

mape_poi=mape(y_c,glm_Poisson.predict(X_ct))
print(100-mape_poi)

mape_quad=mape(y_c,poly_corn.predict(X_poly))
print(100-mape_quad)



conda activate base
pip install tensorflow
pip install keras
pip uninstall keras-base

### Sandardization of data ###
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X_wt)

TargetVarScalerFit=TargetVarScaler.fit(y_wt)
 
# Generating the standardized values of X and y
X_wt=PredictorScalerFit.transform(X_wt)
y_wt=TargetVarScalerFit.transform(y_wt)



TargetVariable=['Value']
Predictors=['ppt_5','tmean_5','tdmean_5','vpdmin_5','vpdmax_5','ppt_6','tmean_6','tdmean_6','vpdmin_6','vpdmax_6',
            'ppt_7','tmean_7','tdmean_7','vpdmin_7','vpdmax_7','ppt_8','tmean_8','tdmean_8','vpdmin_8','vpdmax_8']
 
X_=df_corn[Predictors].values
y=df_corn[TargetVariable].values

# importing the libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
import keras
print(keras.__version__)
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.optimizers import RMSprop


from keras.models import Sequential
from keras.layers import Dense
# create ANN model
model = Sequential()
# Defining the Input layer 
model.add(Dense(units=9, input_dim=20, kernel_initializer='normal', activation='relu'))

# Defining the hidden kayer
model.add(Dense(units=15, kernel_initializer='random_normal', activation='relu'))

# The output neuron is a single fully connected node 
model.add(Dense(1, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='Adam')

# Fitting the ANN to the Training set
model.fit(X_c_train, y_c_train ,batch_size = 15, epochs = 40, verbose=0)
### pedicting with our model 
yprd_tn=model.predict(X_c_train)
yprd_ts=model.predict(X_c_test)

### scaling back the predicted data 
yprd_tn=TargetVarScalerFit.inverse_transform(yprd_tn)
yprd_ts=TargetVarScalerFit.inverse_transform(yprd_ts)

### scaling back the train and test data
y_c_test=TargetVarScalerFit.inverse_transform(y_c_test)
y_c_train=TargetVarScalerFit.inverse_transform(y_c_train)
#### accuracy of training and testing sets
insample_mape= mean_absolute_percentage_error(y_c_train, yprd_tn)
outsample_mape= mean_absolute_percentage_error(y_c_test, yprd_ts)
print(1-insample_mape,1-outsample_mape)


###### ANN with one layer
adam=tf.keras.optimizers.Adam(learning_rate = 0.0029, beta_1 = 0.9, beta_2 = 0.999)
def FindBestParams(X_train,y_train,X_test,y_test):
    #### hyperparameters number of layers and number of neurons
    n_units=[3,5,7,9]
    n_layers=3
    #### DF for search results
    search_rs_=pd.DataFrame(columns=['ANN','Number of layers','Number of Neurons','Accuracy trained_data','Accuracy test_data'])
    #initializing the Trial
    Trial_num=0
    for num_units_trial in n_units:
            Trial_num+=1
            
            #### creating ANN
            model=Sequential()
            
            #### creating the input in ANN(first layer)
            model.add(Dense(units=13,input_dim=X_train.shape[1],kernel_initializer='normal',activation='relu'))
            
            #### Hidden layers of the model
            model.add(Dense(units=num_units_trial,kernel_initializer='normal',activation='relu'))
            model.add(Dense(units=num_units_trial,kernel_initializer='normal',activation='relu'))
            model.add(Dense(units=num_units_trial,kernel_initializer='normal',activation='relu'))
            ##### output layer of production
            model.add(Dense(units=1))
            
            #### compiling the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.00032, beta_1 = 0.91, beta_2 = 0.999),loss=tf.keras.losses.MeanSquaredError())
            ##### fitting ANN to training set
            model.fit(X_train,y_train,batch_size=8,epochs=45,verbose=0)
            
            #### Generating Predictions on testing data
            ypred_tn=model.predict(X_train)
            ypred_ts=model.predict(X_test)
            
            ### scaling back the data predicted 
            ypred_tn=TargetVarScalerFit.inverse_transform(ypred_tn)
            ypred_ts=TargetVarScalerFit.inverse_transform(ypred_ts)
            
            ### scaling back the train and test data
            y_test=TargetVarScalerFit.inverse_transform(y_test)
            y_train=TargetVarScalerFit.inverse_transform(y_train)
            #### mean absolute %error
            MAPE_ts = 100*mean_absolute_percentage_error(y_test,ypred_ts)
            MAPE_tn = 100*mean_absolute_percentage_error(y_train,ypred_tn)
            
            #### Printing results
            print(Trial_num,'-','num_layers',n_layers,'-','num_neurons',num_units_trial,'Accuracy_Tn:',100-MAPE_tn,'Accuracy_Ts:',100-MAPE_ts)
                         
            search_rs_=search_rs_.append(pd.DataFrame(data=[[Trial_num,n_layers,num_units_trial,100-MAPE_tn,100-MAPE_ts]],
                                                        columns=['ANN','Number of layers','Number of Neurons','Accuracy trained_data','Accuracy test_data']))
    
    return(search_rs_)
  
rs_=FindBestParams(X_train=X_wt_train, y_train=y_wt_train, X_test=X_wt_test, y_test=y_wt_test)

rs=pd.concat([rs_1,rs_2,rs_],axis=0)


import matplotlib_inline            
rs.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')    

# Fitting the ANN to the Training set
model.fit(X_c_train, y_c_train ,batch_size = 5, epochs =40, verbose=0)

# Generating Predictions on testing data
Predictions=model.predict(X_c_test)

# Scaling the predicted Price data back to original price scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)

# Scaling the y_test Price data back to original price scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_c_test)



TestingData=pd.DataFrame(index=range(0,1221))
TestingData['Actual values']=y_test_orig
TestingData['PredictedPrice']=Predictions
TestingData.head()    
# Computing the absolute percent error
APE=100*(abs(TestingData['Actual values']-TestingData['PredictedPrice'])/TestingData['Actual values'])
TestingData['APE']=APE

print('The Accuracy of ANN model is:', 100-np.mean(APE))

TestingData.head()
TestingData=pd.DataFrame(data=y_test_orig,columns='Actrual values',Index=len(y_test_orig))
TestingData['Price']=y_test_orig
TestingData['PredictedPrice']=Predictions

















