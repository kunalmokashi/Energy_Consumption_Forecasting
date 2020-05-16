#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read energy data
df = pd.read_csv("..\\Data\\energy_data.csv")
weather_df = pd.read_csv("..\\Data\\weather_daily_darksky.csv")


# In[2]:


df.head()


# In[3]:


df = df.dropna()


# In[4]:


df.loc[df['energy_count'] < 48, 'energy_sum'] = (48 - df['energy_count']) * df['energy_median'] + df['energy_sum']


# In[5]:


weather_df.head()


# In[6]:


holiday_df = pd.read_csv("..\\Data\\uk_bank_holidays.csv")
holiday_df['day'] = holiday_df['Bank holidays']
holiday_df['day'] = pd.to_datetime(holiday_df['day'],format='%Y-%m-%d').dt.date
holiday_df = holiday_df.set_index('day')
holiday_df = holiday_df.dropna()
holiday_df['isHoliday'] = [True]* len(holiday_df)
holiday_df = holiday_df.drop(['Bank holidays', 'Type'], axis=1)
holiday_df.head()


# In[7]:


housecount = df.groupby('day')[['LCLid']].count()
housecount


# In[92]:


housecount.plot(figsize=(20,5))


# In[9]:


# calculate the sum of total energy per day.
total_energy = df.groupby("day")[["energy_sum", "energy_median"]].sum()
total_energy


# In[10]:


energy_df = total_energy.merge(housecount, on=["day"])
energy_df = energy_df.reset_index()
energy_df


# In[11]:


energy_df['energy_sum'] = energy_df['energy_sum']/energy_df['LCLid']
energy_df['energy_median'] = energy_df['energy_median']/energy_df['LCLid']
energy_df.drop(energy_df.tail(1).index, inplace = True)


# In[12]:


energy_df['day'] = pd.to_datetime(energy_df['day'],format='%Y-%m-%d').dt.date


# In[13]:


energy_df = energy_df.set_index('day')


# In[14]:


energy_hol_df = energy_df.merge(holiday_df, how='left', on='day')
energy_hol_df.head()


# In[15]:


energy_hol_df = energy_hol_df.fillna(value={'isHoliday': False})


# In[16]:


weather_df.head()


# In[17]:


weather_df['day'] = pd.to_datetime(weather_df['time'],format='%Y-%m-%d').dt.date
weather_df = weather_df.drop('time', axis=1)
weather_df = weather_df.set_index('day')


# In[18]:


merged_df = energy_hol_df.merge(weather_df, how='inner', on='day')
merged_df_1 = merged_df.drop(['temperatureHigh', 'uvIndex', 'windBearing', 'dewPoint', 'windSpeed', 'apparentTemperatureHigh', 'apparentTemperatureLow', 'temperatureMax', 'temperatureMin', 'apparentTemperatureMax', 'apparentTemperatureMin', 'temperatureMaxTime', 'temperatureMinTime', 'icon', 'apparentTemperatureMinTime', 'apparentTemperatureHighTime', 'precipType', 'visibility', 'sunsetTime', 'sunriseTime', 'temperatureHighTime', 'uvIndexTime', 'summary', 'temperatureLowTime', 'apparentTemperatureMaxTime', 'apparentTemperatureLowTime', 'moonPhase'], axis=1)
merged_df_1.dropna(inplace=True)
merged_df_1.head()


# In[19]:


merged_df_1.head()


# In[20]:


plt.plot(merged_df_1['temperatureLow'])


# In[21]:


merged_df1 = merged_df_1.sort_values(by=['day'])
merged_df1


# In[22]:


merged_df_varmax = merged_df1.copy()


# In[23]:


merged_df_varmax_1 = merged_df_varmax.reset_index()


# In[24]:


merged_df_varmax_1['day'] = pd.to_datetime(merged_df_varmax_1['day'], format="%Y-%m-%d")


# In[25]:


merged_df_varmax_1['dayofweek'] = merged_df_varmax_1['day'].dt.dayofweek


# In[26]:


merged_df_varmax_1['month'] = merged_df_varmax_1['day'].dt.month


# In[27]:


merged_df_varmax_1.head()


# In[28]:


merged_df_varmax_1 = merged_df_varmax_1.set_index('day')


# In[29]:


merged_df_varmax_1.head()


# Drop some features that are not required.

# In[30]:


corr = merged_df_varmax_1.corr("pearson")
corr


# In[31]:


merged_df_varmax_1_updated = merged_df_varmax_1.drop(['LCLid', 'cloudCover', 'pressure'], axis=1)


# In[32]:


merged_df_varmax_1_updated.columns


# Encoding for categorical features.

# In[33]:


merged_df_varmax_1_updated['weekend'] = [False] * len(merged_df_varmax_1_updated)
merged_df_varmax_1_updated.loc[(merged_df_varmax_1_updated['dayofweek'] == 5), 'weekend'] = True
merged_df_varmax_1_updated.loc[(merged_df_varmax_1_updated['dayofweek'] == 6), 'weekend'] = True


# In[35]:


merged_df_varmax_1_updated['dayofweek'] = pd.Categorical(merged_df_varmax_1_updated['dayofweek'])
merged_df_varmax_1_updated = pd.concat([merged_df_varmax_1_updated, pd.get_dummies(merged_df_varmax_1_updated['dayofweek'], prefix='day')],axis=1)

merged_df_varmax_1_updated['weekend'] = pd.Categorical(merged_df_varmax_1_updated['weekend'])
merged_df_varmax_1_updated = pd.concat([merged_df_varmax_1_updated, pd.get_dummies(merged_df_varmax_1_updated['weekend'], prefix='weekend')],axis=1)

merged_df_varmax_1_updated['month'] = pd.Categorical(merged_df_varmax_1_updated['month'])
merged_df_varmax_1_updated = pd.concat([merged_df_varmax_1_updated, pd.get_dummies(merged_df_varmax_1_updated['month'], prefix='month')],axis=1)

merged_df_varmax_1_updated['isHoliday'] = pd.Categorical(merged_df_varmax_1_updated['isHoliday'])
merged_df_varmax_1_updated = pd.concat([merged_df_varmax_1_updated, pd.get_dummies(merged_df_varmax_1_updated['isHoliday'], prefix='holiday')],axis=1)


# In[36]:


merged_df_varmax_1_updated.head()


# In[37]:


# merged_df_varmax_final = merged_df_varmax_1_updated.drop(['dayofweek'], axis=1)
merged_df_varmax_final = merged_df_varmax_1_updated.drop(['dayofweek', 'month', 'weekend', 'isHoliday'], axis=1)
merged_df_varmax_final.head()


# In[ ]:


corr = merged_df_varmax_final.corr("pearson")
corr['energy_sum']


# In[39]:


plt.plot(merged_df_varmax_final['energy_sum'])


# In[40]:


#split data

merged_df_varmax_train = merged_df_varmax_final[: len(merged_df_varmax_final) - 30]
merged_df_varmax_test = merged_df_varmax_final[len(merged_df_varmax_train):]


# In[41]:


merged_df_varmax_train.columns


# In[42]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(merged_df_varmax_train.energy_sum.values, model='additive', freq=12)
result.plot()


# In[43]:


endog = merged_df_varmax_train[['energy_sum', 'energy_median']]


# In[44]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


# In[45]:


X = endog['energy_sum'].values
result = adfuller(X)
print(result)

X = endog['energy_median'].values
result = adfuller(X)
print(result)


# In[46]:


endog_diff = endog.diff().dropna()


# In[47]:


endog_diff.head()


# In[48]:


X = endog_diff['energy_sum'].values
result = adfuller(X)
print(result)

X = endog_diff['energy_median'].values
result = adfuller(X)
print(result)


# In[49]:


merged_df_varmax_train.columns


# In[53]:


#exog = merged_df_varmax_train[['humidity', 'temperatureLow', 'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'holiday_0.0', 'holiday_1.0']]
exog = merged_df_varmax_train[['humidity', 'temperatureLow', 'month_1', 'month_2', 'month_3', 'month_4', 'month_7']]
# exog = merged_df_varmax_train[['humidity', 'temperatureLow', 'weekend_True', 'holiday_True', 'month_1', 'month_2', 'month_3', 'month_4','month_5',
#                                 'month_6', 'month_7', 'month_8', 'month_9', 'month_10','month_11', 'month_12']]
#exog = merged_df_varmax_train[['humidity', 'temperatureLow', 'weekend_False', 'weekend_True']]
exog = exog[1:]
exog.head()


# In[50]:


plot_acf(endog_diff['energy_sum'], lags=20)


# In[51]:


plot_pacf(endog_diff['energy_sum'], lags=20)


# In[54]:


from statsmodels.tsa.statespace.varmax import VARMAX
model_varmax = VARMAX(endog=endog_diff, exog=exog, order=(15, 0))
results_varmax = model_varmax.fit(maxiter=5000, disp=False)
results_varmax.summary()


# In[55]:


results_varmax.plot_diagnostics()


# In[56]:


#exog_test = merged_df_varmax_test[['humidity', 'temperatureLow', 'month_1', 'month_2', 'month_3',
#                                   'month_4','month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10','month_11', 'month_12']]
#exog_test = merged_df_varmax_test[['humidity', 'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
#                                'holiday_0.0', 'holiday_1.0']]
#exog_test = merged_df_varmax_test[['humidity', 'temperatureLow', 'weekend_False', 'weekend_True']]
exog_test = merged_df_varmax_test[['humidity', 'temperatureLow', 'month_1', 'month_2', 'month_3', 'month_4', 'month_7']]
exog_test.head()


# In[57]:


forecast_out = results_varmax.forecast(steps=30, exog=exog_test)


# In[58]:


plt.plot(forecast_out['energy_sum'])


# In[59]:


merged_df_varmax_test.head()


# In[101]:


#merged_df_varmax_test.set_index('day', inplace=True)


# In[60]:


df_forecast = pd.DataFrame(forecast_out.values, index=merged_df_varmax_test.index[-len(merged_df_varmax_test):], columns=['energy_sum', 'energy_median'])
df_forecast


# In[61]:


len(df_forecast)


# In[62]:


df_forecast.head()


# In[63]:


df_forecast_inverse = df_forecast.copy()


# In[64]:


merged_df_varmax_train['energy_sum'].iloc[-1]


# In[65]:


df_forecast_inverse['energy_sum'] = merged_df_varmax_train['energy_sum'].iloc[-1] + df_forecast_inverse['energy_sum'].cumsum()


# In[99]:


plt.plot(merged_df_varmax_test['energy_sum'], label='Actual')
plt.plot(df_forecast_inverse['energy_sum'], color='red', label='Predicted')
plt.xticks(size=12)
plt.ylabel('total energy', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.legend()


# In[71]:


merged_output = df_forecast_inverse.merge(merged_df_varmax_test, how='inner', on='day')


# In[76]:


merged_output = merged_output.rename({'energy_sum_x': 'Predicted', 'energy_sum_y': 'Actual'}, axis=1)
merged_output_final = merged_output[['Predicted', 'Actual']]
merged_output_final.to_csv("..\\Data\\varx_output.csv")


# In[77]:


#evaluation

#MAE
mae = np.mean(np.abs(df_forecast_inverse['energy_sum'] - merged_df_varmax_test[:30]['energy_sum']))
print('MAE -', mae)

#RMSE
rmse = np.mean((df_forecast_inverse['energy_sum'] - merged_df_varmax_test[:30]['energy_sum'])**2)**0.5
print('RMSE - ', rmse)

#MAPE
mape = np.mean(np.abs(df_forecast_inverse['energy_sum'] - merged_df_varmax_test[:30]['energy_sum']) / np.abs(merged_df_varmax_test[:30]['energy_sum'])) * 100
print('MAPE - ', mape)

