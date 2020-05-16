# -*- coding: utf-8 -*-
"""

VAR_model

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson

def read_energy_data():
	# read energy data.
	df = pd.read_csv("..\\Data\\energy_data.csv")
	df = df.dropna()
	# The smart meters captures half hourly readings. 
	# So, daily energy is calculated by aggregating all the half hourly readings. 
	# Some entries do not have all 24 hr data, so we multiply the remaining factor 
	# by the median energy to normalize the energy value."""
	df.loc[df['energy_count'] < 48, 'energy_sum'] = (48 - df['energy_count']) * df['energy_median'] + df['energy_sum']
	
	housecount = df.groupby('day')[['LCLid']].count()
	# data preprocessing for the energy dataset.
	# We see that the number of houses on each day is not the same. So, we cannot use the energy_sum, energy_mean values directly for analysis.
	# We calculate the energy/household so as to normalize the data.
	# calculate the sum of total energy per day.
	total_energy = df.groupby("day")[["energy_sum", "energy_median"]].sum()
	energy_df = total_energy.merge(housecount, on=["day"])
	energy_df = energy_df.reset_index()
	energy_df['energy_sum'] = energy_df['energy_sum']/energy_df['LCLid']
	energy_df['energy_median'] = energy_df['energy_median']/energy_df['LCLid']
	energy_df.drop(energy_df.tail(1).index,inplace=True)
	energy_df['day'] = pd.to_datetime(energy_df['day'],format='%Y-%m-%d').dt.date
	energy_df = energy_df.set_index('day')
	return energy_df

def read_weather_data():
	weather_df = pd.read_csv("..\\Data\\weather_daily_darksky.csv")
	weather_df['day'] = pd.to_datetime(weather_df['time'],format='%Y-%m-%d').dt.date
	weather_df = weather_df.drop('time', axis=1)
	weather_df = weather_df.set_index('day')
	return weather_df

def merge_data(energy_df, weather_df):
	merged_df = energy_df.merge(weather_df, how='inner', on='day')
	merged_df_1 = merged_df.drop(['LCLid', 'temperatureMax', 'temperatureHigh', 'pressure', 'windBearing', 'windSpeed', 'apparentTemperatureHigh', 'apparentTemperatureLow', 'temperatureMin', 'apparentTemperatureMax', 'apparentTemperatureMin', 'temperatureMaxTime', 'temperatureMinTime', 'icon', 'apparentTemperatureMinTime', 'apparentTemperatureHighTime', 'precipType', 'visibility', 'sunsetTime', 'sunriseTime', 'temperatureHighTime', 'uvIndexTime', 'summary', 'temperatureLowTime', 'apparentTemperatureMaxTime', 'apparentTemperatureLowTime', 'moonPhase'], axis=1)
	merged_df_1.dropna(inplace=True)
	merged_df_1['temperature'] = merged_df_1['temperatureLow']
	merged_df_1 = merged_df_1.drop('temperatureLow', axis=1)
	return merged_df_1
	
def do_adftest(endog):
	X = endog['energy_sum'].values
	result = adfuller(X)
	print(result)
	X = endog['temperature'].values
	result = adfuller(X)
	print(result)
	X = endog['energy_median'].values
	result = adfuller(X)
	print(result)
	
def granger_causality_test(merged_df_train_diff, maxlag):
	test1_a = grangercausalitytests(merged_df_train_diff[['energy_sum', 'energy_median']], maxlag, verbose=False)
	test1_b = grangercausalitytests(merged_df_train_diff[['energy_median', 'energy_sum']], maxlag, verbose=False)
	test2_a = grangercausalitytests(merged_df_train_diff[['energy_sum', 'temperature']], maxlag, verbose=False)
	test2_b = grangercausalitytests(merged_df_train_diff[['temperature', 'energy_sum']], maxlag, verbose=False)
	test3_a = grangercausalitytests(merged_df_train_diff[['temperature', 'energy_median']], maxlag, verbose=False)
	test3_b = grangercausalitytests(merged_df_train_diff[['energy_median', 'temperature']], maxlag, verbose=False)
	
	#Extract the p-value
	test1_a[1][0]['ssr_chi2test'][1]
	min_p_value = min([round(test1_a[i+1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)])
	print(min_p_value)
	print(min([round(test1_b[i+1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)]))
	print(min([round(test2_a[i+1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)]))
	print(min([round(test2_b[i+1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)]))
	print(min([round(test3_a[i+1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)]))
	print(min([round(test3_b[i+1][0]['ssr_chi2test'][1], 4) for i in range(maxlag)]))

def plot_results(merged_df_test, df_forecast_inverse):
	ax = merged_df_test['energy_sum'].plot(figsize=(12,5))
	df_forecast_inverse['energy_sum'].plot(ax=ax, color = 'red')
	
if __name__ == '__main__':
	energy_df = read_energy_data()
	energy_df['energy_sum'].plot(figsize=(10,5))
	weather_df = read_weather_data()
	merged_dataframe = merge_data(energy_df, weather_df)
	merged_dataframe['temperature'].plot(figsize=(20,5))
	
	# Check correlation - numerical features.
	corr_matrix = merged_dataframe.corr("pearson")
	plt.figure(figsize=(10, 10))
	sns.heatmap(corr_matrix, square=True, annot=True)
	merged_dataframe = merged_dataframe.sort_values(by=['day'])

	# Split into train and test.
	merged_df_train = merged_dataframe.iloc[0:len(merged_dataframe) - 30]
	print(len(merged_df_train))
	merged_df_test = merged_dataframe.iloc[len(merged_df_train):(len(merged_dataframe))]
	print(len(merged_df_test))
	
	# Auto-correlation and partial auto-correlation plots.
	#plot_acf(merged_df_train['energy_sum'], lags=100)
	#plot_pacf(merged_df_train['energy_sum'], lags=50)
	
	# Seasonal decomposition of energy, temperature time series to check trend and seasonality.
	result = seasonal_decompose(merged_df_train.energy_sum.values, model='additive', freq=12)
	result1 = seasonal_decompose(merged_df_train.temperature.values, model='additive', freq=12)
	
	# Dependent variables (time series data)
	endog = merged_df_train[['energy_sum', 'energy_median', 'temperature']]

	#Augmented Dickey Fuller test for stationarity check.
	do_adftest(endog)
	
	# Difference the time series to remove trend.
	merged_df_train_diff = endog.diff().dropna()
	#merged_df_train_diff['energy_sum'] = [endog['energy_sum'][i] - endog['energy_sum'][i-interval] for i in range(interval, len(endog['energy_sum']))]
	do_adftest(merged_df_train_diff)
	
	#Second diff if necessary.
	#merged_df_train_diff = merged_df_train_diff.diff().dropna()
	merged_df_train_diff['energy_sum'].plot(figsize=(20,5))
	#plot_acf(merged_df_train_diff['energy_sum'])
	#plot_pacf(merged_df_train_diff['energy_sum'])
	
	# Granger's causality test to check time series inter dependence.
	granger_causality_test(merged_df_train_diff, maxlag=20)
	
	# Fit VAR model
	model_var = VAR(merged_df_train_diff)
	results_var = model_var.fit(ic='aic')
	print('AIC score - ', results_var.aic)
	
	# Durbin watson test for residual correlation check.
	out = durbin_watson(results_var.resid)
	print('Durbin watson stat - ', out)
	
	# Get the lag order and forecast the values.
	lag_order = results_var.k_ar
	forecast_inp = merged_df_train_diff.values[-lag_order:]
	fc = results_var.forecast(y=forecast_inp, steps=len(merged_df_test)) 
	df_forecast = pd.DataFrame(fc, index=merged_df_test.index[-len(merged_df_test):], columns=['energy_sum', 'energy_median', 'temperature'])
	
	# The forecast in the differenced scale. Invert back to normal scale.
	df_forecast_inverse = df_forecast.copy()
	# Invert second difference
	#df_forecast_inverse['energy_sum'] = (merged_df_train['energy_sum'].iloc[-1] - merged_df_train['energy_sum'].iloc[-2]) + df_forecast_inverse['energy_sum'].cumsum()
	# Invert first difference.
	df_forecast_inverse['energy_sum'] = merged_df_train['energy_sum'].iloc[-1] + df_forecast_inverse['energy_sum'].cumsum()
	
	# Save results.
	merged_output = df_forecast_inverse.merge(merged_df_test, how='inner', on='day')
	merged_output = merged_output.rename({'energy_sum_x': 'Predicted', 'energy_sum_y': 'Actual'}, axis=1)
	merged_output_final = merged_output[['Predicted', 'Actual']]
	merged_output_final.to_csv('..\\Data\\var_output.csv')
	#Plot results.
	#plot_results(merged_df_test, df_forecast_inverse)
	
	#Evaluation
	
	#MAE
	mae = np.mean(np.abs(df_forecast_inverse['energy_sum'] - merged_df_test['energy_sum']))
	print(mae)
	#RMSE
	rmse = np.mean((df_forecast_inverse['energy_sum'] - merged_df_test['energy_sum'])**2)**0.5
	print(rmse)
	#MAPE
	mape = np.mean(np.abs(df_forecast_inverse['energy_sum'] - merged_df_test['energy_sum']) / np.abs(merged_df_test['energy_sum'])) * 100
	print(mape)

# End