
# This repositary will help beginners to make themselves familiar with the Time Series Analysis in Python.
# repositary includes all the essential things required for a beginner to start with time series. Let's begin!

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6


data = pd.read_csv("C:\\Users\\Siddharth\\Desktop\\AirPassengers.csv")
print(data.head())
print("\n")

print(data.dtypes)
print("\n")

#Convert Month object into datetime
data['Month'] = pd.to_datetime(data.Month)
data = data.set_index(data.Month)
data.drop('Month', axis = 1, inplace = True)
print(data.head())
print("\n")

ts = data['#Passengers']
print(ts.head())
print("\n")

print(ts['1949'])

# 1. Check for Stationarity of Time Series¶
# A TS is said to be stationary if its statistical properties such as mean, variance
# remain constant over time and an autocovariance that does not depend on time.


rcParams['figure.figsize'] = 15,6
plt.plot(ts)
plt.show()

plt.subplot(221)
plt.hist(ts)
plt.subplot(222)
ts.plot(kind = 'kde')
plt.show()

from statsmodels.tsa.stattools import adfuller
#from pandas.core import datetools
#from pandas.core.tools.datetimes import datetools

def test_stationarity(timeseries):
    # Determining rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # plotting rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    st = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling  Mean & Standard Deviation')
    plt.show()

    # Dickey_Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


test_stationarity(ts)

#Standard deviation has very less variation but mean is increasing continously. Hence, it's not a stationary series.
# Also, the test statistic is way more than the critical values (compare signed values and not the absolute values).

# 2. Make the Series Stationary¶
# Its almost impossible to make a series perfectly stationary, but we try to take it as close as possible.
#
# Lets understand what is making a TS non-stationary. There are 2 major reasons behind non-stationaruty of a TS:
#
# Trend – varying mean over time. For eg, in this case we saw that on average, the number of passengers was growing over time.
# Seasonality – variations at specific time-frames. eg people might have a tendency to buy cars in a particular month because of pay increment or festivals.
# The underlying principle is to model or estimate the trend and seasonality in the series and remove those from the series to get a stationary series.
# Then statistical forecasting techniques can be implemented on this series. The final step would be to convert the
# forecasted values into the original scale by applying trend and seasonality constraints back.

# Estimating & Eliminating Trend
# We can clearly see that the there is a significant positive trend. So we can apply transformation which
# penalize higher values more than smaller values. These can be taking a log, square root, cube root, etc.
# Lets take a log transform here for simplicity.

#let's take a log transform for simplicity:
ts_log = np.log(ts)
plt.plot(ts_log)
plt.show()

# In this simpler case, it is easy to see a forward trend in the data. But its not very intuitive in presence of noise.
# So we can use some techniques to estimate or model this trend and then remove it from the series. There can be many
# ways of doing it and some of most commonly used are:
#
# Aggregation – taking average for a time period like monthly/weekly averages
# Smoothing – taking rolling averages
# Polynomial Fitting – fit a regression model
#
# We will apply smoothing here.

# Moving average
# In this approach, we take average of ‘k’ consecutive values depending on the frequency of time series.
# Here we can take the average over the past 1 year, i.e. last 12 values. Pandas has specific functions
# defined for determining rolling statistics.

moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color = 'red')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace = True)
test_stationarity(ts_log_moving_avg_diff)

exp_weighted_avg = ts_log.ewm(halflife = 12).mean()
plt.plot(ts_log)
plt.plot(exp_weighted_avg, color = 'red')
plt.show()

ts_log_ema_diff = ts_log - exp_weighted_avg
test_stationarity(ts_log_ema_diff)

# This TS has even lesser variations in mean and standard deviation in magnitude.
# Also, the test statistic is smaller than the 1% critical value, which is better than the previous case.

# 3. Eliminating Trend and Seasonality
# Two methods:
#
# Differencing (taking the differece with a particular time lag)
# Decomposition (modeling both trend and seasonality and removing them from the model)

# Differencing

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
#ts_log.diff().plot()
plt.show()

ts_log_diff.dropna(inplace = True)
test_stationarity(ts_log_diff)

# We can see that the mean and std variations have small variations with time.
# Also, the Dickey-Fuller test statistic is less than the 10% critical value,
# thus the TS is stationary with 90% confidence. We can also take second or third
# order differences which might get even better results in certain applications.

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label = 'Original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonality')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual, label = 'Residual')
plt.legend(loc = 'best')
plt.show()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace = True)
test_stationarity(ts_log_decompose)


# The Dickey-Fuller test statistic is significantly lower than the 1% critical value. So this TS is very close to stationary.

#ACF plot
pd.plotting.autocorrelation_plot(ts_log_diff)
plt.show()

# Due to seasonality, at lag 12 autocorrelation is high and for every multiple of 12, autocorrelation will be high
# but will keep decreasing moving further.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.subplot(211)
plot_acf(ts_log_diff, ax=plt.gca())
plt.subplot(212)
plot_pacf(ts_log_diff, ax=plt.gca())
plt.show()


#Another method
#ACF and PACF plots
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0, linestyle = '--', color = 'gray') #Add a horizontal line across the axis at y = 0.
plt.axhline(y = -1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
plt.title('Autocorrelation Function')
plt.show()


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y = -1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color = 'gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout() #Automatically adjust subplot parameters to give specified padding.
plt.show()

from statsmodels.tsa.arima_model import ARIMA, ARMAResults

# AR Model
model = ARIMA(ts_log, order = (2,1,0))
results_AR = model.fit(disp = -1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color = 'red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues - ts_log_diff)**2)) #Residual sum of squares
plt.show()

# MA Model

model = ARIMA(ts_log, order= (0, 1, 2))
results_MA = model.fit(disp = -1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color= 'red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues - ts_log_diff)**2))
plt.show()

# Combined Model

model = ARIMA(ts_log, order = (2, 1, 2))
results_ARIMA = model.fit(disp = -1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color= 'red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - ts_log_diff)**2))
plt.show()

# Here we can see that the AR and MA models have almost the same RSS but combined is significantly better.

#Taking it back to original scale

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()

# This predicton is not that much good as it should be and root mean square error is also very high.

# Another method (using SARIMAX)
# 1. Parameter Selection for the ARIMA Time Series Model

import itertools
p = d = q = range(2)
pdq = list(itertools.product(p, d, q))      #Generate all different combinations of p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]       #Generate all different combinations of seasonal p, q and q triplet

# We can now use the triplets of parameters defined above to automate the process of training and
# evaluating ARIMA models on different combinations. In Statistics and Machine Learning,
# this process is known as grid search (or hyperparameter optimization) for model selection.

import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")           #Specify to ignore warning messages
AIC_df = pd.DataFrame({}, columns = ['param', 'param_seasonal', 'AIC'])

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(ts_log, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
            results = mod.fit()
            #print('ARIMA{}x{}-AIC:{}'.format(param, param_seasonal, results.aic))
            temp = pd.DataFrame([[param, param_seasonal, results.aic]], columns = ['param', 'param_seasonal', 'AIC'])
            AIC_df = AIC_df.append(temp, ignore_index = True)
            del temp
        except:
            continue




# 2. Fitting an ARIMA Time Series Model

min_aic = AIC_df.sort_values(by = 'AIC').iloc[0]    #Row with minimum AIC value
model = sm.tsa.statespace.SARIMAX(ts_log, order = min_aic.param, seasonal_order = min_aic.param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
results = model.fit()
print(results.summary())
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))    #Generate model diagnostics and investigate for any unusual behavior.
plt.show()

# Validating Forecasts

## SIMPLE & DYNAMIC FORECASTING

## SIMPLE FORECASTING

#Obtain values for forecasts of the time series from 1958
pred = results.get_prediction(start = pd.to_datetime('1958-01-01'), dynamic = False)
#Obtain associated confidence intervals for forecasts of the time series
pred_ci = pred.conf_int()
print(pred_ci.head())

#Plot the forecasted values with historical data
ax = ts_log['1953':].plot(label = 'observed')
pred.predicted_mean.plot(ax = ax, label = 'One-step ahead forecast', alpha = 0.7)
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color = 'k', alpha = .2)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1958-01-01'), ts_log.index[-1], alpha = .1, zorder = -1)
plt.xlabel('Year')
plt.ylabel('ts_log')
plt.title('Simple')
plt.legend()
plt.show()

print(pred.predicted_mean.head())

ts_log_forecasted = pred.predicted_mean     #From 1958 to 1960 (validation process)
ts_log_original = ts_log['1958-01-01':]
mse = ((ts_log_forecasted - ts_log_original) ** 2).mean()  #Mean square error
print('Mean Squared Error of forecast : {}'.format(round(mse,3)))

# Mean square error is almost zero. This means our prediction is very accurate.
#
# However, a better representation of our true predictive power can be obtained using dynamic forecasts.
# In this case, we only use information from the time series up to a certain point, and after that,
# forecasts are generated using values from previous forecasted time points. Let's try with computing the dynamic forecast.

## DYNAMIC FORECASTING

pred_dynamic = results.get_prediction(start = pd.to_datetime('1958-01-01'), dynamic = True, full_results = True)
pred_dynamic_ci = pred_dynamic.conf_int()
print(pred_dynamic_ci.head())

ax = ts_log['1953':].plot(label = 'observed')
pred_dynamic.predicted_mean.plot(ax = ax, label = 'Dynamic Forecast')
ax.fill_between(pred_dynamic_ci.index, pred_dynamic_ci.iloc[:, 0], pred_dynamic_ci.iloc[:, 1], color = 'k', alpha = .2)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1958-01-01'), ts_log.index[-1], alpha = .1, zorder = -1)
plt.xlabel('Year')
plt.ylabel('ts_log')
plt.title('Dynamic')
plt.legend()

ts_log_forecasted_dyna = pred_dynamic.predicted_mean
ts_log_truevalue = ts_log['1958-01-01':]
mse = ((ts_log_forecasted_dyna - ts_log_truevalue) ** 2).mean()
print('Mean Squared Error of forecast : {}'.format(round(mse, 3)))

# The predicted values obtained from the dynamic forecasts yield an MSE of .008.
# This is slightly higher than the one-step ahead, which is to be expected given
# that we are relying on less historical data from the time series.
#
# Both the one-step ahead and dynamic forecasts confirm that this time series model is valid.
# However, much of the interest around time series forecasting is the ability to forecast future values way ahead in time.


# 4. Producing and Visualizing Forecasts¶

pred_uc = results.get_forecast(steps=100)     #Get forecast 100 steps ahead in future (ts_log)
pred_ci = pred_uc.conf_int()                  #Get confidence intervals of forecasts (ts_log)
print(pred_ci.head())

ax = ts_log['1955':].plot(label='Observed')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Year')
ax.set_ylabel('ts_log')
plt.legend(loc = 'upper left')
plt.show()

#Take exponential function
pred_uc = np.exp(pred_uc.predicted_mean)
pred_ci = np.exp(pred_ci)

#Plot original data prediction
ax = ts['1955':].plot(label='Observed')
pred_uc.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Year')
ax.set_ylabel('ts (Passengers)')
plt.legend(loc = 'upper left')
plt.show()

# References:
# Time Series Analysis in Python with statsmodels - Wes McKinney, Josey Perktold, Skipper Seabold
# Analytics Vidhya article on Time Series Forecasting.
# DigitalOcean article on Time Series Forecasting.





