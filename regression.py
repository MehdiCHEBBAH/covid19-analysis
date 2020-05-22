#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:42:47 2020

@author: mehdi
"""
# Importing labraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./datasets/countries-aggregated.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.query('Country == "Algeria"')
dataset = dataset.loc[dataset['Date'] > '2020-02-24']
dataset['Relative_Date'] = (dataset['Date'] - pd.to_datetime('2020-02-24')).dt.total_seconds() / (60 * 60 * 24)

dataset['log_confirmed'] = np.log(dataset['Confirmed'])

ax = plt.gca()
dataset.plot(kind='scatter',x='Relative_Date',y='log_confirmed',ax=ax)
plt.title('Total cases')
plt.show()


X = dataset.iloc[:, 5:6].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, np.exp(y_train), color = 'red')
plt.scatter(X_train, np.exp(regressor.predict(X_train)), color = 'blue')
plt.title('Training set')
plt.xlabel('Days since the first case')
plt.ylabel('cases')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, (y_test), color = 'red')
plt.plot(X_train, (regressor.predict(X_train)), color = 'blue')
plt.title('Confirmed cases in Algeria')
plt.xlabel('Days since the first case')
plt.ylabel('log cases')
plt.show()

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

####################################################
dataset['log_deaths'] = np.log(dataset['Deaths'])
dataset = dataset.replace([np.inf, -np.inf], 0)
ax = plt.gca()
dataset.plot(kind='scatter',x='Relative_Date',y='log_deaths',ax=ax)
plt.show()


# Time Based regression
X = dataset.iloc[:, 5:6].values
y = dataset.iloc[:, 7].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, (y_test), color = 'red')
plt.plot(X_train, (regressor.predict(X_train)), color = 'blue')
plt.show()

plt.scatter(X_test, np.exp(y_test), color = 'red')
plt.scatter(X_train, np.exp(regressor.predict(X_train)), color = 'blue')
plt.show()

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# Confirmed cases based regression
ax = plt.gca()
dataset.plot(kind='scatter',x='Confirmed',y='Deaths',ax=ax, color='red')
plt.show()

X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, (y_test), color = 'red')
plt.plot(X_train, (regressor.predict(X_train)), color = 'blue')
plt.show()

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# Time and confirmed cases based regression
X = dataset.iloc[:, [2, 5]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(y_test, y_pred, color = 'red')
plt.show()


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
