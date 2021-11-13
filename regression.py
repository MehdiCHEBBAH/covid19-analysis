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
dataset = pd.read_csv('./datasets/countries-aggregate.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.query('Country == "Algeria"')
dataset = dataset.loc[dataset['Date'] > '2020-02-24']
dataset['Relative_Date'] = (dataset['Date'] - pd.to_datetime('2020-02-24')).dt.total_seconds() / (60 * 60 * 24)

ax = plt.gca()
dataset.plot(kind='scatter',x='Relative_Date',y='Confirmed',ax=ax)
dataset.plot(kind='scatter',x='Relative_Date',y='Deaths', color='red', ax=ax)
plt.title('Total cases')
plt.xticks(rotation=45)
plt.show()


###################################################
ax = plt.gca()
dataset.plot(kind='scatter',x='Relative_Date',y='Deaths',ax=ax)
plt.show()


# Time Based regression
X = dataset.iloc[:, 5:6].values
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
X = dataset.iloc[:, [2,5]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test[:,0], (y_test), color = 'red')
plt.scatter(X_train[:,0], (regressor.predict(X_train)), color = 'blue')
plt.show()


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
