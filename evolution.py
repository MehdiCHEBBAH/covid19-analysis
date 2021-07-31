#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:01:41 2020

@author: mehdi
"""

import pandas as pd
import matplotlib.pyplot as plt

################ WORLD WIDE ###################
df = pd.read_csv('./datasets/worldwide-aggregate.csv')

ax = plt.gca()
df.plot(kind='line',x='Date',y='Confirmed',ax=ax)
df.plot(kind='line',x='Date',y='Recovered', color='red', ax=ax)
df.plot(kind='line',x='Date',y='Deaths', color='green', ax=ax)
plt.title('Total cases')
plt.xticks(rotation=45)
plt.show()


ax = plt.gca()
df.plot(kind='line',x='Date',y='Increase rate',ax=ax)
plt.title("Le taux de d'augmantation")
plt.xticks(rotation=45)
plt.show()


############### KEY COUNTRIES #####################
df = pd.read_csv('./datasets/key-countries-pivoted.csv')

fig, ax_lst = plt.subplots(4, 2, figsize=(7, 7), sharex='col', sharey='row')

for i in range(1, 9):
    ax = ax_lst.ravel()[i-1]
    ax.xaxis.get_major_locator().set_params(nbins=3)
    ax.yaxis.get_major_locator().set_params(nbins=2)
    df.plot(kind='line',x='Date',y=df.columns[i],ax=ax)

fig.tight_layout(pad=0, h_pad=.1, w_pad=.1)

################### CONTINENTS ######################
continents = ['Africa', 'South America', 'Europe', 'North America', 'Asia']
df = pd.read_csv('./datasets/continents-cases-deaths.csv')

fig, ax_lst = plt.subplots(3,2 , figsize=(8, 7), sharex='col', sharey='row')
fig.delaxes(ax_lst[2,1])

for i in range(5):
    ax = ax_lst.ravel()[i]
    ax.xaxis.get_major_locator().set_params(nbins=3)
    ax.yaxis.get_major_locator().set_params(nbins=2)
    ax.title.set_text(continents[i])
    df.query('Entity == "' + continents[i] + '"').plot(kind='line',x='Date',y='Total confirmed deaths (deaths)', color='red', ax=ax)
    df.query('Entity == "' + continents[i] + '"').plot(kind='line',x='Date',y='Total confirmed cases (cases)', color='blue', ax=ax)    

fig.tight_layout(pad=0, h_pad=.1, w_pad=.1)
fig.show()

###################### ALGERIA ######################
df = pd.read_csv('./datasets/countries-aggregated.csv')
df['Date'] = pd.to_datetime(df['Date'])

ax = plt.gca()
df_algeria = df.query('Country == "Algeria"')
df_algeria.loc[df_algeria['Date'] > '2020-02-15'].plot(kind='line',x='Date',y='Confirmed',ax=ax)
df_algeria.loc[df_algeria['Date'] > '2020-02-15'].plot(kind='line',x='Date',y='Recovered', color='green', ax=ax)
df_algeria.loc[df_algeria['Date'] > '2020-02-15'].plot(kind='line',x='Date',y='Deaths', color='red', ax=ax)
plt.title('Total cases')
plt.xticks(rotation=45)
plt.show()


df = pd.read_csv('./datasets/COVID-19-geographic-disbtribution-worldwide.csv')
df['dateRep'] = pd.to_datetime(df['dateRep'], dayfirst=True)

ax = plt.gca()
df_algeria = df.query('countriesAndTerritories == "Algeria"')
df_algeria.loc[df_algeria['dateRep'] > '2020-02-15'].plot(kind='line',x='dateRep',y='cases',ax=ax)
df_algeria.loc[df_algeria['dateRep'] > '2020-02-15'].plot(kind='line',x='dateRep',y='deaths', color='red', ax=ax)
plt.title('Daily cases')
ax.set_xlabel("Date")
plt.xticks(rotation=45)
plt.show()


df = pd.read_csv('./datasets/countries-aggregated.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.loc[df['Date'] > '2020-03-09']

ax = plt.gca()
df.query('Country == "Algeria"').plot(kind='line',label='Algeria', x='Date',y='Confirmed',ax=ax, marker='', color='red', linewidth=4, alpha=0.9)
df.query('Country == "Egypt"').plot(kind='line',label='Egypt',x='Date',y='Confirmed', ax=ax, marker='', color='orange', linewidth=2, alpha=0.5)
df.query('Country == "Morocco"').plot(kind='line',label='Morocco',x='Date',y='Confirmed', ax=ax, marker='', color='green', linewidth=2, alpha=0.5)
df.query('Country == "Tunisia"').plot(kind='line',label='Tunisia',x='Date',y='Confirmed', ax=ax, marker='', color='gray', linewidth=2, alpha=0.5)
df.query('Country == "Saudi Arabia"').plot(kind='line',label='Saudi Arabia',x='Date',y='Confirmed', ax=ax, marker='', color='blue', linewidth=2, alpha=0.5)
plt.title('Total cases')
plt.xticks(rotation=45)
plt.show()

