#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 06:41:21 2020

@author: mehdi
"""

import pandas as pds
import matplotlib.pyplot as plt

df = pds.read_csv('./datasets/detailed-cases.csv')
boxplot = df.boxplot(column=['age'])


fig, ax = plt.subplots()
ax.boxplot([df.query('gender == "male"')['age'], df.query('gender == "female"')['age']])
ax.set_xticklabels(['male', 'female'])


fig, ax = plt.subplots()
ax.boxplot([df.query('gender == "male" and death == "1"')['age'], df.query('gender == "female" and death == "1"')['age']])
ax.set_xticklabels(['male', 'female'])


df.death.groupby(df.death).count().plot(kind='pie')
plt.axis('equal')
plt.show()


from wordcloud import WordCloud
separateur = ' '
df = df.fillna(value={'symptom': ''})
text = separateur.join(df['symptom'])
text = text.replace(",", "").strip()
word_cloud = WordCloud().generate(text)
image = word_cloud.to_image()
image.show()
