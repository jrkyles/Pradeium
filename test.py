



import base64
import io
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from flask import Flask, redirect, render_template, request, session, url_for, Response, send_file
import os
import seaborn as sn

df = pd.read_csv('CRE_Credit_Rating.csv')
pd.set_option('display.max_columns', None)
'''print(df.columns)

print(df.info())

string_columns = [ ' NOI ',' IE (UNADJUSTED) ', ' Interest Expense ', ' Total Debt ', ' Total Equity ', ' Cash ', ' Total Assets ']
for i in string_columns:
 df[i] = df[i].str.replace(',', '', regex=False).astype(float)

print(df.info())
Net_Operating_Income = ("$" + str(round(df[' NOI '].median(), 2)))
Interest_Coverage_Ratio = (str(round(df['Interest_Expense_Coverage_Ratio'].median(), 2)))
Leverage_Ratio = ( str(round(df['Net_Debt_Leverage_Ratio'].median(), 2)))

#print(Interest_Coverage_Ratio)

#print(df['Rating'].unique())
sn.countplot(data=df, x='Rating')
#plt.show()'''
ETF_specific_columns = ['Sticker', 'Full name', 'Focus', 'Numerical_Rating', 'Rating','Residential', 'Office',
                            'Retail', 'Industrial']
features = []
for i in df.columns:
        if i not in ETF_specific_columns:
            features.append(i)
print(features)
print(df.info())

count = 0
fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # Create grid
axes = axes.flatten()
for count, i in enumerate(features):
        ax = axes[count]
        sn.stripplot(data=df, x=i, y='Rating', hue='Rating', alpha=1, ax=ax,
                     palette={'A-': 'blue',
                              'BBB+': 'green',
                              'BBB': 'orange',
                              'BBB-': 'indigo',
                              'BB+': 'black',
                              'BB': 'yellow',
                              'B': 'red'}, hue_order=['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'B'], legend=False,
                     order=['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'B'])
        ax.set_title(str(i))
plt.title('Credit Rating Correlation')
'''max_x = df[i].max()
    #print(max_x)
     plt.xticks(np.linspace(0, max_x, 6))  # 6 evenly spaced x-ticks

    # Set y-ticks
     max_y = feature_range.max()
     plt.yticks(np.linspace(0, max_y, 6))'''
handles, labels = ax.get_legend_handles_labels()

    # Place the legend outside the grid
fig.legend(handles, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
#plt.show()

ETF_specific_columns = ['Sticker', 'Full name', 'Focus', 'Numerical_Rating']
for i in ETF_specific_columns:
    del df[i]
string_columns = [' NOI ', ' IE (UNADJUSTED) ', ' Interest Expense ', ' Total Debt ', ' Total Equity ', ' Cash ',
                   ' Total Assets ']
for i in string_columns:
    df[i] = df[i].str.replace(',', '', regex=False).astype(float)

y = df['Rating']
del df['Rating']
filtered_features = [' NOI ', ' Interest Expense ', ' Total Debt ', ' Total Equity ', ' Cash ', ' Total Assets ',
                     'Interest_Expense_Coverage_Ratio','Total_Leverage_Ratio','Debt_to_Equity_Ratio',
                     'EBITDA_Asset_Ratio', 'EBITDA_Equity_Ratio']
for i in df.columns:
    if i not in filtered_features:
        del df[i]
x= df
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
#gives the method the mean and stdv of EACH column in the df

x_normalized = scaler.transform(x)
#uses the mean and stdv found in .fit() to shift the data how we saw above
#returns as a numpy array

x_normalized = pd.DataFrame(data=scaler.transform(x), columns=x.columns)
#changes the numpy array into a dataframe
print(x.columns)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_normalized,y)
#predicting the probability of loan approval based on sample data
#y_pred = model.predict_proba(sample_df)

for i in filtered_features:
    print(str(i))
    print(str(round(df[i].min(),2)) + ' - ' + str(round(df[i].max(),2)))
    print(str(round(df[i].median(),2)))
    print()

num = random.randint(1,47)
print(num)
sample_df = df.iloc[[num]]
print(sample_df)
sample_df_scaled = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)
y_pred = model.predict_proba(sample_df_scaled)
Y_predict = model.predict(sample_df_scaled)

print(model.classes_)

print(y_pred)
print(Y_predict)
proba_df = pd.DataFrame({
    'Category': model.classes_,
    'Probability': y_pred[0]  # first and only row of probabilities
})
idx = np.argmax(y_pred[0])

# 2) extract the class name and its probability
predicted_category = model.classes_[idx]
predicted_probability = y_pred[0][idx]
print(predicted_category)
print(predicted_probability)
# --- Plot ---
plt.figure(figsize=(10,6))
sn.barplot(data=proba_df, x='Category', y='Probability', palette='muted', order=['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'B'])

plt.title('Probability Distribution Across Credit Categories')
plt.ylabel('Predicted Probability')
plt.xlabel('Credit Rating Category')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

