# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:43:58 2017

@author: hp
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import os.path
import re
import seaborn as sns
from matplotlib import cm as cm
from sklearn.ensemble import RandomForestClassifier


path = 'D:/mariem/Academics/master/cours/Mini-Project'
#Read Data
df=pd.read_csv(os.path.join(path, 'input','prescriber-info.csv'))
opioids = pd.read_csv(os.path.join(path, 'input','opioids.csv'))


def correlation_matrix(df):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax.grid(True)
    plt.title('Drug-related Features Correlation')
    labels= df.columns
    ax.set_xticklabels(labels,fontsize=7)
    ax.set_yticklabels(labels,fontsize=7)
    fig.colorbar(cax, ticks=[-1,-.9,-.8,-.7,-.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    
    
    
if __name__=='__main__':
               
    #summary statistics of our dataset
    stats = df.describe()
#    df_no_opioids.hist(column='Gender')
#    df_no_opioids.hist(column='State')
#    df_no_opioids.hist(column='Specialty')
    g=plt.figure()
    sns.countplot(x="State", data= df, order = df['State'].value_counts().index,
                  palette="Greens_d")
    plt.xticks(rotation=85, fontsize = 8)
    f=plt.figure()
    sns.countplot(x="Specialty", data= df, order = df['Specialty'].value_counts().index,
                  palette="Greens_d")
    plt.xticks(rotation=85, fontsize = 8)
    h =  plt.figure()
    sns.countplot(x="Gender", data= df, order = df['Gender'].value_counts().index,
                  palette="Greens_d")
    k = plt.figure()
    sns.countplot(x="Opioid.Prescriber", data= df, order = df['Opioid.Prescriber'].value_counts().index, 
                  palette="RdBu")
    
    #replace the least frequent states by a new category "other"
    freq_state = df['State'].value_counts()
    least_frequent=list(freq_state[freq_state<10].index)
    df_copy = df
    for lf in least_frequent:
        df_copy['State'].loc[df_copy['State']==lf] = 'other'
   
    #replace hyphens and spaces with periods to match the dataset
    opioids_names = list(re.sub(r'[-\s]','.',x) for x in opioids.values[:,0])
    #determe how many/ what opioids are mentioned in the data
    df_cols = list(df.columns)
    common_drugs = [name for name in df_cols if name in opioids_names]
    print("there are %d opioids cited among the drugs"%len(common_drugs))
    
    #removing opioid prescriptions from the data, otherwise we'll be cheating!
    df_cols_no_opioids = [col for col in df_cols if not col in opioids_names]
    df_no_opioids = df_copy[df_cols_no_opioids]
    #factorizing categorical variables
    uniques = dict()
    Categorical=['Gender','State','Credentials','Specialty']
    for col in Categorical:
        uniques[col] = df_no_opioids[col].unique()
        df_no_opioids[col]=pd.factorize(df_no_opioids[col])[0]
    
    #studying correlation of drug-related features
    corr = (df_no_opioids.iloc[:,5:244]).corr()
    correlation_matrix(df_no_opioids.iloc[:,5:244])
    #split our dataset into train/validation/test sets
    y = df_no_opioids['Opioid.Prescriber']
    X_train, X_test, y_train, y_test = train_test_split(np.array(df_no_opioids)[:,:244],
                                np.array(y), test_size = 0.4, random_state=42)    
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5,
                                                    random_state=42)
    
    #trying out some classification models
    clf = RandomForestClassifier(n_estimators=150, criterion ='entropy', 
                                 random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mean_acc_score = clf.score(X_test, y_test)
    