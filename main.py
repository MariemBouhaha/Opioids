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
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, recall_score
from time import time
import itertools

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
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    
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
    
    #split our dataset into train/test sets
    y = df_no_opioids['Opioid.Prescriber']
    X_train, X_test, y_train, y_test = train_test_split(np.array(df_no_opioids)[:,:244],
                                np.array(y), test_size = 0.2, random_state=42)    
    
    #trying out some classification models
    """clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, criterion ='entropy', 
                                 random_state=42)
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
    
    param_grid = { 
    'n_estimators': [100, 200, 500],
    'min_samples_leaf': [10,20,50]
    }
    
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 10)
    start = time()
    grid_search.fit(X_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
    print(grid_search.best_params_)#10/100
    """
    opt_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, criterion ='entropy', 
                                 random_state=42)
    opt_clf.fit(X_train, y_train)
    y_pred = opt_clf.predict(X_test)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    class_names = ["0","1"]
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    plt.show()
    
    #compute AUC score
    auc_score = roc_auc_score(y_test, y_pred)
    print("AUC score: %.2f"%auc_score)
    
    #compute balanced accuracy score
    bal_accuracy1 =recall_score(y_test,y_pred, average='macro')
    mean_acc_score = accuracy_score(y_test,y_pred)
    
    fpr1, tpr1, thresholds1 = roc_curve(y_test, opt_clf.predict_proba(X_test)[:,1])
   
    #PLOT ROC curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % auc_score)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    #analyzing feature importance
    feature_import = opt_clf.feature_importances_
    feature_import = np.c_[np.array(df_cols_no_opioids)[:244],feature_import]
    feature_import = np.sort(feature_import, 0)
    