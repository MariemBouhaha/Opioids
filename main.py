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
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, recall_score, precision_recall_curve, average_precision_score
from time import time
import itertools
from sklearn.model_selection import validation_curve, learning_curve

path = 'D:/mariem/Academics/master/cours/Mini-Project'
#Read Data
df=pd.read_csv(os.path.join(path, 'input','prescriber-info.csv'))
opioids = pd.read_csv(os.path.join(path, 'input','opioids.csv'))
overdoses = pd.read_csv(os.path.join(path, 'input','overdoses.csv'))

overdoses = overdoses.apply(lambda x: x.str.replace(',',''))
overdoses.Deaths = overdoses.Deaths.astype(int)
overdoses.Population = overdoses.Population.astype(int)


def OverdoseFatalities(overdoses):
    deathRatio = (overdoses.Deaths/overdoses.Population)*100
    temp = pd.DataFrame()
    temp['Abbrev'] = overdoses['Abbrev']
    temp['deathRatio'] = deathRatio
    temp = temp.sort_values('deathRatio')
    sns.barplot(temp.Abbrev, temp.deathRatio, color='g')
    plt.title('Opioid Overdose Fatalities Percentage per State')
    plt.xlabel('State')
    plt.xticks(rotation=80)
    plt.ylabel('Overdose Death Percentage')
    
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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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
    
    #Most Comon specialies to opioid prescribers
    opSpec = df['Specialty'].loc[df['Opioid.Prescriber']==1]
    most_freq_opSpec = opSpec.value_counts()
    
    #replace the least frequent states and specialties by a new category "other"
    df_copy = df
    freq_state = df['State'].value_counts()
    least_frequent_st=list(freq_state[freq_state<10].index)
    for lf in least_frequent_st:
        df_copy['State'].loc[df_copy['State']==lf] = 'other'
        
    freq_specialty = df['Specialty'].value_counts()
    least_frequent_sp=list(freq_specialty[freq_specialty<50].index)
    for lf in least_frequent_sp:
        df_copy['Specialty'].loc[df_copy['Specialty']==lf] = 'other'
        
    #replace hyphens and spaces with periods to match the dataset
    opioids_names = list(re.sub(r'[-\s]','.',x) for x in opioids.values[:,0])
    #determe how many/ what opioids are mentioned in the data
    df_cols = list(df.columns)
    common_drugs = [name for name in df_cols if name in opioids_names]
    print("there are %d opioids cited among the drugs"%len(common_drugs))
    
    #removing opioid prescriptions from the data, otherwise we'll be cheating!
    df_cols_no_opioids = [col for col in df_cols if not col in opioids_names]
    df_no_opioids = df_copy[df_cols_no_opioids]
    #removing 0 columns
    df_no_opioids = df_no_opioids.loc[:, (df_no_opioids != 0).any(axis=0)]
    
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
    X=np.array(df_no_opioids)[:,:244]
    y = df_no_opioids['Opioid.Prescriber']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                np.array(y), test_size = 0.2, random_state=42)    
    
    #trying out some classification models
    clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, criterion ='entropy', 
                                 random_state=42)
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
    """
    param_grid = { 
    'n_estimators': [50, 100, 150],
    'min_samples_leaf': [5,10,15],
    'criterion': ['entropy', 'gini']
    }
    
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 10)
    start = time()
    grid_search.fit(X_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
    print(grid_search.best_params_)#10/100
    """
    opt_clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, 
                                     criterion ='gini',  random_state=42)
    opt_clf.fit(X_train, y_train)
    y_pred = opt_clf.predict(X_test)
    y_score = opt_clf.predict_proba(X_test)[:,1]
    
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
    #precision-recall curve
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    #learning curves
    
    title = "Learning Curves (Random Forest)"
    estimator = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, 
                                     criterion ='gini',  random_state=42)
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=10)

    #compute AUC score
    auc_score = roc_auc_score(y_test, y_pred)
    print("AUC score: %.2f"%auc_score)
    
    #compute balanced accuracy score
    bal_accuracy1 =recall_score(y_test,y_pred, average='macro')
    mean_acc_score = accuracy_score(y_test,y_pred)
    
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_score)
   
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
    FI = pd.DataFrame()
    FI['Feature']= list(df_no_opioids.columns)[:244]
    FI['Importance'] = feature_import
    FI.sort(columns='Importance', ascending=False, inplace=True)
    sns.barplot(FI['Feature'].iloc[:50], FI['Importance'].iloc[:50], color='r')
    plt.title('50 most important features according to RandomForestClassifier')
    plt.xlabel('Feature')
    plt.xticks(rotation=70)
    plt.ylabel('Feature Importance')
    
    