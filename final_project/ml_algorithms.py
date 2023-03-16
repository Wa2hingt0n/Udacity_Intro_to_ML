# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:10:13 2023

@author: Washington
"""

""" Defines Machine Learning algorithms to be applied to the poi identifier
"""

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV


#############################################################
# GAUSSIAN NAIVE-bAYES
#############################################################
def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    
    return Pipeline([('Scaler', MinMaxScaler()),
                     ('clf_NB', GaussianNB())])



def svc():
    """Initializes a support vector machine
    
    
    Returns: A support vector classifier
    """
    from sklearn.svm import SVC
    
    # Hyperparameter test set for optimization
    param_grid = {
              'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
    
    return Pipeline([('Scaler', MinMaxScaler()), ('PCA', PCA()),
                      ('clf_SVC', GridSearchCV(SVC(kernel='rbf',
                                       class_weight='balanced'),
                                   param_grid))])


#############################################################
# DECISION TREE CLASSIFIER
#############################################################
def decision_tree(scaler=MinMaxScaler()):
    """Initializes a decision tree classifier
    
    
    Returns: A decision tree classifer
    """
    from sklearn.tree import DecisionTreeClassifier as DTC
    
    # Test parameters for optimization
    # param_grid_DTC = {'criterion': ['gini', 'entropy'],
    #                   'max_features': [2, 3, 5, 8, 10],
    #                   'min_samples_split': [2, 3, 5, 8, 13],
    #                   'splitter': ['best', 'random']}
    
    # ('Scaler', MinMaxScaler()), ('PCA', PCA()),
    
    return Pipeline([('Scaler', scaler), ('clf_DTC',
                      DTC(criterion='gini', max_features=2, min_samples_split=2))])

#############################################################
# RANDOM FOREST CLASSIFIER
#############################################################
def random_forest():
    """Creates classifier based of the Random Forest ensemble ML algorithm
    
    
    Returns: A random forest ensemble classifier
    """
    from sklearn.ensemble import RandomForestClassifier as RFC
    
    return Pipeline([('Scaler', MinMaxScaler()), 
                    ('clf_RFC',
                      RFC())])

#############################################################
# K-NEAREST NEIGHBORS CLASSIFIER
#############################################################
def k_nearest_neighbors():
    """Creates a classifier based on the K Nearest Neighbors ML algorithm
    
    
    Returns: A KNN classifier with tuned parameters
    """
    from sklearn.neighbors import KNeighborsClassifier as KNNC
    
    # param_grid_KNNC = {'p': [3, 1, 2],
                        # 'n_neighbors': list(range(5, 9)),
                        # }
    # n_neighbors=6, weights='distance', n_jobs=1, p=1, , 2
    
    # searched_grid = GridSearchCV(KNNC(weights='distance', n_jobs=1), param_grid_KNNC, cv=5)
    
    # clf = Pipeline([('Scaler', MinMaxScaler()), 
    #                 ('clf_KNCC',
    #                   searched_grid)])
    
    return Pipeline([('Scaler', StandardScaler()), 
                    ('clf_KNCC', KNNC(n_neighbors=6, weights='distance',
                                      n_jobs=1, p=1))])

