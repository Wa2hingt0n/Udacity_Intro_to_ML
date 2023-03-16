# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:08:51 2023

@author: Washington
"""

""" Features and labels' train and test split."""
        
def stratified_shuffle_split(features, labels):
    from sklearn.model_selection import StratifiedShuffleSplit
    
    cv = StratifiedShuffleSplit(random_state=42)
    
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        
        for i in train_idx:
            features_train.append(features[i])
            labels_train.append(features[i])
        for j in test_idx:
            features_test.append(features[j])
            labels_test.append(labels[j])
            
    return features_train, labels_train, features_test, labels_test
    
    