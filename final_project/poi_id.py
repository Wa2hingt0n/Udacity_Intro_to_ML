#!/usr/bin/python

import os
import pickle
import sys
import ml_algorithms

sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from cross_validation import stratified_shuffle_split
from sklearn.model_selection import train_test_split


##########################################################################
######################## FEATURE SELECTION ###############################
##########################################################################
# List of all features
all_features_list = ['poi', 'salary', 'total_payments', 'bonus',
                  'deferred_income', 'total_stock_value', 'expenses',
                  'from_poi_to_this_person', 'exercised_stock_options',
                  'from_this_person_to_poi', 'long_term_incentive',
                  'shared_receipt_with_poi', 'restricted_stock']

# Optimized list of features to be used for training
features_list_1 = ['poi', 'bonus', 'total_stock_value',
                'exercised_stock_options']

# Second best performihg training features list
features_list_2 = ['poi', 'bonus', 'total_stock_value',
                   'total_payments', 'exercised_stock_options']

##########################################################################
##########################################################################


### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Remove outliers
data_dict.pop("TOTAL")

# Creation of new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_1, sort_keys = True)
labels, features = targetFeatureSplit(data)


clf = ml_algorithms.k_nearest_neighbors()

features_train, labels_train, features_test, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    
# clf.fit(features_train, labels_train)

# print(searched_grid.best_estimator_.get_params())
# #if isinstance(clf, DTC):
#     from sklearn import tree
#     tree.plot_tree(clf)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list_1)