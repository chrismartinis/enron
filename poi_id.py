#!/usr/bin/python

# import sys
import pickle
import pandas as pd
import numpy as np
# sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# defining the initial features list and removing features that could hinder our futher processing
features_list = data_dict[next(iter(data_dict))].keys()
print '\n\nInitial data points:', len(data_dict)
print 'Initial features list length:', len(features_list)
features_list.remove('poi')
features_list.remove('email_address')
features_list.remove('total_payments')
features_list.remove('total_stock_value')
features_list.remove('other')
# print len(features_list)

# converting data dictionary to dataframe for easy processing
# e.g. replacing NaN with np.nan values, and removing certain features
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace(to_replace='NaN', value=np.nan, inplace=True)

# remove features with more than 50% NaN entries
# add 'poi' to the features list afterwards - we need this for future processing
for feature in features_list:
    if df[feature].isnull().sum() > df.shape[0] * 0.5:
        features_list.remove(feature)
features_list = ['poi'] + features_list
# print features_list # visualise the features list


### Task 2: Remove outliers
# features = ["salary", "bonus"]
# data = featureFormat(data_dict, features)
# for key, value in data_dict.items():
#     if value['bonus'] == data.max():
#         print key

### find persons with all features set to NaN values
# for key in data_dict:
#     if df.loc[key].isnull().sum() == df.shape[1] - 1:
#         print key

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART E EUGENE', 0)

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Task 3: Create new feature(s)
for person in my_dataset:
    if my_dataset[person]['from_this_person_to_poi'] != 'NaN' and my_dataset[person]['to_messages'] != 'NaN':
        my_dataset[person]['to_poi_prop'] = float(my_dataset[person]['from_this_person_to_poi']) / float(my_dataset[person]['to_messages'])
    else:
        my_dataset[person]['to_poi_prop'] = 'NaN'

for person in my_dataset:
    if my_dataset[person]['from_poi_to_this_person'] != 'NaN' and my_dataset[person]['from_messages'] != 'NaN':
        my_dataset[person]['from_poi_prop'] = float(my_dataset[person]['from_poi_to_this_person']) / float(my_dataset[person]['from_messages'])
    else:
        my_dataset[person]['from_poi_prop'] = 'NaN'

features_list += ['to_poi_prop', 'from_poi_prop']
print '\n\n', features_list # visualise the features list
print '\nFeatures list length after trimming:', len(features_list), '\n\n'


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### combined with ...

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# create function to use in the following step
def select_clf(n_features, features, labels):

    clf_list = []

    ### Select/visualise feature scores (k highest)
    select_k = SelectKBest(f_classif, k=n_features)
    features = select_k.fit_transform(features, labels)
    scores = select_k.scores_
    indices = select_k.get_support(indices=True)
    # sorted_scores = np.sort(scores)[::-1]
    # print sorted_scores

    # split data into train and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    tree = DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_split': [2,3,4,5,6,7],
                'random_state': [0,10,42]}
    clf_tree = GridSearchCV(tree, parameters, scoring='f1')

    nb = GaussianNB() # no parameters available for tuning

    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [1,3,5,7,9],
                'weights': ['uniform', 'distance']}
    clf_knn = GridSearchCV(knn, parameters, scoring='f1')


    svm = SVC()
    parameters = {'kernel': ['rbf'],
                  'C': [1, 10, 100, 1000, 10000, 100000],
                  'random_state': [0,10,42]}
    clf_svm = GridSearchCV(svm, parameters, scoring='f1')


    # fit classifiers - predict on test features
    target_names = ['non_poi', 'poi']

    clf_tree.fit(features_train, labels_train)
    clf_tree = clf_tree.best_estimator_
    pred_tree = clf_tree.predict(features_test)
    clf_list.append([metrics.f1_score(labels_test, pred_tree), metrics.accuracy_score(labels_test, pred_tree), n_features, scores, indices, clf_tree,
                                    metrics.classification_report(labels_test, pred_tree, target_names = target_names, output_dict=True)])

    nb.fit(features_train, labels_train)
    pred_nb = nb.predict(features_test)
    clf_list.append([metrics.f1_score(labels_test, pred_nb), metrics.accuracy_score(labels_test, pred_nb), n_features, scores, indices, nb,
                                    metrics.classification_report(labels_test, pred_nb, target_names = target_names, output_dict=True)])

    scaler = MinMaxScaler()
    scaler.fit_transform(features_train)
    scaler.fit(features_test)

    clf_knn.fit(features_train, labels_train)
    clf_knn = clf_knn.best_estimator_
    pred_knn = clf_knn.predict(features_test)
    clf_list.append([metrics.f1_score(labels_test, pred_knn), metrics.accuracy_score(labels_test, pred_knn), n_features, scores, indices, clf_knn,
                                    metrics.classification_report(labels_test, pred_knn, target_names = target_names, output_dict=True)])

    clf_svm.fit(features_train, labels_train)
    clf_svm = clf_svm.best_estimator_
    pred_svm = clf_svm.predict(features_test)
    clf_list.append([metrics.f1_score(labels_test, pred_svm), metrics.accuracy_score(labels_test, pred_svm), n_features, scores, indices, clf_svm,
                                    metrics.classification_report(labels_test, pred_svm, target_names = target_names, output_dict=True)])

    # sort list according to f1_score results
    order_clf_list = sorted(clf_list, key=lambda x: x[0], reverse=True)
    return order_clf_list[0] # return topmost item of list [::-1]

# loop through features to get the best features/classifier combination
results_list = []
for k in range(1, 7):
    results_list.append(select_clf(k, features, labels))
order_results_list = sorted(results_list, key=lambda x: x[0], reverse=True)

clf = order_results_list[0][5]
number_of_features = order_results_list[0][2]

print '\nClassifier used:', clf
print '\nNumber of features used:', number_of_features

features = features_list[1:] # features list without 'poi'
scores = order_results_list[0][3]
ft_chosen_indices = order_results_list[0][4] # array of indices returned by KBest (chosen features)
used_features_list = []
final_features_list = []
for idx in ft_chosen_indices:
    used_features_list.append([features[idx], scores[idx]])
    final_features_list.append(features[idx])

# print '\nScores:\n', scores
print "The features (incl. scores) used are:", sorted(used_features_list, key=lambda x: x[1], reverse=True)

print '\nClassification report:\n'
df = pd.DataFrame.from_dict(order_results_list[0][6]).transpose()
df.to_csv('ClassificationReportExport.csv', sep=',')
print df

final_features_list = ['poi'] + final_features_list


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, final_features_list)
print '\n\nTest classifier result:\n'
test_classifier(clf, my_dataset, final_features_list)
