from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd
import csv
import os

#Read in data from file
dir_path = os.path.dirname(os.path.realpath(__file__))

with open (dir_path + "\\102authors_binary.csv") as file:
    dataset = pd.read_csv(file)

#Separate out feature names, and correct problem with feature names separated across columns
nheader = np.array(dataset.columns)
nh = list(nheader)
k = 1
while k < len(nh)-1:
    if nh[k][-1] != '\'' and nh[k+1][0] != '\'':
        nh[k] += nh[k+1]
        nh.pop(k+1)
    else:
        k += 1
nh.pop(-1)

#Drop empty columns and segregate labels
dataset.drop(dataset.columns[[i for i in range(-1, -44, -1)]], axis = 1, inplace=True)
nlabels = np.array(dataset[dataset.columns[-1]])
dataset.drop(dataset.columns[[-1]], axis = 1, inplace=True)
ndata = np.array(dataset)

#Create smaller dataset (10 programmers) for question 1.3
smallest_set = np.array(ndata[0:90])
smallest_labels = np.array(nlabels[0:90])

#Create medium sized dataset (50 programmers) for question 1.3
middle_set = np.array(ndata[0:450])
middle_labels = np.array(nlabels[0:450])

#Create train and test sets for primary dataset (102 programmers)
test_data = np.array(ndata[0])
for i in range(9, len(ndata), 9):
    test_data = np.vstack([test_data, ndata[i]])
train_data = np.delete(ndata, [i for i in range(0, len(ndata), 9)], axis=0)
test_labels = [nlabels[i] for i in range(0, len(nlabels), 9)]
train_labels = [nlabels[i] for i in range(0, len(nlabels)) if i % 9 != 0]

#Create train and test sets for smallest dataset (10 programmers)
stest_data = np.array(smallest_set[0])
for i in range(9, len(smallest_set), 9):
    stest_data = np.vstack([stest_data, smallest_set[i]])
strain_data = np.delete(smallest_set, [i for i in range(0, len(smallest_set), 9)], axis=0)
stest_labels = [smallest_labels[i] for i in range(0, len(smallest_labels), 9)]
strain_labels = [smallest_labels[i] for i in range(0, len(smallest_labels)) if i % 9 != 0]

#Create train and test sets for medium dataset (50 programmers)
mtest_data = np.array(middle_set[0])
for i in range(9, len(middle_set), 9):
    mtest_data = np.vstack([mtest_data, middle_set[i]])
mtrain_data = np.delete(middle_set, [i for i in range(0, len(middle_set), 9)], axis=0)
mtest_labels = [middle_labels[i] for i in range(0, len(middle_labels), 9)]
mtrain_labels = [middle_labels[i] for i in range(0, len(middle_labels)) if i % 9 != 0]

#Hyperparameters for use in model evaluation
feature_max = (int(np.log2(len(train_data[0]))) + 1)
num_trees = [10, 100, 1000]
num_features = [5, 30, 100]

#Model Performance vs. Trees in Ensemble
for i in num_trees:
    random_forest = RandomForestClassifier(n_estimators=i, max_features=feature_max)
    random_forest.fit(train_data, train_labels)
    cross_val = StratifiedKFold(n_splits=8)
    scores = cross_val_score(random_forest, train_data, train_labels, cv=cross_val, scoring='f1_macro')
    print(sum(scores)/len(scores))
    preds = random_forest.predict(test_data)
    print(classification_report(test_labels, preds))
    print(accuracy_score(test_labels, preds))

#Model Performance vs. Random Features Considered
#Note: Number of trees is treated as a hyperparameter here and fine-tuned using cross-validation
best_trees = []
for i in num_features:
    tree_scores = []
    for j in num_trees:
        random_forest = RandomForestClassifier(n_estimators=j, max_features=i)
        random_forest.fit(train_data, train_labels)
        cross_val = StratifiedKFold(n_splits=8)
        scores = cross_val_score(random_forest, train_data, train_labels, cv=cross_val, scoring='f1_macro')
        tree_scores.append(sum(scores)/len(scores))
    best_trees.append(num_trees[np.argmax(tree_scores)])
print(best_trees)
for i in range(len(num_features)):
    random_forest = RandomForestClassifier(n_estimators=best_trees[i], max_features=num_features[i])
    random_forest.fit(train_data, train_labels)
    cross_val = StratifiedKFold(n_splits=8)
    scores = cross_val_score(random_forest, train_data, train_labels, cv=cross_val, scoring='f1_macro')
    print(sum(scores)/len(scores))
    preds = random_forest.predict(test_data)
    print(classification_report(test_labels, preds))
    print(accuracy_score(test_labels, preds))

#Model Performance vs. Number of Classes
#Small dataset (10 programmers)
tree_scores = []
for j in num_trees:
    random_forest = RandomForestClassifier(n_estimators=j, max_features=feature_max)
    random_forest.fit(strain_data, strain_labels)
    cross_val = StratifiedKFold(n_splits=8)
    scores = cross_val_score(random_forest, strain_data, strain_labels, cv=cross_val, scoring='f1_macro')
    tree_scores.append(sum(scores)/len(scores))

print(num_trees[np.argmax(tree_scores)])
random_forest = RandomForestClassifier(n_estimators=num_trees[np.argmax(tree_scores)], max_features=feature_max)
random_forest.fit(strain_data, strain_labels)
cross_val = StratifiedKFold(n_splits=8)
scores = cross_val_score(random_forest, strain_data, strain_labels, cv=cross_val, scoring='f1_macro')
print(sum(scores)/len(scores))
preds = random_forest.predict(stest_data)
print(classification_report(stest_labels, preds))
print(accuracy_score(stest_labels, preds))

#Medium dataset (50 programmers)
tree_scores = []
for j in num_trees:
    random_forest = RandomForestClassifier(n_estimators=j, max_features=feature_max)
    random_forest.fit(mtrain_data, mtrain_labels)
    cross_val = StratifiedKFold(n_splits=8)
    scores = cross_val_score(random_forest, mtrain_data, mtrain_labels, cv=cross_val, scoring='f1_macro')
    tree_scores.append(sum(scores)/len(scores))

print(num_trees[np.argmax(tree_scores)])
random_forest = RandomForestClassifier(n_estimators=num_trees[np.argmax(tree_scores)], max_features=feature_max)
random_forest.fit(mtrain_data, mtrain_labels)
cross_val = StratifiedKFold(n_splits=8)
scores = cross_val_score(random_forest, mtrain_data, mtrain_labels, cv=cross_val, scoring='f1_macro')
print(sum(scores)/len(scores))
preds = random_forest.predict(mtest_data)
print(classification_report(mtest_labels, preds))
print(accuracy_score(mtest_labels, preds))

#Full dataset (102 programmers)
tree_scores = []
for j in num_trees:
    random_forest = RandomForestClassifier(n_estimators=j, max_features=feature_max)
    random_forest.fit(mtrain_data, mtrain_labels)
    cross_val = StratifiedKFold(n_splits=8)
    scores = cross_val_score(random_forest, mtrain_data, mtrain_labels, cv=cross_val, scoring='f1_macro')
    tree_scores.append(sum(scores)/len(scores))

print(num_trees[np.argmax(tree_scores)])
random_forest = RandomForestClassifier(n_estimators=num_trees[np.argmax(tree_scores)], max_features=feature_max)
random_forest.fit(train_data, train_labels)
cross_val = StratifiedKFold(n_splits=8)
scores = cross_val_score(random_forest, train_data, train_labels, cv=cross_val, scoring='f1_macro')
print(sum(scores)/len(scores))
preds = random_forest.predict(test_data)
print(classification_report(test_labels, preds))
print(accuracy_score(test_labels, preds))

#Linear SVM vs. Random Forest
linear_svm = LinearSVC(penalty='l2', loss='squared_hinge')
linear_svm.fit(train_data, train_labels)
cross_val = StratifiedKFold(n_splits=8)
scores = cross_val_score(linear_svm, train_data, train_labels, cv=cross_val, scoring='f1_macro')
print(sum(scores)/len(scores))
predictions = linear_svm.predict(test_data)
print(classification_report(test_labels, predictions))
print(accuracy_score(test_labels, predictions))

#Most Important Features by Information Gain
info_gain = dict(zip(nh, mutual_info_classif(train_data, train_labels)))
top_20 = sorted(list(info_gain.items()), key = lambda i: i[1], reverse=True)[:20]
with open(dir_path + '\\infogain.csv', 'w', newline='') as ig_file:
    ig_writer = csv.writer(ig_file, delimiter = ',')
    for i in top_20:
        ig_writer.writerow([i[0], i[1]])