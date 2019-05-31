
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from utils.pickle import load_pickles

#%% [markdown]
# #### Load pre-processed data

#%%
(
    train_x,
    valid_x,
    test_x,
    train_y,
    valid_y,
    test_y
) = load_pickles()

#%% [markdown]
# #### Vectorize training data into features

#%%
word_vectorizer = TfidfVectorizer(lowercase=False,ngram_range=(1,3),max_features=20000)
word_vectorizer = word_vectorizer.fit(train_x)
train_features = word_vectorizer.transform(train_x)
validation_features = word_vectorizer.transform(valid_x)

#%% [markdown]
# #### Train and calculate AUC using decision tree classifiers for all class labels and varying max leaf nodes

#%%
classes = 6
max_leaf_nodes = [10, 20, 40, 80, 160]
train_mean_aucs, val_mean_aucs = [], []

for max_leaf_node in max_leaf_nodes:
    val_aucs = []
    train_aucs = []
    for i in range(classes):
        classifier = DecisionTreeClassifier(max_leaf_nodes=max_leaf_node)
        classifier = classifier.fit(train_features, train_y[:, i])
        
        valid_y_hat = classifier.predict_proba(validation_features)[:, 1]
        fpr, tpr, thresholds = roc_curve(valid_y[:, i], valid_y_hat)
        val_aucs.append(auc(fpr, tpr))
        
        train_y_hat = classifier.predict_proba(train_features)[:, 1]
        fpr, tpr, thresholds = roc_curve(train_y[:, i], train_y_hat)
        train_aucs.append(auc(fpr, tpr))

    val_mean_aucs.append(np.mean(val_aucs))
    train_mean_aucs.append(np.mean(train_aucs))


#%%
plt.plot(max_leaf_nodes, train_mean_aucs, "r", label="Training AUC")
plt.plot(max_leaf_nodes, val_mean_aucs, label="Validation AUC")
plt.xlabel("Max Leaf nodes")
plt.ylabel("AUC")
plt.legend()
plt.show()

#%% [markdown]
# Looking at the graph, it is clear that at Max leaf nodes = 80, we have the highest AUC. Anything larger than that seems to overfit the data.

#%%
max_depths = [10, 20, 40, 80, 160, 320]
train_mean_aucs, val_mean_aucs = [], []

for max_depth in max_depths:
    val_aucs = []
    train_aucs = []
    for i in range(classes):
        classifier = DecisionTreeClassifier(max_leaf_nodes=80, max_depth=max_depth)
        classifier = classifier.fit(train_features, train_y[:, i])
        
        valid_y_hat = classifier.predict_proba(validation_features)[:, 1]
        fpr, tpr, thresholds = roc_curve(valid_y[:, i], valid_y_hat)
        val_aucs.append(auc(fpr, tpr))
        
        train_y_hat = classifier.predict_proba(train_features)[:, 1]
        fpr, tpr, thresholds = roc_curve(train_y[:, i], train_y_hat)
        train_aucs.append(auc(fpr, tpr))

    val_mean_aucs.append(np.mean(val_aucs))
    train_mean_aucs.append(np.mean(train_aucs))


#%%
plt.plot(max_depths, train_mean_aucs, "r", label="Training AUC")
plt.plot(max_depths, val_mean_aucs, label="Validation AUC")
plt.xlabel("Max Depth")
plt.ylabel("AUC")
plt.legend()
plt.show()

#%% [markdown]
# In our model, increasing max depth does not seem to overfit the data. Hence, we will be using unbounded max depth where the nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples

#%%
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_mean_aucs, val_mean_aucs = [], []

for min_samples_split in min_samples_splits:
    val_aucs = []
    train_aucs = []
    for i in range(classes):
        classifier = DecisionTreeClassifier(max_leaf_nodes=80, min_samples_split=min_samples_split)
        classifier = classifier.fit(train_features, train_y[:, i])
        
        valid_y_hat = classifier.predict_proba(validation_features)[:, 1]
        fpr, tpr, thresholds = roc_curve(valid_y[:, i], valid_y_hat)
        val_aucs.append(auc(fpr, tpr))
        
        train_y_hat = classifier.predict_proba(train_features)[:, 1]
        fpr, tpr, thresholds = roc_curve(train_y[:, i], train_y_hat)
        train_aucs.append(auc(fpr, tpr))

    val_mean_aucs.append(np.mean(val_aucs))
    train_mean_aucs.append(np.mean(train_aucs))


#%%
plt.plot(min_samples_splits, train_mean_aucs, "r", label="Training AUC")
plt.plot(min_samples_splits, val_mean_aucs, label="Validation AUC")
plt.xlabel("Minimum Sample Split")
plt.ylabel("AUC")
plt.legend()
plt.show()


