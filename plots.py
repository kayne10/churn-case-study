import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from plot_helper import *

def get_variance(fitted_model):
	features = fitted_model.feature_importances_
	trees = fitted_model.estimators_
	variances = np.empty([len(features), len(trees)])
	for idx, feature in enumerate(features):
		for col, tree in enumerate(trees):
			  variances[idx,col] = tree.feature_importances_[idx]
	return variances

def plot_important_features(df, fitted_model):
    feature_names = df.columns.values
    stds = np.std(get_variance(fitted_model))
    # low_error = fitted_model.feature_importances_ - stds
    # high_error = fitted_model.feature_importances_ + stds
    # errors = [low_error,high_error]
    plt.figure(figsize=(15,10))
    plt.bar(feature_names, fitted_model.feature_importances_)
    plt.title('Feature Importances')
    plt.ylabel('Normalized decrease in node impurity')
    plt.xlabel('Features')


def plot_cnf_matrix(model, X_test, y_test):
    cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ['Churn', 'Not Churn']
    plot_confusion_matrix(cnf_matrix, ax, classes=class_names,normalize=True,
                      title='Confusion Matrix')


def plot_roc_curves(X_train,y_train,X_test,y_test):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    models = [GradientBoostingClassifier(),RandomForestClassifier(), DecisionTreeClassifier(),
			KNeighborsClassifier()]
    for model in models:
        model = model
        model.fit(X_train,y_train)
        plot_roc(model, X_test, y_test, ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
    ax.set_xlabel("False Positive Rate (1-Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
    ax.set_title("ROC plot of 'Churn, Not Churn'")
    ax.legend()

def plot_true_vs_predicted(x, y_true, y_pred):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(x,y_true,label="Actual")
    ax.plot(x,y_pred,xlabel="Predicted")
    ax.set_title("Actual Values vs. Predicted Values")
    ax.set_ylabel("Churn or Not Churn")
    ax.set_xlabel("Users")
    ax.legend()
