import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from roc import plot_roc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score

#
df_test = pd.read_csv('../clean_churn_test.csv')
# df_train = pd.read_csv('../clean_churn_train.csv')

# def convert(df,headers):
#     for i in headers:
#         df[i] = df[i].map({'no':False,'yes':True,'False.':False,'True.':True})
#
# convert(df,["Int'l Plan","VMail Plan",'Churn?'])
# df = df.drop(['State','Area Code','Phone'],axis=1)

def prep_df(file_name,y_col):
    df = pd.read_csv(file_name)
    y = df.pop(y_col).values
    X = df.values
    return X, y

X_test, y_test = prep_df('../clean_churn_test.csv','churn')
X_train, y_train = prep_df('../clean_churn_train.csv','churn')


# X_train, X_test, y_train, y_test = train_test_split(X, y)


# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# print("Score w/o Params: ",round(rf.score(X_test,y_test),3))
# y_pred = rf.predict(X_test)
#
# print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
# print("Precision: ", precision_score(y_test,y_pred))
# print("Recall: ", recall_score(y_test,y_pred))


def random_forest_test(X_train, X_test, y_train, y_test,n_est=10,max_features='auto'):
    rf = RandomForestClassifier(n_estimators=n_est,max_features=max_features,oob_score = True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # print(f"Random Forest - \nScore w/ {max_features} Features: ",round(rf.score(X_test,y_test),3))
    # print("Precision: ", round(precision_score(y_test,y_pred),3))
    # print("Recall: ", round(recall_score(y_test,y_pred),3))
    return round(rf.score(X_test,y_test),3)
    # print("Confusion Matrix w/ OOB: \n",confusion_matrix(y_test,y_pred))

#
rf = RandomForestClassifier(n_estimators=10,oob_score = True)
rf.fit(X_train, y_train)

feature_names = list(df_test.columns)[0:df_test.shape[1]-1]
plt.figure(figsize = (12,8))
plt.bar(feature_names, rf.feature_importances_)
plt.title("Feature Importance",fontsize=14,fontweight='bold')
plt.ylabel("Normalized decrease in node impurity")
plt.xlabel("Features")
plt.xticks(rotation=45)
#
# def plot_stuff(thing_to_plot, names,title):
#     plt.figure(figsize = (12,7))
#     plt.plot(names, thing_to_plot)
#     plt.title(title)
#     plt.ylim((.8,1))
#     # plt.xscale('log')
#     plt.xticks(rotation=45)

# n_est = [1,5,10,20,50,100,500,1000]
# trees_accuracy_list = []
# for n in n_est:
#     trees_accuracy_list.append(random_forest_test(X_train, X_test, y_train, y_test,n_est=n))
#
# feat_list = np.arange(1,X_test.shape[1]+1)
# features_accuracy_list = []
# for n in feat_list:
#     features_accuracy_list.append(random_forest_test(X_train, X_test, y_train, y_test,n_est=50,max_features=n))

# def other_method_test(method,X_train, X_test, y_train, y_test):
#     rf = method()
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#     print(f"{method} - \nScore: ",round(rf.score(X_test,y_test),3))
#     print("Precision: ", round(precision_score(y_test,y_pred),3))
#     print("Recall: ", round(recall_score(y_test,y_pred),3))
#     return round(rf.score(X_test,y_test),3)
#
#
# random_forest_test(X_train, X_test, y_train, y_test,n_est=50,max_features=8)
# other_method_test(KNeighborsClassifier,X_train, X_test, y_train, y_test)
# other_method_test(DecisionTreeClassifier,X_train, X_test, y_train, y_test)
# other_method_test(LogisticRegression,X_train, X_test, y_train, y_test)

# def decision_tree():
#     dt = DecisionTreeClassifier()

# plot_roc(X, y, RandomForestClassifier)
# plot_roc(X, y, KNeighborsClassifier)
# plot_roc(X, y, DecisionTreeClassifier)
# plot_roc(X, y, LogisticRegression)


# plot_stuff(trees_accuracy_list,n_est,'Number of Trees')
# plot_stuff(features_accuracy_list,feat_list,'Number of Features')
plt.tight_layout()
plt.show()
