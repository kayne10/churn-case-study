from DecisionTree import DecisionTree
# from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter

class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        trees = []
        for i in range(self.num_trees):
            bootstrap = [np.random.choice(X.shape[0],num_samples)]
            new_X = np.array(X[bootstrap])
            new_y = np.array(y[bootstrap])
            # import pdb; pdb.set_trace()
            dt = DecisionTree(num_features = self.num_features)
            dt.fit(new_X, new_y)
            trees.append(dt)
        return trees

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''

        predictions = []
        for dt in self.forest:
            predictions.append(dt.predict(X))
        avg_predictions = []
        label = list(set(predictions[0]))
        for idx in range(X.shape[0]) :
                y = [predict[idx] for predict in predictions]
                if y.count(label[0])/len(y) > .5:
                    avg_predictions.append(label[0])
                elif y.count(label[0])/len(y) == .5:
                    avg_predictions.append(np.random.choice(label,1))
                else:
                    avg_predictions.append(label[1])
        # import pdb; pdb.set_trace()
        return np.array(avg_predictions)

    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        # y_pred = self.predict(X)
        # score = y == y_pred
        # return np.mean(score)
        return (self.predict(X) == y).mean()

if __name__ =='__main__':

    df = pd.read_csv('../data/playgolf.csv')
    y = df.pop('Result').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # df = pd.read_csv('../data/congressional_voting.csv',header=None)
    # y = df.pop(0).values
    # X = df.values
    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # dt = DecisionTree()
    # dt.fit(X_train, y_train)
    # predicted_y = dt.predict(X_test)

    rf = RandomForest(num_trees=10, num_features=2)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    print("score:", rf.score(X_test, y_test))
