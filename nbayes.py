import csv

import numpy as np
import pandas as pd

from functions import *

# train_data = pd.read_csv(spam_path)

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45, FeatureType
import csv
import os



# Below four maybe should be in util.py?
def accuracy_score(y_true, y_pred):
    return


def recall(y_true, y_pred):
    return


def precision(y_true, y_pred):
    return


def ROC(y_true, y_pred):
    return


# =============================(useless dividing line)


class NaiveBayesClassifier:

    def __init__(self, train_data, features, m):
        self.train_data = train_data
        self.size = train_data.shape[0]
        self.features = features  # 'Outlook', 'Temp', 'Humidity', 'Windy', 'Play'
        self.labels = train_data[features[len(features) - 1]].unique()  # 'no', 'yes'
        self.y = train_data.iloc[:, -1:].to_numpy().reshape(train_data.shape[0], )
        self.num_pos = sum(train_data[self.features[len(self.features) - 1]] == self.labels[1])
        self.num_neg = sum(train_data[self.features[len(self.features) - 1]] == self.labels[0])
        self.liklihood_neg = {}
        self.liklihood_pos = {}
        self.marginal_prob = {}
        self.class_prior = []
        self.m = m

    '''
    get class proir (P(c) - Prior Class Probability)
    '''

    def _calc_class_prior(self):
        for label in self.labels:
            count = 0
            for i in range(len(self.y)):
                if self.y[i] == label:
                    count += 1
            self.class_prior.append(count / self.size)

    '''
    Get the liklihood P(x|c)
    '''

    def _calc_liklihood(self):
        for i in range(len(self.features) - 1):
            feature = self.features[i]
            unique_features = self.train_data[
                feature].unique()  # get all possible outcome for this feature(Rainy, Sunny, Overcast)
            for unique_feature in unique_features:
                feature_data = self.train_data.loc[
                    self.train_data[feature] == unique_feature]  # find data with specific outcome
                count_neg = feature_data.loc[feature_data[features[len(self.features) - 1]] == self.labels[0]].shape[
                                0] + self.m  # if the label is 'no'
                count_pos = feature_data.loc[feature_data[features[len(self.features) - 1]] == self.labels[1]].shape[
                                0] + self.m
                self.liklihood_neg.update({unique_feature: count_neg / self.num_neg})
                self.liklihood_pos.update({unique_feature: count_pos / self.num_pos})

    """ 
    P(x) - Evidence or called Marginal Probability
    """

    def _calc_marginal_prob(self):
        for i in range(len(self.features) - 1):
            feature = self.features[i]
            unique_features = self.train_data[feature].unique()
            for unique_feature in unique_features:
                feature_data = self.train_data.loc[self.train_data[feature] == unique_feature]
                self.marginal_prob.update({unique_feature: (feature_data.shape[0]) / self.size})

    '''
    train here
    '''

    def train(self):
        self._calc_class_prior()
        self._calc_liklihood()
        self._calc_marginal_prob()

    '''
    predict here
    '''

    def predict(self, test_df):
        result = []
        for i in range(test_df.shape[0]):  # get every row of the testing data
            marginal_prob_pred = 0
            liklihood_pos_pred = 0
            liklihood_neg_pred = 0
            df_row = test_df.iloc[i]
            for outcome in df_row:  # get every feature outcome and calculate
                liklihood_pos_pred += np.log(self.liklihood_pos[outcome])
                liklihood_neg_pred += np.log(self.liklihood_neg[outcome])
                marginal_prob_pred += np.log(self.marginal_prob[outcome])
            posterior_prob_pos = liklihood_pos_pred + self.class_prior[1] - marginal_prob_pred
            posterior_prob_neg = liklihood_neg_pred + self.class_prior[0] - marginal_prob_pred
            if posterior_prob_pos > posterior_prob_neg:  # find which is greater and get the result
                result.append(self.labels[1])
            else:
                result.append(self.labels[0])
        return result

    '''
    Not formal function to get accuracy
    '''

    def accuracy(self, predicted_result):
        count = 0
        for i in range(len(predicted_result)):
            if predicted_result[i] == self.y[i]:
                count += 1
        return count / len(predicted_result)

'''
Function just to make calling easier
'''
def nbayes(data, features, m):
    clf = NaiveBayesClassifier(data, features, m)
    clf.train()
    # df = np.array([['Sunny', 'Mild', 'High', 't'],
    #                ['Overcast', 'Cool', 'Normal', 't'],
    #                ['Sunny', 'Hot', 'High', 't']])

    test_df = data.iloc[[0, 1, 2, 3, 4, 5, 6, 7]]

    # df = np.array([["Rainy", "Hot", "High", "f"],
    #                ["Rainy", "Hot", "High", "t"],
    #                ["Overcast", "Hot", "High", "f"],
    #                ["Sunny", "Mild", "High", "f"],
    #                ["Sunny", "Cool", "Normal", "f"],
    #                ["Sunny", "Cool", "Normal", "t"],
    #                ["Overcast", "Cool", "Normal", "t"],
    #                ["Rainy", "Mild", "High", "f"],
    #                ["Rainy", "Cool", "Normal", "f"],
    #                ["Sunny", "Mild", "Normal", "f"],
    #                ["Rainy", "Mild", "Normal", "t"],
    #                ["Overcast", "Mild", "High", "t"],
    #                ["Overcast", "Hot", "Normal", "f"],
    #                ["Sunny", "Mild", "High", "t"]])
    #test_df = pd.DataFrame(df)
    # print(clf.accuracy(clf.predict(test_df)))
    print(clf.predict(test_df))

def split_k_bins(data, k):
    X_Split = np.empty((X.shape[0], 0), int)
    for column in data.T:
        splitter = (np.ptp(column)) / k
        splitcolumn = np.divmod(column, splitter)[0]
        splitcolumn = np.reshape(splitcolumn, (1, splitcolumn.shape[0]))
        X_Split = np.append(X_Split, splitcolumn.T, axis=1)
    return X_Split



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Run a Naive Bayes algorithm.')
    # parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    # parser.add_argument('bins', metavar='BINS', type=int,
    #                     help='Number of bins to divide continuous features into')
    # parser.add_argument('estimate', metavar='ESTIMATE', type=float,
    #                     help='A nonnegative integer m for the m-estimate. If this value is negative, Laplace smoothing will be used.')
    # parser.add_argument('--no-cv', dest='cv', action='store_false',
    #                     help='Disables cross validation and trains on the full dataset.')
    # parser.add_argument('--research', dest='research', action='store_true',
    #                     help='Enables research extension for this project.')
    #
    # parser.set_defaults(cv=True, research=False)
    # args = parser.parse_args()

    # save args
    # data_path = args.path
    # number_bins = args.bins
    # use_cross_validation = args.cv
    # use_research = args.research
    # m_estimate = args.estimate

    schema, X, y = parse_c45('spam', os.path.dirname(os.path.abspath(__file__)) + "\\440data\\spam")
    features = []
    for feature in schema:
        features.append(feature.name)
    features.append('Class label')
    features = np.array(features)

    split_data = split_k_bins(X, 5)
    y = np.reshape(y, (1, y.shape[0]))
    split_data = np.append(split_data,y.T,axis=1)

    # just for easy use
    data = pd.DataFrame()
    i = 0
    for feature in features:
        col = pd.Series(split_data[ : , i], name=feature)
        data[feature] = col
        i += 1

    #data = pd.read_fwf('440data/weather.txt')
    #features = np.array(['Outlook', 'Temp', 'Humidity', 'Windy', 'Play'])
    nbayes(data, features, 0)
    #tree = ID3(X, "1.1")
    #test_data_m = pd.read_csv(spam_path)

    #accuracy = evaluate(tree, test_data_m, "1.1")

    #print("accuracy is: ", accuracy)


