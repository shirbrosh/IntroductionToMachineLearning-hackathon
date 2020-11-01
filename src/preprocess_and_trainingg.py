import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import *
import pandas as pd
from pandas import DataFrame
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import pickle

DIVISION_CONST = 60
TEST_PART = 0.2
TRAIN_PART = 0.8
def read_files():
    """
    A method to read all the text files
    :return: a list of all the text files
    """
    building_tool = open("building_tool_all_data.txt", errors="ignore")
    espnet = open("espnet_all_data.txt", errors="ignore")
    horovod = open("horovod_all_data.txt", errors="ignore")
    jina = open("jina_all_data.txt", errors="ignore")
    PaddleHub = open("PaddleHub_all_data.txt", errors="ignore")
    PySolFC = open("PySolFC_all_data.txt", errors="ignore")
    pytorch_geometric = open("pytorch_geometric_all_data.txt", errors="ignore")
    return [building_tool, espnet, horovod, jina, PaddleHub, PySolFC,
            pytorch_geometric]


def extract_samples(files):
    """
    A method to create samples from the txt files
    :param files: 7 text files
    :return: a list of samples from each file
    """
    raw_samples = []
    for file in files:
        text = file.readlines()
        sample = []
        i = 0
        while i < len(text):
            n = random.randint(1, min(5, len(text) - i))
            sample.append('-'.join(text[i:i + n]))
            i += n
        raw_samples.append(DataFrame(sample))
    return raw_samples


def divide_test_train(raw_samples):
    """
    this method divides the samples between train and test sets
    :param raw_samples: samples to divide
    :return: a train set and a test test
    """
    train = []
    test = []
    for i in range(7):
        x_train, x_test, y_train, y_test = train_test_split(raw_samples[i],
                                                            DataFrame(np.full(
                                                                len(raw_samples[i]),
                                                                i)), test_size=TEST_PART,
                                                            train_size=TRAIN_PART)
        train_buff = pd.concat([x_train, y_train], axis=1)
        test_buff = pd.concat([x_test, y_test], axis=1)
        train.append(train_buff)
        test.append(test_buff)
    train_df = pd.concat(train)
    test_df = pd.concat(test)
    train_df.columns = ['x', 'y']
    test_df.columns = ['x', 'y']
    return train_df, test_df


def filtering_features(all_words, m):
    """
    This function filters out features that there standard deviation is lower then DIVISION_CONST
    :param all_words: a dictionary with all the features=words
    :param m: number of samples
    :return: dictionary with filtered features
    """
    final_dic = {}
    for word in tqdm(all_words):
        avg = np.average(all_words[word])
        Standard_Deviation = all_words[word] - avg
        if max(Standard_Deviation) > DIVISION_CONST or min(
                Standard_Deviation) < -DIVISION_CONST:
            final_dic[word] = np.zeros(m, dtype='int16')
    return final_dic


def from_samples_to_dictionary(df, m):
    """
    This function takes a data frame with all the samples = strings and creates a
    dictionary with all the filtered features = words, and initialize all the key
    values to zero arrays in size of m.
    :param df: the data frame with the samples
    :param m: the number of samples
    :return: the final dictionary
    """
    all_words = {}
    tknzr = nltk.tokenize.TweetTokenizer()
    array_of_counters = []
    i = 0
    amount_of_words = np.zeros(7)
    for sample in tqdm(df['x']):
        words = tknzr.tokenize(sample)
        amount_of_words[np.array(df['y'])[i]] += len(words)
        c = Counter(words)
        array_of_counters.append(c)
        for word in c:
            if word not in all_words.keys():
                all_words[word] = np.zeros(7, dtype='int16')
                all_words[word][np.array(df['y'])[i]] = c[word]
            else:
                all_words[word][np.array(df['y'])[i]] += c[word]
        i += 1
    samples_matrix = []
    for word in tqdm(all_words):
        samples_matrix.append(all_words[word])
    final_dic = filtering_features(all_words, m)
    return final_dic


def create_features(df, all_words):
    """
    This function receives a data frame of samples and creates the data frame with
    matched features
    :param df: the data frame of samples
    :param all_words: dictionary with all the words to be featured
    :return: the ready data frame with features and the matching y labels
    """
    tknzr = nltk.tokenize.TweetTokenizer()
    y = df['y']
    i = 0
    for sample in tqdm(df['x']):
        words = tknzr.tokenize(sample)
        c = Counter(words)
        for word in all_words:
            all_words[word][i] = c[word]
        i += 1
    df = DataFrame.from_dict(all_words)
    return df, y


def decision_param_estim(depths, train_X, train_y, valid_X, valid_y):
    """
    Gets the train and test data and check how mush error there is
    :param depths:  A list of depth for our decision tree
    :param train_X: the x train samples
    :param train_y: the y train labels
    :param valid_X: he x  valid samples
    :param valid_y: the y valid labels
    :return:
    """
    errors = []
    for k in tqdm(depths):
        model = DecisionTreeClassifier(max_depth=k)
        model.fit(train_X, train_y)
        errors.append(round(1 - model.score(valid_X, valid_y), 4))
        print(" decision complete: ", k)
    plt.figure()
    plt.plot(depths, errors)
    plt.title("Decision Tree - errors as a function of depth")
    plt.xlabel("depths")
    plt.ylabel("error")
    plt.show()
    plt.savefig("Decision Tree - errors as a function of depth.png")


def RF_parameter_estimation(tree_nums, train_X, train_y, valid_X, valid_y):
    """
    Gets the train and test data and check how mush error there is
    :param tree_nums: the number of trees
    :param train_X: the x train samples
    :param train_y: the y train labels
    :param valid_X: he x  valid samples
    :param valid_y: the y valid labels
    :return:
    """
    errors = []
    for num in tree_nums:
        model = RandomForestClassifier(n_estimators=num, max_depth=2)
        model.fit(train_X, train_y)
        errors.append(round(1 - model.score(valid_X, valid_y), 4))
        print(" RF complete: ", num)
    plt.figure()
    plt.plot(tree_nums, errors)
    plt.title("error as a function of tree nums (RF)")
    plt.xlabel("tree num")
    plt.ylabel("error")
    plt.show()
    plt.savefig("error as a function of tree nums (RF).png")


def adaboost_model(X_train, y_trian, X_valid, y_valid):
    """
    adaboosting
    :param train_X: the x train samples
    :param train_y: the y train labels
    :param valid_X: he x  valid samples
    :param valid_y: the y valid labels
    :return:
    """
    adaboost_learner = AdaBoostClassifier(n_estimators=15,
                                          learning_rate=1)
    model = adaboost_learner.fit(X_train, y_trian)
    y_pred = model.predict(X_valid)
    print("Accuracy:", metrics.accuracy_score(y_valid, y_pred))


if __name__ == '__main__':
    train, test = divide_test_train(extract_samples(read_files()))
    data = pd.concat([train, test])
    all_words = from_samples_to_dictionary(data,data.shape[0])
    X, y = create_features(data,all_words)
    X.to_csv("final data.csv")
    y.to_csv("final labels.csv")
    pd_train = pd.read_csv("final data.csv")
    x_numpy= pd_train.to_numpy()[:,1:]
    y_numpy = pd.read_csv("final labels.csv").to_numpy()[:,1]
    model = DecisionTreeClassifier(max_depth=100)
    model.fit(x_numpy, y_numpy)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    errors = round(1 - model.score(x_numpy, y_numpy), 4)
    print("error: ", errors)


