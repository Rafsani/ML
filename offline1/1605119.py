from math import log
import re
from numpy.core.arrayprint import printoptions
from numpy.lib.shape_base import row_stack
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.sparse import data
from sklearn.impute import SimpleImputer  # used for handling missing data
# used for encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# used for splitting training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  # used for feature scaling


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
def getDataset1():
    dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    dataset.drop('customerID', axis=1, inplace=True)

    dataset = dataset.replace(r'^\s+$', np.nan, regex=True)

    dataset.TotalCharges = pd.to_numeric(dataset.TotalCharges, errors='coerce')

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(dataset[['TotalCharges']])
    dataset['TotalCharges'] = imputer.transform(dataset[['TotalCharges']])

    #dataset.replace({'MultipleLines': {'No phone service': 'No'}}, inplace=True)
    dataset.replace({'gender': {'Male': 1, 'Female': 0}}, inplace=True)
    # dataset.replace( {'OnlineSecurity': {'No internet service': 'No'}}, inplace=True)
    # dataset.replace({'OnlineBackup': {'No internet service': 'No'}}, inplace=True)
    # dataset.replace({'DeviceProtection': {'No internet service': 'No'}}, inplace=True)
    # dataset.replace({'TechSupport': {'No internet service': 'No'}}, inplace=True)
    # dataset.replace({'StreamingTV': {'No internet service': 'No'}}, inplace=True)
    # dataset.replace({'StreamingMovies': {'No internet service': 'No'}}, inplace=True)


    dataset.replace('No', 0, inplace=True)
    dataset.replace('Yes', 1, inplace=True)
    dataset.replace({'Churn': {0: -1, 1: 1}}, inplace=True)

    # Sptiting label and features

    Y = dataset.Churn
    dataset.drop('Churn', axis=1, inplace=True)
    dataset = pd.get_dummies(dataset)
    normalized_dataset = (dataset-dataset.min())/(dataset.max()-dataset.min())
    X = normalized_dataset.values
    return X,Y


def getDataset3():
    data = pd.read_csv('creditcard.csv')
    data.drop('Time', axis=1, inplace=True)
    data = data.replace(r'^\s+$', np.nan, regex=True)
    NEG_SAMPLE = 1000
    count = 0; idx = 0
    rows = []
    for idx, row in data.iterrows():
        if row['Class'] == 1:
            rows.append(row)
        elif count < NEG_SAMPLE:
            rows.append(row)
            count += 1
    
    data = pd.DataFrame(rows, columns=data.columns)
    data.replace({'Class': {0: -1, 1: 1}}, inplace=True)
    Y = data.Class
    data.drop('Class', axis=1, inplace=True)

    normalized_data = (data-data.min())/(data.max()-data.min())
    X = normalized_data.values

    return X, Y


def getDataset2():
    datatrain = pd.read_csv('adult.data', header=None)
    datatest = pd.read_csv('adult.test', skiprows=1, header=None)
    data = pd.concat([datatrain, datatest], ignore_index=True)
    for i in range(len(data.columns)):
        print(i, np.count_nonzero(data[i] == ' ?'))
    for i in range(len(data.columns)):
        if(np.count_nonzero(datatrain[i] == ' ?')):
            data[i].replace(' ?', data[i].mode().values[0], inplace=True)

    data[14].replace(' >50K.', 1, inplace=True)
    data[14].replace(' <=50K.', -1, inplace=True)
    data[14].replace(' >50K', 1, inplace=True)
    data[14].replace(' <=50K', -1, inplace=True)

    Y = data.iloc[:, -1]
    
    print(Y)
    data = data.iloc[:, :-1]    # delete last col

    data = pd.get_dummies(data)

    normalized_dataset = (data-data.min())/(data.max()-data.min())

    # for i in range(len(data.columns)):
    #     #print(i, np.count_nonzero(data[i] == ' ?'))
    #     print(data[i].unique())

    print(normalized_dataset.shape)
    X = normalized_dataset.values
    return X, Y


class LogisticRegression(object):
    def __init__(self, X, Y):
        self.X_data = X
        self.Y_data = Y
        self.weight = np.zeros(self.X_data.shape[1], dtype=float, order='C')
        pre = np.ones((self.X_data.shape[0], 1))
        self.X_data = np.concatenate((pre, self.X_data), axis=1)

    def Tanh(self, arg):
        return np.tanh(arg)

    def costf(self, yhat):
        m = self.X_data.shape[0]
        error = yhat - self.Y_data
       # print('err',error.shape)
        cost = (1/m)*np.sum(error**2)  # mean sqr error
        # print(cost.shape)
        return cost

    def train(self, alpha=0.1, epochs=1000, threshold=0.0):
        m = self.X_data.shape[0]
        n = self.X_data.shape[1]

        weight = np.zeros(n, dtype=float, order='C')
        # print(weight)
        for i in range(epochs):
            hypothesis = self.Tanh(np.dot(self.X_data, weight))
            cost = self.costf(hypothesis)
            #print('cost :', cost)
            if np.abs(cost) < 0.5:
                break
            gradient = (1/m) * np.dot(self.X_data.T,
                                      (self.Y_data - hypothesis)*(1-hypothesis**2))
            weight = weight + alpha*gradient

        return weight

    def predict(self, X_test, weight, threshold=0.0):
        pre = np.ones((X_test.shape[0], 1))
        X_test = np.concatenate((pre, X_test), axis=1)
        prediction = self.Tanh(np.dot(X_test, weight))
        for i in range(prediction.shape[0]):
            if prediction[i] > threshold:
                prediction[i] = 1
            else:
                prediction[i] = -1
        return prediction


def accuracy(prediction, Y_test):
    correct = 0
    # print(prediction)
    for i in range(len(prediction)):
        if prediction[i] == Y_test.iloc[i]:
            correct += 1
    return (correct/len(prediction))*100



def Adaboost(X_data, Y_data, X_test, Y_test, k):
    w = np.ones(X_data.shape[0])/X_data.shape[0]
    h = []
    x_sample = []
    y_sample = []
    z = []
    logistic_regression = LogisticRegression(X_data, Y_data)
    for i in range(k):
        # print(X_data.shape)
        idx = np.random.choice(X_data.shape[0], X_data.shape[0], p=w)

        x_sample = X_data[idx]
        y_sample = Y_data.iloc[idx]
        logistic_regression = LogisticRegression(x_sample, y_sample)
        h.append(logistic_regression.train())

        error = 0
        y_pred = logistic_regression.predict(X_data, h[i])

        for j in range(X_data.shape[0]):
            if Y_data.iloc[j] != y_pred[j]:
                error += w[j]

        if error > 0.5:
            z.append(0)
            continue
        for j in range(X_data.shape[0]):
            if Y_data.iloc[j] == y_pred[j]:
                w[j] = w[j]*(error/(1-error))

        w = w/np.sum(w)
        z.append(np.log2((1-error)/error))

    return h, z


def AdaboostPredict(X_test, h, z):
    y_pred = []
    for i in range(len(h)):
        y_pred.append(logistic_regression.predict(X_test, h[i]))
    y_pred = np.array(y_pred)
    y_pred = np.sum(y_pred.T*z, axis=1)
    for i in range(len(y_pred)):
        if y_pred[i] > 0.0:
            y_pred[i] = 1
        else:
            y_pred[i] = -1
    return y_pred




def printReports(y_test,prediction):
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    print('\nTp: ', tp, 'Fn: ', fn, 'Tn: ', tn, 'Fp: ', fp)
    print('\nAccuracy: ', np.sum(y_test == prediction)/len(y_test))
    print('Sensitivity: ', tp/(tp+fn))
    print('Specificity: ', tn/(tn+fp))
    print('Precision: ', tp/(tp+fp))
    print('False Discovery Rate: ', fp/(fp+tp))


####### _______   Dataset Select ________ #######

X,Y = getDataset1()
#X, Y = getDataset2()
#X, Y = getDataset3()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
logistic_regression = LogisticRegression(x_train, y_train)
weight = logistic_regression.train()
prediction = logistic_regression.predict(x_test, weight)
print("-----:logistic Regression Test Data:-----")

printReports(y_test,prediction)



H, Z = Adaboost(x_train, y_train, x_test, y_test, 10)
y_pred = AdaboostPredict(x_test, H, Z)
print('----:AdaBoost Test data:----')

printReports(y_test,y_pred)


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
logistic_regression = LogisticRegression(x_train, y_train)
weight = logistic_regression.train()
prediction = logistic_regression.predict(x_train, weight)
print("-----:logistic Regression Train Data:-----")

printReports(y_train, prediction)


H, Z = Adaboost(x_train, y_train, x_test, y_test, 10)
y_pred = AdaboostPredict(x_train, H, Z)
print('----:AdaBoost Train data:----')

printReports(y_train, y_pred)

