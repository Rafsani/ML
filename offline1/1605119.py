from math import log
from numpy.core.arrayprint import printoptions
import pandas as pd;
import numpy as np
from pandas.core.frame import DataFrame
from scipy.sparse import data;
from sklearn.impute import SimpleImputer  # used for handling missing data
# used for encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# used for splitting training and testing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # used for feature scaling

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv');
# print(dataset.isnull().sum());
# #df = pd.DataFrame(dataset, columns=['SeniorCitizen','Partner','Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn' ]);
dataset.drop('customerID', axis=1, inplace=True);

dataset = dataset.replace(r'^\s+$', np.nan, regex=True);
#print(dataset.isnull().sum());
#print(dataset.loc[[488,489]]);

dataset.TotalCharges = pd.to_numeric(dataset.TotalCharges, errors='coerce')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(dataset[['TotalCharges']])
dataset['TotalCharges'] = imputer.transform(dataset[['TotalCharges']])

#dataset.replace({'MultipleLines': {'No phone service': 'No'}}, inplace=True)
dataset.replace({'gender': {'Male': 1 , 'Female': 0}}, inplace=True)
# dataset.replace( {'OnlineSecurity': {'No internet service': 'No'}}, inplace=True)
# dataset.replace({'OnlineBackup': {'No internet service': 'No'}}, inplace=True)
# dataset.replace({'DeviceProtection': {'No internet service': 'No'}}, inplace=True)
# dataset.replace({'TechSupport': {'No internet service': 'No'}}, inplace=True)
# dataset.replace({'StreamingTV': {'No internet service': 'No'}}, inplace=True)
# dataset.replace({'StreamingMovies': {'No internet service': 'No'}}, inplace=True)


#print(dataset.DeviceProtection.unique())
dataset.replace('No', 0, inplace=True);
dataset.replace('Yes', 1, inplace=True);
#dataset.replace({'Churn': {0: -1, 1: 1}}, inplace=True)

Y = dataset.Churn;
dataset.drop('Churn', axis=1, inplace=True)

dataset = pd.get_dummies(dataset)
# for i in dataset.columns:
#     print(i, dataset[i].unique());

normalized_dataset = (dataset-dataset.min())/(dataset.max()-dataset.min())


X =  dataset.values



class LogisticRegression(object):
    def __init__(self,X,Y):
        self.X_data = X
        self.Y_data = Y
        self.weight = np.zeros(self.X_data.shape[1], dtype=float, order='C')
        pre = np.ones((self.X_data.shape[0],1))
        self.X_data =  np.concatenate((pre,self.X_data),axis=1)
        

    def Tanh(self,arg):
        return np.tanh(arg)
    
    def costf(self,yhat):
        m = self.X_data.shape[0]
      #  print(yhat.shape)
        error = yhat - self.Y_data
       # print('err',error.shape)
        cost = (-1/m)*np.sum(error**2)  # mean sqr error
        # print(cost.shape)
        return cost


    def train(self,alpha=0.000001,epochs=500,threshold = 0.5):
        m = self.X_data.shape[0]
        n = self.X_data.shape[1]
       

        weight = np.zeros(n,dtype=float,order='C')
        #print(weight)
        for i in range(epochs):
            print('epoch :', i+1)
            print(np.dot(self.X_data,weight))
            hypothesis = self.Tanh(np.dot(self.X_data,weight))
            #print(hypothesis)
            cost = self.costf(hypothesis)
            print('cost :', cost)
            if np.abs(cost) < 0.1:
                break
            gradient = (1/m) * np.dot(self.X_data.T,(self.Y_data - hypothesis)*(1-hypothesis**2))
            weight = weight + alpha*gradient

        return weight

    def predict(self,X_test,Y_test,weight,threshold=0.5):
        pre = np.ones((X_test.shape[0], 1))
        X_test = np.concatenate((pre, X_test), axis=1)
        prediction = self.Tanh(np.dot(X_test, weight))
        for i in range(prediction.shape[0]):
            if prediction[i] > threshold:
                prediction[i] = 1
            else:
                prediction[i] = 0
        return prediction
    
    def accuracy(self,prediction,Y_test):
        correct = 0
       # print(prediction)
        for i in range(len(prediction)):
             if prediction[i] == Y_test.iloc[i]:
                correct += 1
        return (correct/len(prediction))*100


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

logistic_regression = LogisticRegression(x_train,y_train)
weight = logistic_regression.train()
prediction = logistic_regression.predict(x_test,y_test,weight)
print(logistic_regression.accuracy(prediction,y_test))
