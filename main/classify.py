import numpy as np
import pandas as pd
import time
from classification.knn import knn
from classification.svm import SVM
from classification.learningshapelet import LearningShapelet
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

train_filepath='data/data_simple/train2.txt'
test_filepath='data/data_simple/test2.txt'

def read_datasets(train_filepath,test_filepath):
    train_df=pd.DataFrame(np.loadtxt(train_filepath))
    test_df=pd.DataFrame(np.loadtxt(test_filepath))
    X, y = train_df[train_df.columns[1:]].values,train_df[train_df.columns[0]].values.astype(int)
    time_series, labels = test_df[test_df.columns[1:]].values,test_df[test_df.columns[0]].values
    return X,y,time_series,labels


if __name__ == '__main__':

   # load the data
    print('Loading data...')
    x_train, y_train, x_test, y_test = read_datasets(train_filepath,test_filepath)
    x_train=x_train
    y_train=y_train
    x_test=x_test
    y_test=y_test

    # create a classifier
    print('\nBuilding classifier...')
    clf=knn()
    #clf=SVM()
    #clf = LearningShapelet()


    # train the classifier
    print('\nTraining...')
    clf.fit(x_train, y_train)

    # evaluate on test data
    print('\nPredicting...')
    time_1 = time.time()
    predicted = clf.predict(x_test)
    print('accuracy ',accuracy_score(predicted, y_test))
    report = classification_report(y_test,predicted, target_names=['1','2', '3'])
    print(report)
    time_2 = time.time()
    print('cost ', time_2 - time_1, ' second', '\n')
