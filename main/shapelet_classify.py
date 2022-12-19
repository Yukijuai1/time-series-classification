import time
from math import ceil
import numpy as np
import pandas as pd
from classification.shapelet import find_shapelets_bf
from classification.shapelet import subsequence_dist
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

train_filepath='data/data_simple/2class/train2_2class.txt'
test_filepath='data/data_simple/2class/test2_2class.txt'


def data_cut(lst, num):
    return list(map(lambda x: lst[x * num : x * num + num], list(range(0, ceil(len(lst) / num)))))


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

    print('Cuting data...')
    new_x_train=data_cut(x_train,100)
    new_y_train=data_cut(y_train,100)
    
    '''
    print('Generating shapelets...')
    all_shapelet=[]
    for i in range(len(new_x_train)):
        print("Now batch: ",i)
        train_data=[]
        for j in range(len(new_x_train[i])):
            train_data.append((new_x_train[i][j],new_y_train[i][j]))
        shapelet, dist=find_shapelets_bf(train_data, max_len=10, min_len=8)
        all_shapelet.append(shapelet)
    
    print('Generating model...')
    model_file=open('model/shapelet.txt','w')
    for i in range(len(all_shapelet)):
        for j in range(len(all_shapelet[i])):
            model_file.write(str(all_shapelet[i][j])+" ")
        model_file.write("\n")

    '''
    print('Loading model...')
    all_shapelet=[] 
    model=open('main/model/shapelet.txt')
    for line in model:
        part=line.split(" ")
        list=[]
        for i in range(len(part)-1):
            list.append(float(part[i]))
        all_shapelet.append(list)
    
    print('Generating features...')
    new_x_train=[]
    for i in range(len(x_train)):
        features=[]
        for j in range(len(all_shapelet)):
            dist=subsequence_dist(x_train[i],all_shapelet[j])
            features.append(dist[0])
        new_x_train.append(features)
    
    new_x_test=[]
    for i in range(len(x_test)):
        features=[]
        for j in range(len(all_shapelet)):
            dist=subsequence_dist(x_test[i],all_shapelet[j])
            features.append(dist[0])
        new_x_test.append(features)
    

    print('\nBuilding classifier...')
    clf = tree.DecisionTreeClassifier(max_depth=len(all_shapelet))

    print('\nTraining...')
    clf.fit(new_x_train, y_train)

    print('\nPredicting...')
    time_1 = time.time()
    predicted = clf.predict(new_x_train)
    print('accuracy ',accuracy_score(predicted, y_train))
    report = classification_report(y_train,predicted, target_names=['1','2'])
    print(report)
    time_2 = time.time()
    print('cost ', time_2 - time_1, ' second', '\n')



    