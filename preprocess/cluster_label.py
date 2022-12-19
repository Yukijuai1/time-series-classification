import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from main.cluster.k_means import ts_cluster
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

std_path='/home/ubuntu/xuexi/2022_2/lab/shapelet/'

def cal_retweet(start_time, end_time, datalist):
    res_list = []
    for i in datalist['retweet'].values:
        cnt = 0
        retweet_path = i.split(" ")
        for j in retweet_path:
            single_retweet = j.split(":")
            retweet_past_time = int(single_retweet[1])
            if retweet_past_time >= start_time and retweet_past_time < end_time:
                cnt += 1
        res_list.append(cnt)
    return res_list


def gen_timeSeries_cluster(retweet, all):
    retweet_path = retweet.split(" ")
    cnt = []
    for i in range(24):
        cnt.append(0)
    for i in retweet_path:
        single_retweet = i.split(":")
        retweet_past_time = int(single_retweet[1])
        for j in range(24):
            if retweet_past_time >= (j*60*60) and retweet_past_time < ((j+1)*60*60):
                cnt[j] += 1
    res = []
    for i in cnt:
        res.append(float(i))
    return res


def gen_timeSeries(retweet):
    retweet_path = retweet.split(" ")
    cnt=[]
    for i in range(12):
        cnt.append(0)
    for i in retweet_path:
        single_retweet = i.split(":")
        retweet_past_time = int(single_retweet[1])
        for j in range(12):
            if retweet_past_time >= j*15*60 and retweet_past_time < (j+1)*15*60:
                cnt[j] += 1
    return cnt


def clustering(data, n_cluster):
    cluster = ts_cluster(n_cluster)
    centroids = cluster.k_means_clust(data, n_cluster, 10, 4)
    label = []
    for i in range(len(data)):
        label.append(0)
    for i in range(len(data)):
        if i in cluster.get_assignments()[0]:
            label[i]=0
        elif i in cluster.get_assignments()[1]:
            label[i]=1
        elif i in cluster.get_assignments()[2]:
            label[i]=2
    return label


if __name__ == '__main__':

    # 读取数据
    print('\nLoading data...')
    file = open(std_path+'dataset_weibo.txt')
    datalist = []
    for line in file:
        part = line.split("\t")
        datalist.append(part)
    datalist = pd.DataFrame(datalist)
    datalist.columns = ['message_id', 'root_user_id',
                        'publish_time', 'retweet_times', 'retweet']

    #计算总序列
    print('\nCaculating series...')
    all_ts = []
    for i in range(len(datalist)):
        time_series = gen_timeSeries_cluster(
            datalist['retweet'].values[i], datalist['retweet_times'].values[i])
        all_ts.append(time_series)
    Scaler = MinMaxScaler()
    all_ts = Scaler.fit_transform(np.array(all_ts))

    # 制作标签
    print('\nCreating label...')
    label = clustering(np.array(all_ts), 3)

    # 制作数据集
    print('\nMaking dataset...')
    features = []
    for i in range(len(datalist)):
        time_series = gen_timeSeries(
            datalist['retweet'].values[i])
        features.append(time_series)
    #Scaler = MinMaxScaler()
    #features = Scaler.fit_transform(np.array(features))
    train_attributes, test_attributes, train_labels, test_labels = train_test_split(
        features, label, test_size=0.25, random_state=0)

    # 生成数据文件
    print('\nWriting files...')
    train_file = open(std_path+'data/data_cluster/train1_normal.txt', 'w')
    test_file = open(std_path+'data/data_cluster/test1_normal.txt', 'w')
    for i in range(len(test_attributes)):
        test_file.write(str(test_labels[i])+" ")
        for j in range(len(test_attributes[i])):
            test_file.write(str(test_attributes[i][j])+" ")
        test_file.write("\n")
        
    for i in range(len(train_attributes)):
        train_file.write(str(train_labels[i])+" ")
        for j in range(len(train_attributes[i])):
            train_file.write(str(train_attributes[i][j])+" ")
        train_file.write("\n")

    print('\nFINISH!')
