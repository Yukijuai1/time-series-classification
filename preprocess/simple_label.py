import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

std_path='/home/ubuntu/xuexi/2022_2/lab/shapelet/'

def cal_retweet(start_time,end_time,datalist):
    res_list=[]
    for i in datalist['retweet'].values:
        cnt=0
        retweet_path=i.split(" ")
        for j in retweet_path:
            single_retweet=j.split(":")
            retweet_past_time=int(single_retweet[1])
            if retweet_past_time >= start_time and retweet_past_time < end_time:
                cnt+=1
        res_list.append(cnt)
    return res_list

def cnt_percentage(labels):
    a=0
    b=0
    c=0
    for i in labels:
        if int(i)==1:
            a+=1
        elif int(i)==2:
            b+=1
        elif int(i)==3:
            c+=1
    return a,b,c

def gen_timeSeries(retweet):
    retweet_path=retweet.split(" ")
    cnt=[]
    for i in range(12):
        cnt.append(0)
    for i in retweet_path:
        single_retweet=i.split(":")
        retweet_past_time=int(single_retweet[1])
        for j in range(12):
            if retweet_past_time >=j*15*60 and retweet_past_time < (j+1)*15*60:
                cnt[j]+=1
    
    return cnt



if __name__ == '__main__':

    #读取数据
    print('Loading data...')
    file =open(std_path+'dataset_weibo.txt')
    datalist=[]
    for line in file:
        part=line.split("\t")
        datalist.append(part)
    datalist = pd.DataFrame(datalist)
    datalist.columns = ['message_id', 'root_user_id', 'publish_time', 'retweet_times', 'retweet']

    #打标签
    print('\nCreating label...')  
    labels=[]
    cnt_list1=cal_retweet(0,12*3600,datalist)
    for i in range(len(cnt_list1)):
        if cnt_list1[i]<100:
            labels.append(1)
        elif cnt_list1[i]>=100 and cnt_list1[i]<500 :
            labels.append(2)
        elif cnt_list1[i]>=500:
            labels.append(3)
    a,b,c=cnt_percentage(labels)
    print('percentage: ',a/(a+b+c),b/(a+b+c),c/(a+b+c))

    #制作数据集
    print('\nMaking dataset...')
    features=[]
    for i in datalist['retweet'].values:
        time_series=gen_timeSeries(i)
        features.append(time_series)
    #stand = MinMaxScaler()
    #features = stand.fit_transform(np.array(features))
    train_attributes, test_attributes, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=0)

    #生成数据文件
    print('\nWriting files...')
    train_file=open(std_path+'data/data_simple/train2_normal.txt','w')
    test_file=open(std_path+'data/data_simple/test2_normal.txt','w')
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

