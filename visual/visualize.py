import matplotlib.pyplot as plt
import numpy as np
import random

FILEPATH='data/data_simple/train2.txt'

def visualize_scatter(index,datax, datay):
    window=random.randint(1,80000)
    plt.style.use('ggplot')
    colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
              '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    for i in range(window,window+1000):
        # 绘制单条折线图
        plt.plot(index, # x轴数据
         datax[i], # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 2, # 折线宽度
         color = colors[datay[i]], # 折线颜色
         marker = markers[2], # 折线图中添加圆点
         markersize = 4, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='brown', # 点的填充色
         )
        
    #对于X轴，只显示x中各个数对应的刻度值
    plt.xticks(fontsize=8, )  #改变x轴文字值的文字大小
    plt.xlabel('time/min')  # x轴
    plt.ylabel('value')  # y轴
    plt.title('Data Visualize')  # 图像的名称
    plt.show()

def read_datasets(filepath):
    file=open(filepath)
    x=[]
    y=[]
    for line in file:
        part=line.split(" ")
        list=[]        
        for i in range(1,len(part)-1):
            list.append(float(part[i]))
        x.append(list)
        y.append(int(part[0]))
    return np.array(x), np.array(y)


if __name__ == '__main__':
    
    #读取数据
    data_X,data_y=read_datasets(FILEPATH)

    index=[]
    for i in range(len(data_X[0])):
        index.append((i+1)*15)

    #绘制散点图
    visualize_scatter(np.array(index),data_X,data_y)