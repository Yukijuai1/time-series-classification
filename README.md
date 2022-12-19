<h1><div align = "center"><font size="6"><b>天津大学《机器学习》课程设计</b></font></div></h1>
<div align = "center"><font size="6"><b>基于时间序列分类的微博热度预测</b></font></div>

## 目录介绍

- `data`  训练集和测试集数据。`data_cluster`为聚类生成的标签，`data_simple`为正常计算的标签。带`_normal`的文件数据已经过归一化
- `preprocess` 数据预处理代码
- `result` 五类分类器的分类结果
- `visual` 可视化代码
- `main` 分类器代码：
  1.  `classification` 五种分类器
  2. `cluster` K-Means聚类
  3. `model` Shapelet模型文件

## 环境

​	测试计算资源配置：

- Ubuntu 16.04  
- 20  Intel(R) Core(TM) i9-10900F CPU @ 2.80GHz
- 128G内存

​	代码运行环境：

- Python 3.7

​	依赖：

- sklearn
- torch
- numpy
- pandas
- scipy
- saxpy
- timeit
- matplotlib
- numba
- pyts

## 部署步骤

1. 下载数据集:https://pan.baidu.com/s/1c2rnvJq 密码ijp6 
命名为dataset_weibo.txt 放置在项目文件夹下

2. 若执行数据可视化，需修改`visualize.py`中`FILEPATH`变量为想要可视化的数据文件,后执行：

   ```shell
   python visual/visualize.py
   ```

3. 若执行数据预处理(已预处理完成)，需修改`std_path`变量，后执行：

   ```shell
   python preprocess/cluster_label.py
   ```
   三个`.py`文件代表三种标签生成方式。若想生成归一化数据，需去除`.py`文件中下列注释前的`#`：

   ```python
   #Scaler = MinMaxScaler()
   #features = Scaler.fit_transform(np.array(features))
   ```

4. 执行五种分类器的命令见`code.sh`。其中，SVM、KNN和LearningShapelet为同一文件，根据需求调整`classify.py`中下列注释即可：

   ```python
   print('\nBuilding classifier...')
   clf=knn()
   #clf=SVM()
   #clf = LearningShapelet()
   ```

   且可以修改`train_filepath`和`test_filepath`变量以测试不同数据集
