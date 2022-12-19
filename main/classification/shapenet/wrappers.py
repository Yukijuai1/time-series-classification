import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import math
import numpy
import torch
import random
import sklearn
import sklearn.linear_model
# from sklearn.externals import joblib
import joblib
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import timeit
import matplotlib.pyplot as plt
from numba import njit, jit, objmode
from numba.typed import List
import time
from scipy.cluster.vq import vq
from sklearn.metrics import classification_report

import shapenet.utils as utils
import shapenet.losses as losses
import shapenet.networks as networks
import shapenet.slide as slide

class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):

    def __init__(self, compared_length,
                 batch_size, epochs, lr,
                 encoder, params, in_channels, cuda=False, gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.loss = losses.triplet.PNTripletLoss(
            compared_length, cuda, gpu
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)


    def save_shapelet(self, prefix_file, shapelet, shapelet_dim, utility_sort_index):
        '''
        write the shapelet and its dimension to file
        '''
        # save shapelet
        fo_shapelet =open(prefix_file+"shapelet.txt", "w")
        for j in range(len(shapelet)):
            shapelet_tmp = numpy.asarray(shapelet[j])
            s = shapelet_tmp.reshape(1,-1)
            numpy.savetxt(fo_shapelet, s)
        fo_shapelet.close()

        # save shapelet variable
        fo_shapelet_dim = open(prefix_file+"shapelet_dim.txt", "w")
        numpy.savetxt(fo_shapelet_dim, shapelet_dim)
        fo_shapelet_dim.close()

        # save utility sort index
        fo_utility =open(prefix_file+"utility.txt", "w")
        numpy.savetxt(fo_utility, utility_sort_index)
        fo_utility.close()

    def load_shapelet(self, prefix_file):
        '''
        load the shapelet and its dimension from disk
        '''
        # save shapelet
        fo_shapelet = prefix_file+"shapelet.txt"
        with open(fo_shapelet, "r") as fo_shapelet:
            shapelet_text = []
            for line in fo_shapelet:
                shapelet_text.append(line)
        fo_shapelet.close()
        shapelet = []
        for row in shapelet_text:
          shapelet.append(numpy.fromstring(row, dtype='float64', sep=' '))

        # save shapelet dimension
        fo_shapelet_dim = open(prefix_file+"shapelet_dim.txt", "r")
        shapelet_dim = numpy.loadtxt(fo_shapelet_dim)
        fo_shapelet_dim.close()
        shapelet_dim = shapelet_dim.astype('int32')

        # 添加
        # load shapelet utility index
        fo_utility_sort_index = open(prefix_file+"utility.txt", "r")
        utility_sort_index = numpy.loadtxt(fo_utility_sort_index)
        fo_utility_sort_index.close()
        utility_sort_index = utility_sort_index.astype('int32')

        return shapelet, shapelet_dim, utility_sort_index

    def save_encoder(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_svm_linear(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an svm linear
        classifier.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        self.classifier = SVC(kernel='linear' ,gamma='auto')
        self.classifier.fit(features, y)

        return self.classifier

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = utils.Dataset(X)
        
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )


        #epochs = 0 # Number of performed epochs

        # 保存loss的变化
        loss_value = []

        # Encoder training
        for i in range(self.epochs):
            epoch_start = timeit.default_timer()
            for batch in train_generator:
                batch_start = timeit.default_timer()
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                loss = self.loss(
                   batch, self.encoder, self.params, save_memory=save_memory
                )
                # 保存loss
                loss_value.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                batch_end = timeit.default_timer()
                print("batch time: ", (batch_end- batch_start)/60)

            #epochs += 1
            epoch_end = timeit.default_timer()
            print(f"epoch {i} loss: ", loss.item())
            print(f"epoch {i} time: ", (epoch_end- epoch_start)/60)


        fo_loss =open("loss.txt", "w")
        numpy.savetxt(fo_loss, loss_value)
        fo_loss.close()

        # 绘制loss的变化图像并保存
        plt.plot(loss_value)
        plt.ylabel('Train Loss')
        plt.savefig('Loss.png', dpi=300)

        return self.encoder

    def fit(self, X, y, test, test_labels, prefix_file, cluster_num, save_memory=False, verbose=False, load=False):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param test testing set.
        @param test_labels testing labels.
        @param prefix_file prefix path.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        final_shapelet_num = 50
        
        # Fitting encoder
        if not load:
            print("train encoder")
            encoder_start = timeit.default_timer()
            self.encoder = self.fit_encoder(
                                            X, y=y, save_memory=save_memory, verbose=verbose
                                            )
            encoder_end = timeit.default_timer()
            print("encode time: ", (encoder_end- encoder_start)/60)
            self.save_encoder(prefix_file)
        else:
            print("load encoder")
            self.load_encoder(prefix_file)

        # shapelet discovery
        discovery_start = timeit.default_timer()
        shapelet, shapelet_dim, utility_sort_index = self.shapelet_discovery(X, y, cluster_num, batch_size=50)
        discovery_end = timeit.default_timer()
        self.save_shapelet(prefix_file, shapelet, shapelet_dim, utility_sort_index)
        print("discovery time: ", (discovery_end- discovery_start)/60)
        

        # shapelet transformation
        transformation_start = timeit.default_timer()
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        transformation_end = timeit.default_timer()
        print("transformation time: ", (transformation_end - transformation_start)/60)

        # SVM classifier training
        classification_start = timeit.default_timer()
        self.classifier = self.fit_svm_linear(features, y)
        classification_end = timeit.default_timer()
        print("classification time: ", (classification_end - classification_start)/60)
        print("svm linear Accuracy: "+str(self.score(test, test_labels, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)))
        features = self.shapelet_transformation(test, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        predicted=self.classifier.predict(features)
        t = classification_report(test_labels,predicted, target_names=['1','2', '3'])
        print(t)
        return self

    def encode(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                features[
                    count * batch_size: (count + 1) * batch_size
                ] = self.encoder(batch)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def shapelet_discovery(self, X, train_labels, cluster_num, batch_size = 50):
        '''
        slide raw time series as candidates
        encode candidates
        cluster new representations
        select the one nearest to centroid
        trace back original candidates as shapelet
        '''

        slide_num = 3   # 原来是 3
        alpha = 0.6
        beta = 6
        count = 0
        X_slide_num = []
        gama = 0.5
        
        # 将所有的子时间都保存到一个字典里
        subTimeSeriesDict = {}

        print("\nStart sliding the raw time series and the corresponding class and variate label")
        sliding_start = timeit.default_timer()
        for m in range(slide_num):
            print(m)
            # slide the raw time series and the corresponding class and variate label
            X_slide, candidates_dim, candidates_class_label = slide.slide_MTS_dim_step(X, train_labels, alpha)
            X_slide_num.append(numpy.shape(X_slide)[0])

            subTimeSeriesDict[m] = X_slide
            
            print(f"X_slide shape: {X_slide.shape}")
            print(f"candidates_dim: {X_slide.shape}")
            print(f"candidates_class_label: {X_slide.shape}")
            beta =  beta -2
            alpha = beta/10
            
            test = utils.Dataset(X_slide)
            test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size)

            self.encoder = self.encoder.eval()

            # encode slide TS
            with torch.no_grad():
                print('batch number: ', len(test_generator))
                for batch in test_generator:
                    
                    if count % 100 == 0:
                        print('count: ', count) # 输出次数
                    
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    
                    # 2D to 3D
                    batch.unsqueeze_(1)
                    batch = self.encoder(batch)
                    
                    if count == 0:
                        representation = batch
                        representation = representation.cpu()
                        
                    else:
                        representation = numpy.concatenate((representation, batch.cpu()), axis=0)
                    count += 1
            self.encoder = self.encoder.train()
            count = 0
            
            # concatenate the new representation from different slides
            if m == 0 :
                representation_all = representation
                representation_dim = candidates_dim
                representation_class_label = candidates_class_label
            else:
                representation_all = numpy.concatenate((representation_all, representation), axis = 0)
                representation_dim = representation_dim + candidates_dim
                representation_class_label = numpy.concatenate((representation_class_label, candidates_class_label), axis=0)
            
            print(f"representation_all shape: {representation_all.shape}")
        
        sliding_end = timeit.default_timer()
        print("End sliding")
        print(f"siding time: ", (sliding_end - sliding_start)/60)

        print("\n* All the shapelets info:")
        print(representation_all.shape)
        print(len(representation_dim))
        print(representation_class_label.shape)


        print(f"all subTs num: {X_slide_num}")
        

        # cluster all the new representations
        print('\nStart clustring all the new representations...')  
        cluster_start = timeit.default_timer()
        num_cluster = cluster_num
        kmeans = KMeans(n_clusters = num_cluster)
        kmeans.fit(representation_all)
        cluster_end = timeit.default_timer()
        print('End clustring')
        print("clustring time:", (cluster_end - cluster_start)/60)


        # init candidate as list
        candidate = []
        candidate_dim = numpy.zeros(num_cluster)
        
        # two parts of utility function
        candidate_cluster_size = []
        candidate_first_representation = []
        utility = []


        # select the nearest to the centroid #
        
        # 计算出每个簇有几个元素
        @njit
        def count_candidate_cluster_size(labels, cluster_num):
            candidate_cluster_size = numpy.array([0] * cluster_num)
            for i in labels:
                candidate_cluster_size[i] += 1
            return candidate_cluster_size
        print("\ncalculate elements number in each cluster")
        start_time = time.time()
        candidate_cluster_size = count_candidate_cluster_size(kmeans.labels_, cluster_num)
        print(f"time: {time.time() - start_time}\n")
        
        
        # 计算出离簇新最近的那些点的下标
        print("get the nearest index")
        start_time = time.time()
        nearest_index, dst = vq(kmeans.cluster_centers_, representation_all)
        print(f"time: {time.time() - start_time}\n")

        # 根据下标encode以后的表示存下来
        for i in nearest_index:
            candidate_first_representation.append(representation_all[i])
        # candidate_first_representation = np.array(candidate_first_representation)

        # 计算出candidate
        candidate = []
        candidate_dim = []
        index = 0
        key = 0

        for i in nearest_index:
            for j in range(len(X_slide_num)):
                if(i < X_slide_num[j]):
                    index = i - sum(X_slide_num[:j])
                    key = j
                    break
            candidate.append(subTimeSeriesDict[key][index])
            candidate_dim.append(representation_dim[index])
        

        # utility
        print('Start scoring all the candidate...')
        for i in range(num_cluster):
            ed_dist_sum = 0
            for j in range(len(candidate_first_representation)):
                ed_dist_sum += numpy.linalg.norm(candidate_first_representation[i] - candidate_first_representation[j])
            utility.append(gama * candidate_cluster_size[i] + (1-gama) * ed_dist_sum)
        print('End scoring')

        # sort utility namely candidate
        utility_sort_index = numpy.argsort(-numpy.array(utility))

        return candidate, candidate_dim, utility_sort_index

    def shapelet_transformation(self, X, candidate, candidate_dim, utility_sort_index, final_shapelet_num):
        '''
        transform the original multivariate time series into the new one vector data space
        transformed date label the same with original label
        '''
        # init transformed data with list
        feature = []

        # transform original time series
        for i in range(numpy.shape(X)[0]):
            for j in range(final_shapelet_num):
            #for j in range(len(candidate)):
                dist = math.inf
                candidate_tmp = numpy.asarray(candidate[utility_sort_index[j]])
                for k in range(numpy.shape(X)[2]-numpy.shape(candidate_tmp)[0]+1):
                    difference = X[i, int(candidate_dim[utility_sort_index[j]]), 0+k : int(numpy.shape(candidate_tmp)[0])+k] - candidate_tmp
                    feature_tmp = numpy.linalg.norm(difference)
                    if feature_tmp < dist:
                        dist = feature_tmp
                feature.append(dist)

        # turn list to array and reshape
        feature = numpy.asarray(feature)
        feature = feature.reshape(numpy.shape(X)[0], final_shapelet_num)

        return feature

    def predict(self, X, batch_size=50):
        """
        Outputs the class predictions for the given test data.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.shapelet_transformation(X, shapelet, shapelet_dim, utility_sort_index, final_shapelet_num)
        return self.classifier.score(features, y)

class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.

    @param compared_length Length of the compared positive and negative samples
           in the loss. Ignored if None, or if the time series in the training
           set have unequal lengths.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param epochs Number of epochs to run during the training of the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=50, batch_size=1, epochs=100, lr=0.001,
                 channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, batch_size,
            epochs, lr,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # First applies the causal CNN
                output_causal_cnn = causal_cnn(batch)
                after_pool = torch.empty(
                    output_causal_cnn.size(), dtype=torch.double
                )
                if self.cuda:
                    after_pool = after_pool.cuda(self.gpu)
                after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                # Then for each time step, computes the output of the max
                # pooling layer
                for i in range(1, length):
                    after_pool[:, :, i] = torch.max(
                        torch.cat([
                            after_pool[:, :, i - 1: i],
                            output_causal_cnn[:, :, i: i+1]
                         ], dim=2),
                        dim=2
                    )[0]
                features[
                    count * batch_size: (count + 1) * batch_size, :, :
                ] = torch.transpose(linear(
                    torch.transpose(after_pool, 1, 2)
                ), 1, 2)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, batch_size, epochs, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size, epochs, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self
