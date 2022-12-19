import os
import json
import torch
import numpy
import argparse
import timeit
import pandas as pd
import classification.shapenet.wrappers as wrappers

train_filepath='data/data_simple/train2.txt'
test_filepath='data/data_simple/test2.txt'

def read_datasets(train_filepath,test_filepath):
    train_df=pd.DataFrame(numpy.loadtxt(train_filepath))
    test_df=pd.DataFrame(numpy.loadtxt(test_filepath))
    X=train_df[train_df.columns[1:]].values
    test1=test_df[test_df.columns[1:]].values
    train=[]
    test=[]
    for i in range(len(X)):
        line=[]
        line.append(X[i])
        train.append(line)
    for i in range(len(test1)):
        line=[]
        line.append(test1[i])
        test.append(line)
    train_labels=train_df[train_df.columns[0]].values.astype(int)
    test_labels =test_df[test_df.columns[0]].values.astype(int)
    return numpy.array(train),train_labels,numpy.array(test),test_labels


def fit_parameters(file, train, train_labels, test, test_labels, cuda, gpu, save_path, cluster_num,
                        save_memory=False, load=False):
    """
    Creates a classifier from the given set of parameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    
    classifier = wrappers.CausalCNNEncoderClassifier()


    # Loads a given set of parameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['in_channels'] = 1
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, test, test_labels, save_path, cluster_num, save_memory=save_memory, verbose=True, load=load
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of parameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False
    '''
    train, train_labels, test, test_labels = load_UEA_dataset(
        args.path, args.dataset
    )
    print(numpy.size(train,0),numpy.size(train,1),numpy.size(train,2))
    '''

    train, train_labels, test, test_labels = read_datasets(train_filepath,test_filepath)
    print(numpy.size(train,0),numpy.size(train,1),numpy.size(train,2))
    
    
    # !!! CLUSTER 数量没有定义
    cluster_num = 200
    # 训练新的模型
    if not args.load and not args.fit_classifier:
        print('start new network training')
        classifier = fit_parameters(
            args.hyper, train, train_labels, test, test_labels, args.cuda, args.gpu, args.save_path, cluster_num,
            save_memory=False
        )
    # 用已有的模型或者分类器
    else:
        classifier = wrappers.CausalCNNEncoderClassifier()
        hf = open(
            os.path.join(
                args.save_path, args.dataset + '_parameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        classifier.set_params(**hp_dict)
        print("load cnn encoder")

        
        # 在已有encoder的情况下
        classifier = fit_parameters(
            args.hyper, train, train_labels, test, test_labels, args.cuda, args.gpu, args.save_path, cluster_num,
            save_memory=False, load=args.load
        )

        
        

    if not args.load:
        if args.fit_classifier:
            classifier.fit_classifier(classifier.encode(train), train_labels)
        classifier.save(
            os.path.join(args.save_path, args.dataset)
        )
        with open(
            os.path.join(
                args.save_path, args.dataset + '_parameters.json'
            ), 'w'
        ) as fp:
            json.dump(classifier.get_params(), fp)

    end = timeit.default_timer()
    print("All time: ", (end- start)/60)
