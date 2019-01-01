# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import numpy as np
import time
from utils import *
from graph import Graph
import scipy.sparse as sp
from layers import GraphConvLayer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--edgelist',required=True,help='input edgelist file')
    parser.add_argument('--weighted',action='store_true',help='treat graph as weighted')
    parser.add_argument('--directed',action='store_true',help='treat graph as directed')
    parser.add_argument('--labels',required=True,help='input labels file')
    parser.add_argument('--label_nums',required=True,type=int,help='label numbers')
    parser.add_argument('--features',default=None,help='input features file')
    parser.add_argument('--epochs',default=200,type=int,help='number of training epochs')
    parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
    parser.add_argument('--clf_ratio',default=0.5,type=float,help='training data ratio')
    parser.add_argument('--act',default='relu',choices=[
        'relu',
        'leaky_relu',
        'tanh',
        'sigmoid'
        ],help='The activation function will be used')
    args = parser.parse_args()
    return args



class GCN():
    def __init__(self,G,out_dims_list,has_features,lr=0.01,epochs=200,act=tf.nn.relu,clf_ratio=0.5):
        '''
        Args:
        G: Graph class.
        out_dims_list: a list of output dimension of each graph conv layer.
        has_features: bool object. if we don't have any node features, feature matrix will be an identity matrix.
        '''
        self.g = G
        self.clf_ratio = clf_ratio
        self.lr = lr
        self.epochs = epochs
        self.out_dims_list = out_dims_list
        self.act = act
        self.adj,self.labels,self.features,self.train_mask,self.val_mask,self.test_mask = preprocess_data(G,clf_ratio,has_features)
        self.build_placeholders()

    def build_placeholders(self):
        num_supports = 1
        self.placeholders = {
            'adj': tf.sparse_placeholder(tf.float32),
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, self.labels.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            # helper variable for sparse dropout
            # 'num_features_nonzero': tf.placeholder(tf.int32)
            'is_training': tf.placeholder(tf.bool)
        }


    def gcn(self):
        L = len(self.out_dims_list)
        indim = self.features[2][1]
        y = self.placeholders['features']
        for i in range(L):
            if i==0:
                sparse = True
            else:
                sparse = False
                y = tf.layers.dropout(y,0.5,training = self.placeholders['is_training'])
            y = GraphConvLayer(input_dim=indim,
                              output_dim=self.out_dims_list[i],
                              name='gc%d'%i,
                              act=self.act)(adj_norm=self.placeholders['adj'],
                                            x= y, sparse=sparse)
            indim = self.out_dims_list[i]
        loss = self.masked_softmax_cross_entropy(y,self.placeholders['labels'],self.placeholders['labels_mask'])
        accuracy = self.masked_accuracy(y,self.placeholders['labels'],self.placeholders['labels_mask'])
        opt = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return y,loss,accuracy,opt


    def construct_feed_dict(self, labels_mask, is_training):
        """Construct feed dictionary."""
        feed_dict = dict()
        feed_dict.update({self.placeholders['is_training']: is_training})
        feed_dict.update({self.placeholders['labels']: self.labels})
        feed_dict.update({self.placeholders['labels_mask']: labels_mask})
        feed_dict.update({self.placeholders['features']: self.features})
        feed_dict.update({self.placeholders['adj']: self.adj})
        # feed_dict.update({self.placeholders['num_features_nonzero']: self.features[1].shape})
        return feed_dict


    def masked_softmax_cross_entropy(self, preds, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)


    def masked_accuracy(self, preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def train_and_evaluate(self):
        ###getembs: if true, return the gcn output before training.###
        output,loss,accuracy,opt = self.gcn()
        ###start training and evaluate gcn model###
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        feed_dict_train = self.construct_feed_dict(self.train_mask,True)
        feed_dict_val = self.construct_feed_dict(self.val_mask,False)
        feed_dict_test = self.construct_feed_dict(self.test_mask,False)

        start = time.time()
        for i in range(self.epochs):
            loss_tr,_,acc_tr = sess.run([loss,opt,accuracy],feed_dict = feed_dict_train)
            if (i+1)%10 == 0:
                loss_v,acc_v = sess.run([loss,accuracy],feed_dict = feed_dict_val)
                end = time.time()
                duration = end - start
                print('-'*150)
                print('step {:d} \t train_loss = {:.3f} \t train_accuracy =  {:.3f} \t val_loss = {:.3f} \t val_accuracy = {:.3f} \t ({:.3f} sec/10_steps)'.format(i+1,loss_tr,acc_tr,loss_v,acc_v,duration))
                start = time.time()

        acc_te = sess.run(accuracy,feed_dict = feed_dict_test)
        print('-'*150)
        print('after training, test accuracy is %f'%acc_te)


def main(args):
    G = Graph(args.edgelist,args.weighted,args.directed,args.labels,args.features)
    out_dims_list = [64,32,args.label_nums]
    if args.features is None:
        has_features = False
    else:
        has_features = True
    if args.act == 'relu':
        act = tf.nn.relu
    elif args.act == 'leaky_relu':
        act = tf.nn.leaky_relu
    elif args.act == 'tanh':
        act = tf.nn.tanh
    elif args.act == 'sigmoid':
        act = tf.nn.sigmoid
    model = GCN(G,out_dims_list,has_features,args.lr,args.epochs,act,args.clf_ratio)
    model.train_and_evaluate()



if __name__ == '__main__':
    main(parse_args())


























