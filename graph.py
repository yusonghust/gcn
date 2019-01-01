# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from utils import sparse_to_tuple
import scipy.sparse as sp

class Graph():
    def __init__(self,edgelist,weighted,directed,labelfile,featurefile):
        self.edgelist = edgelist
        self.weighted = weighted
        self.directed = directed
        self.G = self.build_graph()
        self.node_list = list(self.G.nodes())
        self.look_up = {}
        self.node_size = 0
        for node in self.node_list:
            self.look_up[node] = self.node_size
            self.node_size += 1
        self.labels = self.read_node_labels(labelfile)
        if featurefile is None:
            self.features = np.identity(n=len(self.node_list))
            #scipy.sparse.coo_matrix: A sparse matrix in COOrdinate format.
            #Where A[i[k], j[k]] = data[k].
            self.features = sparse_to_tuple(sp.coo_matrix(self.features))
        else:
            self.features = self.read_node_features(featurefile)


    def build_graph(self):
        '''
        Reads the input network using networkx.
        '''
        if self.weighted:
            G = nx.read_edgelist(self.edgelist, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(self.edgelist, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

        if not self.directed:
            G = G.to_undirected()
        return G

    def read_node_labels(self,filename):
        '''
        read node labels
        '''
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[int(vec[0])]['label'] = vec[1:]
        fin.close()

    def read_node_features(self,filename):
        '''
        read node features
        '''
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[int(vec[0])]['feature'] = np.array([float(x) for x in vec[1:]])
        fin.close()


