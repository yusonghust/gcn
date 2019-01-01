# -*- coding: utf-8 -*-
import scipy.sparse
import networkx as nx
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import layers as lg
import utils as us
from sklearn.manifold import TSNE


def gcn():
    g = nx.read_edgelist('karate.edgelist',nodetype=int,create_using=nx.Graph())
    
    adj = nx.to_numpy_matrix(g)
    
    # Get important parameters of adjacency matrix
    n_nodes = adj.shape[0]
    
    # Some preprocessing
    adj_tilde = adj + np.identity(n=n_nodes)
    #np.squeeze()--从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))
#    print(adj_norm_tuple)
    
    # Features are just the identity matrix
    feat_x = np.identity(n=n_nodes)
    feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))
    
    # TensorFlow placeholders
    '''
    ###sparse_placeholder demo:###
    x = tf.sparse_placeholder(tf.float32) 
    y = tf.sparse_reduce_sum(x) 
    with tf.Session() as sess: 
        print(sess.run(y)) # ERROR: will fail because x was not fed. 
        indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64) 
        values = np.array([1.0, 2.0], dtype=np.float32) 
        shape = np.array([7, 9, 2], dtype=np.int64) 
        print(sess.run(y, feed_dict={x: tf.SparseTensorValue(indices, values, shape)})) 
        # Will succeed. 
        print(sess.run(y, feed_dict={ x: (indices, values, shape)})) # Will succeed. 
        sp = tf.SparseTensor(indices=indices, values=values, shape=shape) 
        sp_value = sp.eval(session) 
        print(sess.run(y, feed_dict={x: sp_value})) # Will succeed. 
    '''
    ph = {
        'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
        'x': tf.sparse_placeholder(tf.float32, name="x")}
    
    l_sizes = [32,16,8]
    
    o_fc1 = lg.GraphConvLayer(input_dim=feat_x.shape[-1],
                              output_dim=l_sizes[0],
                              name='fc1',
                              act=tf.nn.tanh)(adj_norm=ph['adj_norm'],
                                              x=ph['x'], sparse=True)
    
    o_fc2 = lg.GraphConvLayer(input_dim=l_sizes[0],
                              output_dim=l_sizes[1],
                              name='fc2',
                              act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)
    
    o_fc3 = lg.GraphConvLayer(input_dim=l_sizes[1],
                              output_dim=l_sizes[2],
                              name='fc3',
                              act=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {ph['adj_norm']: adj_norm_tuple,
                 ph['x']: feat_x_tuple}
    
    outputs = sess.run(o_fc3, feed_dict=feed_dict)
    print(outputs.shape)
    nodes = list(g.nodes())
    labels = node2label(nodes)
    return outputs,labels,nodes

def node_id(nodes):
    res = []
    for i in nodes:
        res.append(nodes.index(i))
    return res

def node2label(nodes):
    d = {}
    res = []
    with open("karate.node", 'r') as f:
        lines = f.readlines()
    for line in lines:
        node, label = line.strip().split()
        d[int(node)] = int(label)
    for node in nodes:
        res.append(d[node])
    return res

def emb_reduction(embeddings):
    print("Embedding shape:", embeddings.shape)
    # TSNE's parameter perplexity maybe useful for visualization.
    tsne = TSNE(n_components=2, perplexity=10, init='pca', random_state=0, n_iter=5000, learning_rate=0.1)
    emb= tsne.fit_transform(embeddings)
#    print("After feature reduction:", emb_2.shape)
    return emb
    
def plot_embedding(X, nodes, labels):
    x= X[:, 0]
    y= X[:, 1]
    colors = []
    d = {0:'tomato', 1:'blue', 2:'lightgreen', 3:'lightgray'}
    for i in range(len(labels)):
        colors.append(d[labels[i]])
    plt.scatter(x, y, s=200, c=colors)
    for x,y, node in zip(x, y, nodes):
        plt.text(x, y, node, ha='center', va='center', fontsize=8)
    plt.show()

outputs,labels,nodes = gcn()
nodes = node_id(nodes)
emb = emb_reduction(outputs)
plot_embedding(emb, nodes, labels)
#x_min, x_max = outputs[:, 0].min(), outputs[:, 0].max()
#y_min, y_max = outputs[:, 1].min(), outputs[:, 1].max()
#
#node_pos_gcn = {n: tuple(outputs[j]) for j, n in enumerate(nx.nodes(g))}
#node_pos_ran = {n: (np.random.uniform(low=x_min, high=x_max),
#                    np.random.uniform(low=y_min, high=y_max))
#                for j, n in enumerate(nx.nodes(g))}
#
#all_node_pos = (node_pos_gcn, node_pos_ran)
#plot_titles = ('3-layer randomly initialised graph CNN', 'random')
#
## Two subplots, unpack the axes array immediately
#f, axes = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)
#
#for i, ax in enumerate(axes.flat):
#    pos = all_node_pos[i]
#    ax.set_title(plot_titles[i])
#
#    nx.draw(
#        g,
#        cmap=plt.get_cmap('jet'),
#        node_color=np.log(
#            list(nx.get_node_attributes(g, 'membership').values())),
#        pos=pos, ax=ax)