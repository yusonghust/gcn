# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def emb_reduction(X):
    tsne = TSNE(n_components=2, perplexity=10, init='pca', random_state=0, n_iter=5000, learning_rate=0.1)
    emb= tsne.fit_transform(embeddings)
    return emb

def plot_embeddings(X,nodes,labels,savefile):
    '''
    Args:
    X: embeddings.
    nodes: node ids
    labels: node labels
    '''
    embs = emb_reduction(X)[nodes]
    x = embs[:,0]
    y = embs[:,1]
    colors = []
    d = {0:'tomato', 1:'blue', 2:'lightgreen', 3:'lightgray', 4: 'orange', 5: 'purple'}
    for i in range(len(labels)):
        colors.append(d[labels[i]])
    plt.scatter(x, y, s=200, c=colors)
    for x,y, node in zip(x, y, nodes):
        plt.text(x, y, node, ha='center', va='center', fontsize=8)
    plt.savefig(savefile)



