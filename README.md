# Graph Convolutional Network  
## Basic Introduction  
An much simpler and easy understand graph convolutional network.  

[original gcn code can be found here](https://github.com/tkipf/gcn)  
[A pretty nice gcn blog can be found here](https://tkipf.github.io/graph-convolutional-networks/)  
[datasets are from here](https://github.com/thunlp/OpenNE)

This implementation of gcn is easier to understand, which will be more friendly to newcomer.  

## Usage:  
**Input:**  
an edgelist and a label file are must. If feature file is available, gcn will use it for learning.  
If we don't have fearture matrix, the fearure matrix will be replaced by an identity matrix, as we don't have any node features. Besides, we also need to provied the total label numbers for node classification.  
**Parameters:** 
--edgelist: edgelist file, looks like node1 node2 <weight_float, optional>;  
--weighted: treat the graph as weighted; this is an action;  
--directed: treat the graph as directed; this is an action;  
--labels: node labels, looks like node_id label_id;  
--features: node features, looks like node feature_1 feature_2 ... feature_n;  
--label_nums: number of labels. for Wiki dataset, 17 labels; for cora dataset, 7 labels;  
--lr: learning rate, default is 0.01;  
--epochs: training epochs, default is 200;  
--act: activation function, default is relu;  
--clf-retio: the ratio of training data for node classification; the default is 0.5;   

