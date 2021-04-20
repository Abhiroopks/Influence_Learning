import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Learns the edge weights for the entire network. Here we are given partial observations:
Only the starting configuration of the network, and the ending. 

Parameters
----------
input_data: (S x (N x N)) matrix
    S: # of simulations - each simulation is an (N x N) matrix,
       but for partial observations, we can only use the first and last rows of these matrices.
    N: # of nodes in the network
    
Returns
----------
hyp: N x (N+1) weights corresponding to the edge weights for each node, as well as their thresholds.
    
"""
    
def learn(input_data):

    (X,Y) = process(np.array(input_data))
        


"""
Process the data. Does a few things:

* Deletes all rows for each NxN matrix (data[i]) except for first and last
* Isolates last row of each data[i] as Y[i]
* Isolates first row of each data[i] as X[i]

Parameters
----------
data: (S x (NxN)) training matrix

Returns
----------
X: training matrix with various start configurations
Y: 2D array for the end state of each starting config


"""
def process(data):
    
    S = data.shape[0]
    num_nodes = data.shape[1]          
    
    data = np.delete(data, np.arange(start=1, stop=num_nodes-1, step=1), axis=1)
    
    X = data[:,0,:]
    Y = data[:,1,:]
   
    return X,Y
