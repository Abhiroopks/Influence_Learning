import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

"""
Learns the incoming edge weights and threshold for a particular node in an influence network

Parameters
----------
node : int
    index of node for which learner will learn incoming edge weights

input_data: (S x (N x N)) matrix
    S: # of simulations - each simulation is an (N x N) matrix,
       but really each row of each matrix can be used for training
    N: # of nodes in the network
    
Returns
----------
hyp: (N+1) weights corresponding to the incoming edge weights for the local node, and local threshold
    
"""
    
def learn(node, input_data):

    data = np.array(input_data)
    
    (X,Y) = process(node=node, data=data)  
    
    # test_size is large because we are given ALL data possible
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.95,random_state=42)
    
    num_dimensions = X_train.shape[1]
        
    num_samples = X_train.shape[0]
    
    # the weights we are learning - init to 0
    hyp = np.zeros(num_dimensions)
        
    # Perceptron Algorithm
    # We assume realizability, so keep training until 0 error achieved on training set
    
    done = False
    iter = 0

    while not done:
        iter += 1
        done = True
        # each training data point x = each row of X
        for i in range(num_samples):

            guess = np.sign(np.dot(X_train[i], hyp))
            
            # Wrong predicton - update weights. Ensure another round of learning occurs
            if(guess*Y_train[i] <= 0):
                hyp += X_train[i]*Y_train[i]
                done = False
            
            
    # calculate final training error (should be 0 to leave the above loop)
    
    print("Rounds: " + str(iter))
    
    err = 0
    
    guesses = np.sign(np.matmul(X_train, hyp) * Y_train)
    for i in range(num_samples):
        if guesses[i] <= 0:
            err += 1
    
    err /= num_samples
                   
    print("node " + str(node) + " train error rate: " + str(err))           

    # Validate hypothesis on test set
    err = 0
    guesses = np.sign(np.matmul(X_test, hyp) * Y_test)
    for i in range(X_test.shape[0]):
        if guesses[i] <= 0:
            err += 1 
            
    err /= X_test.shape[0]
                
    print("node " + str(node) + " test error rate: " + str(err))
    
    return hyp


"""
Process the data. Does a few things:

* Deletes simulations for which node of interest is initialized to influenced
* Combines all remaining matrices into a single 2D matrix
* Isolates [node] columns as Y : the 'answers' to our training
* Aligns inputs with proper outputs - each y_i = f(x_i-1). This is to say, the status of each node at time t is a function of the status of the other nodes at time t-1.
* Fills [node] column with 0s for each matrix : signifies no self-weight
* Adds column at end of data filled with -1s : for the threshold
* Converts 0s in Y to -1s

Parameters
----------
node: int
    index of node of interest

data: (S x (NxN)) training matrix

Returns
----------
X: training matrix with various configurations of the nodes
Y: array for the state of the node of interest


"""
def process(node, data):
    
    S = data.shape[0]
    num_nodes = data.shape[1]
    
    # Delete the cascades for which [node] is initially influenced - it doesn't make sense to learn the incoming weights here
    # Because it is ALWAYS influenced - not a function of other nodes' influences
    
    toDelete = []
    
    for i in range(S):
        if(data[i][0][node] == 1):
            toDelete.append(i)
            
    
    data = np.delete(data, toDelete, axis=0)
    
    # flatten over all simulations
    X = data.reshape(data.shape[0] * num_nodes, num_nodes)
    
    # isolate Y as the [node] column
    Y = np.copy(X[:,node])
        
    # Delete last row of each remaining matrix of X
    # Delete first row of each Y
    # Because y_i = f(x_i-1)
    
    X = np.delete(X, np.arange(num_nodes - 1, X.shape[0], num_nodes), axis=0)
    Y = np.delete(Y, np.arange(0, Y.shape[0], num_nodes), axis=0)    
        
    # add col of -1s to end of X for threshold learning
    X = np.c_[X, -1*np.ones(X.shape[0])]
    
    # fill 0s in [node] col of X
    X[:,node] = 0
    
    # replace 0s with -1s in Y
    Y = np.where(Y == 0, -1, Y)
    
    return X,Y
