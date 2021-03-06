import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Learns the incoming edge weights and threshold for a particular node in an influence network

Important to note that the datasets given are FULL observations: that is, they show the state of the network
throughout the influence cascades, at every time step. Using this information, we can estimate the incoming weights
of edges for every node.

Parameters
----------
node : int
    index of node for which learner will learn incoming edge weights

input_data: (S x (N x N)) matrix
    S: # of simulations - each simulation is an (N x N) matrix,
       but really each row of each matrix can be used for training
    N: # of nodes in the network
    
test_size: float
    Fraction of input data to use for validation

fraction: foat
    Overall fraction of training data to use
    
Returns
----------
hyp: (N+1) weights corresponding to the incoming edge weights for the local node, and local threshold
    
"""
    
def learn(node, input_data, test_size=0.25, fraction = 1):

    #truncated = random.sample(input_data, int(fraction * len(input_data)))
    data = np.array(input_data)

    (X,Y) = process(node=node, data=data)  
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=42)
    
    # Truncate portion of training data based on specified fraction
    
    use_training = int(fraction * len(X_train))
    X_train = X_train[:use_training - 1]
    Y_train = Y_train[:use_training - 1]
    
    num_dimensions = X_train.shape[1]
        
    num_samples = X_train.shape[0]
        
    # the weights we are learning - init to 0
    hyp = np.zeros(num_dimensions)
        
    # Perceptron Algorithm
    # Although there exists a set of weights with 0 training error, limit the # of rounds for time considerations
    
    done = False
    currRound = 0
    rate = 0.005
    maxRounds = 50

    while not done and currRound < maxRounds:
        currRound += 1
        done = True
        # each training data point x = each row of X
        for i in range(num_samples):

            guess = np.sign(np.dot(X_train[i], hyp))
            
            # Wrong predicton - update weights. Ensure another round of learning occurs
            if(guess*Y_train[i] <= 0):
                hyp += X_train[i]*Y_train[i]*rate
                done = False
            
            
    # calculate final training error (should be 0, or close to it if underlying mechanism is Linear Threshold)
    
    #print("Rounds: " + str(currRound))
    
    err = 0
    
    guesses = np.sign(np.matmul(X_train, hyp) * Y_train)
    for i in range(num_samples):
        if guesses[i] <= 0:
            err += 1
    
    err /= num_samples
    
    train_err = err
                   
    # Validate hypothesis on test set
    err = 0
    naive_0_err = 0
    naive_1_err = 0
    
    guesses = np.sign(np.matmul(X_test, hyp) * Y_test)
    for i in range(X_test.shape[0]):
        if guesses[i] <= 0:
            err += 1
        if Y_test[i] == -1:
            naive_1_err += 1
        elif Y_test[i] == 1:
            naive_0_err += 1
            
    err /= X_test.shape[0]
    naive_0_err /= X_test.shape[0]
    naive_1_err /= X_test.shape[0]
               
    test_err = err
    
    return train_err, test_err, naive_0_err, naive_1_err


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
