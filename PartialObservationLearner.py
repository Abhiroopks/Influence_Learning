import numpy as np
import importlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

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
    
    # test_size is large because we are given ALL data possible
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.75,random_state=42)
    
    num_nodes = X.shape[1]
        
    model = Sequential()

    kernel_initializer = tf.keras.initializers.Constant(0.0)
    bias_initializer = tf.keras.initializers.Constant(0.0)

    # Add an input layer
    model.add(Dense(num_nodes,use_bias=True, activation='tanh', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(num_nodes,)))
    
    # Add intermediate layers?
    for i in range(0):
        model.add(Dense(num_nodes,use_bias=True, activation='tanh', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
    
    model.compile(loss='mse',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))
                   
    model.fit(X_train, Y_train,epochs=100, batch_size=1, verbose=1)
    
    sum_err = 0
    
    for i in range(len(X_test)):
        y_pred = model.predict(X_test[i].reshape(1,-1))
        sum_err += getLoss(y_pred, Y_test[i])
    
    print(f"total errors: {sum_err}")
    print(f"test error rate: {sum_err / (Y_test.shape[0] * Y_test.shape[1])}")
        
        
     
    

def getLoss(y_pred, y_true):
    
    err = 0
    
    for i in range(y_pred.shape[1]):
        
        
        if np.sign(y_pred[0, i]) != y_true[i]:
            #print(f"prediction: {y_pred[0, i]}\ntrue: {y_true[i]}\n")
            err += 1
    
    
    return err
    
    
    
    
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
    # add col of -1s to end of X for threshold learning
    #X = np.c_[X, -1*np.ones(X.shape[0])]
    
    Y = data[:,1,:]
    # replace 0s with -1s in Y
    Y = np.where(Y == 0, -1, Y)
   
    return X,Y
