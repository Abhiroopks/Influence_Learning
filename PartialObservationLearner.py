import numpy as np
import importlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import random

"""
Learns the edge weights for the entire network. Here we are given partial observations:
Only the starting configuration of the network, and the ending. 

Parameters
----------
input_data: (S x (N x N)) matrix
    S: # of simulations - each simulation is an (N x N) matrix,
       but for partial observations, we can only use the first and last rows of these matrices.
    N: # of nodes in the network
    
test_size : float
    fraction of input to use for validation
    
num_layers: int
    number of ADDITIONAL layers to add to the neural network.
    
fraction: float
    fraction of data to actually use
    
Returns
----------
hyp: N x (N+1) weights corresponding to the edge weights for each node, as well as their thresholds.
    
"""
    
def learn(input_data, test_size, num_layers, fraction):

    (X,Y) = process(np.array(input_data))
    
    # test_size is large because we are given ALL data possible
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=42)
    
    X_train = X_train[:int(fraction*X_train.shape[0])]
    Y_train = Y_train[:int(fraction*Y_train.shape[0])]
    
    
    num_nodes = X.shape[1]
        
    model = Sequential()

    kernel_initializer = tf.keras.initializers.Constant(0.0)
    bias_initializer = tf.keras.initializers.Constant(0.0)

    # Add an input layer
    input_layer = Dense(num_nodes,use_bias=True, activation='tanh', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(num_nodes,))
    
    model.add(input_layer)
    
    # Add intermediate layers
    for i in range(num_layers):
        model.add(input_layer)
    
    model.compile(loss='mse',
              optimizer=tf.keras.optimizers.SGD(learning_rate=2))
                   
    model.fit(X_train, Y_train,epochs=50, batch_size=1, verbose=0)
    
    sum_err = 0
    
    for i in range(len(X_train)):
        y_pred = model.predict(X_train[i].reshape(1,-1))
        sum_err += getLoss(y_pred, Y_train[i])

    train_err = sum_err / (Y_train.shape[0] * Y_train.shape[1])
    #print(f"final train error rate: {train_err}")
    
    sum_err = 0
    
    # Testing error
    for i in range(len(X_test)):
        y_pred = model.predict(X_test[i].reshape(1,-1))
        sum_err += getLoss(y_pred, Y_test[i])
        
    test_err = sum_err / (Y_test.shape[0] * Y_test.shape[1])
        
    # Naive predictors - just classify all nodes as -1 or 1
    naive_err0 = 0
    naive_err1 = 0
    y_pred0 = np.ones((1,num_nodes)) * -1
    y_pred1 = np.ones((1,num_nodes))

    for i in range(len(X_test)):
        naive_err0 += getLoss(y_pred0, Y_test[i])
        naive_err1 += getLoss(y_pred1, Y_test[i])
        
    naive_err0 /= (Y_test.shape[0] * Y_test.shape[1])
    naive_err1 /=(Y_test.shape[0] * Y_test.shape[1])
    
    #print(f"test error rate: {}")
    #print(f"naive 0 classifier test error rate: {naive_err0}")
    #print(f"naive 1 classifier test error rate: {naive_err1}")
        
    return train_err, test_err, naive_err0, naive_err1

     
    

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
