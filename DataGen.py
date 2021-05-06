import numpy as np
import pandas as pd
import itertools
import Network


"""
Runs simulations of influence and returns data generated.
Goes through all possible 2^N initial configs as inputs to simulations

Parameters
----------
network : Network
    Network to be simulated. Assumes weights and thresholds already initialized

Returns
----------
out : 2^N x (N x N) array
    N: # of nodes in the network

"""
    
def genSamples(network, visualize=False):

    N = network.num_nodes

    out = []
    
    lst = itertools.product([0, 1], repeat=N)

    for init_infl in lst:
                        
        # run the simulation and get the output matrix
        
        matrix = network.simulate(list(init_infl), visualize=visualize)
        out.append(matrix)

    return out


'''
Builds matrix cascades from the SMH datasets. Each csv file is one cascade

'''
def smhData():
    
    out = []
    
    # number of unique users in SMH dataset
    N = 1695
    
    
    for i in range(20):
        df = pd.read_csv(f'Datasets/SMH/SMH-cascade-{i}.csv')
        
        matrix = np.zeros((N,N))
        prev_row = np.zeros(N)
        
        for index, row in df.iterrows():
            user_id = row['user_id']
            matrix[index] = np.copy(prev_row)
            matrix[index][user_id] = 1
            prev_row = matrix[index]
            
        # append redundant rows to end (for consistency)
        for i in range(N - df.shape[0]):
            matrix[df.shape[0] + i] = matrix[df.shape[0] - 1]
            
        out.append(matrix)
        
    return out
            
            
    
'''
Constructs graph from Kissler Dataset, and generates data from simulations

'''
def kisslerData():
    
    df = pd.read_csv('Datasets/Kissler_DataS1.csv')    
    
    # Find average distance between individuals over all time steps
    df = df.values
    
    N = 469
    
    sum_distance = np.zeros((N,N))
    
    print(df.shape)
    
    
    
    
    
            

        
        
    
                                        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
                
        
            
            
                        
                        
        
        
        