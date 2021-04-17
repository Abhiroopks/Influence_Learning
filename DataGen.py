import numpy as np
import itertools


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
    
def genSamples(network):

    N = network.num_nodes

    out = []
    
    lst = itertools.product([0, 1], repeat=N)

    for init_infl in lst:
                        
        # run the simulation and get the output matrix
        
        matrix = network.simulate(list(init_infl))
        out.append(matrix)

    return out
            
            
            
            
            
            

        
        
    
                                        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
                
        
            
            
                        
                        
        
        
        