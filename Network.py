import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Network():
    
    """
    Simulates rounds of influence until no more new nodes are influenced in a round
    
    Max Rounds: (N-1), because it is the max length influence cascade
    
    
    Parameters
    ----------
    init_infl : array
        boolean array of initially influenced nodes
    
    visualize: bool
        Whether or not to visualize the cascades.
        
    Returns
    ----------
    out : matrix
        boolean matrix of influenced nodes at each time step.
        Each row corresponds to a point in time and each column is the status of a node
        
    """
    def simulate(self, init_infl, visualize=False):
        
        # running sum of influence input for each node
        infl_sum = [0]*self.num_nodes
                
        # use a set to keep track of newly influenced nodes by index
        infl = set()
        
        for i in range(self.num_nodes):
            if(init_infl[i] == 1):
                infl.add(i)
                
        infl_tot = infl.copy()
        
        prev_row = init_infl.copy()
        # return 2D boolean vector of activated nodes at each time step
        out = []
        out.append(prev_row)

        round = 0
        
        G = None
        labels = None
        layout = None
        colors = None
        
        # Red nodes = influenced
        # grey nodes = not influenced
        if(visualize):
            
            plt.rcParams['figure.dpi'] = 200
            G = nx.from_numpy_matrix(self.weights, create_using=nx.DiGraph)  
            # planar layout is better for smaller and sparser graphs
            layout = nx.planar_layout(G)
            colors = ['grey']*self.num_nodes
            for i in range(self.num_nodes):
                if init_infl[i] == 1:
                    colors[i] = 'red'
            
            #layout = nx.spring_layout(G)
            nx.draw(G, pos = layout, with_labels=True, node_color = colors) 
            labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(G, pos=layout,edge_labels=labels, label_pos=.5)
            plt.show()
        
        
        while len(infl) > 0:
            
            #print("Influenced in round " + str(round) + "\n" + str(infl) + "\n")
                            
            # newly influenced nodes
            for node in infl:
                                
                # outgoing edges
                for i in range(self.num_nodes):
                    infl_sum[i] += self.weights[node][i]

            infl.clear()
            
            new_row = prev_row.copy()
            
            new_infl = False

            # check if any previously unactivated nodes have been activated this round
            for i in range(self.num_nodes):
                
                if(i not in infl_tot and infl_sum[i] > self.thresh[i]):
                    new_row[i] = 1
                    infl.add(i)
                    infl_tot.add(i)
                    new_infl = True
                    
            
            if(new_infl):
                out.append(new_row)
                prev_row = new_row
                
                if(visualize):
                    # update colors of newly influenced nodes to red
                    for node in infl:
                        #print(f'newly infl: {node}')
                        colors[node] = 'red'

                    nx.draw(G, pos = layout, with_labels=True, node_color = colors) 
                    labels = nx.get_edge_attributes(G, "weight")
                    nx.draw_networkx_edge_labels(G, pos=layout,edge_labels=labels, label_pos=.5)
                    plt.show()
                
                
                
                      
            round += 1        
          
        
        # Append copies of the final config IFF cascade length < N
        # This is redundant data, but makes things more consistent by ensuring varying cascade lengths have equal size matrices
        while(len(out) < self.num_nodes):
            out.append(out[-1])
        
        
        return out
    
    
    
class CustomNetwork(Network):
    '''
    Allow user to set weights manually
    Sets thresholds to random values
    
    Parameters
    ----------
    weights: 2D ndarray
            NxN array of weights. Normalize weights here
    
    
    '''
    
    def __init__(self, weights):
        super().__init__()
        
        self.num_nodes = weights.shape[0]
        
        self.weights = weights
        
        # init thresholds to influence a node - (0,1) for each
        self.thresh = np.random.rand(num_nodes)
        
        # round to 2 decimals
        for i in range(num_nodes):
            self.thresh[i] = round(self.thresh[i],2)
        
        # Normalize incoming edge weights for every node
        
        for j in range(self.num_nodes):
            sum_incoming = 0
            
            for i in range(self.num_nodes):
                sum_incoming += self.weights[i][j]
                
            if(sum_incoming > 0):
                for i in range(self.num_nodes):
                    self.weights[i][j] = round(self.weights[i][j]/sum_incoming, 2)                    

                    
class RandomNetwork(Network):

    """
    Randomly initializes:
    * Weights of edges
    * Thresholds
    
    Normalizes:
    * Weights of incoming edges for each node
    
    Also calculates the sum of outgoing weights for each node.
        - Only used for sizing of nodes in networkx visualization
    
    """
    def __init__(self, num_nodes=10, sparsity=0.5, normalize_incoming=True):  
        
        super().__init__()
        
        self.num_nodes = num_nodes
                
        # matrix of weights
        self.weights = np.zeros((num_nodes, num_nodes))        
        
        # initialize weights of edges
        for j in range(num_nodes):
            
            sum_incoming = 0
            
            for i in range(num_nodes):
                
                curr_weight = np.random.rand() 
                
                # make 0 for sparsity
                if(curr_weight > sparsity or i == j):
                    curr_weight = 0
                
                self.weights[i][j] = curr_weight
                sum_incoming += curr_weight
                
                
            # normalize incoming edges to node i by total sum
            if(normalize_incoming):
                if(sum_incoming > 0):
                    for i in range(num_nodes):
                        self.weights[i][j] = round(self.weights[i][j]/sum_incoming, 2)                    
        
        # init thresholds to influence a node - (0,1) for each
        self.thresh = np.random.rand(num_nodes)
        
        for i in range(num_nodes):
            self.thresh[i] = round(self.thresh[i],2)
        
        

    
        
                                        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
                
        
            
            
                        
                        
        
        
        