# Learning Linear Threshold Influence Networks

This repository hosts several files that you may use to simulate and learn influence networks.

## Partial Observation Learner
* Only observes the initial and the final state of the network, for multiple cascades
* Builds neural-network using fully-connected layers of N nodes each
  * Each layer **should** simulate a single time step of influence, but testing shows a single layer actually works best
  * Use tanh activation for each node in the layers, and mean squared error loss for stochastic gradient descent learning
## Full Observation Learner
* Observes every time step of every cascade
  * This is a lot of data, but turns out a lot of the data from simulated cascades are redundant (~ 80% redundancy)
* Uses perceptron algorithm to linearly classify the state of each node at any time step
  * Based on the fact that the state of each node is a linear function of incoming edge weights and local threshold
  * **Should** learn the weights to achieve 0 training error, but we limit max # iterations over dataset
  * Overall achieves good training and testing error
  
## SMH Data Set
* 20 cascades of 1695 twitter users
* Performs poorly with partial observations - not enough cascades for learning
* Performs well with full observations

## Notes
* See PDF for written report and deeper discussion
* See the Python Notebook files for examples on how to use the learners, data generators, etc.

## References
[1] Narasimhan, H., Parkes, D., and Singer, Y. (2015). Learnability of
Influence in Networks. NIPS.

[2] David Kempe, Jon M. Kleinberg, and Eva Tardos. Maximizing the
spread of influence through a social network. In KDD, 2003.

[3] Wielki, J. Analysis of the Role of Digital Influencers and Their Impact
on the Functioning of the Contemporary On-Line Promotional System
and Its Sustainable Development. Sustainability 2020, 12, 7138.
https://doi.org/10.3390/su12177138

[4] Kumar, Raj Gaurav Ballabh, ”Evaluating the role of critical nodes in disrupting
diffusion in independent cascade diffusion model” (2019). Graduate
Theses and Dissertations. 17723. https://lib.dr.iastate.edu/etd/17723

[5] Rizoiu, M.-A., Graham, T., Zhang, R., Zhang, Y., Ackland,
R., & Xie, L, Cascade Influence, (2018), GitHub Repository,
https://github.com/computationalmedia/cascade-influence

[6] Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su.
ArnetMiner: Extraction and Mining of Academic Social Networks. In
Proceedings of the Fourteenth ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining (SIGKDD’2008). pp.990-998
