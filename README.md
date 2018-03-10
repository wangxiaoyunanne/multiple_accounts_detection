# multiple_accounts_detection

################################
To run sample dataset 
################################

Quick start

1 Using Katz unsupervised learning and semi-supervised learning with Katz

python katz_sim.py

You could explore the trade off between precision and recall by changing the "thres" in line 106. 

2 Using semi-supervised learning with graph embbing 

python test_norm_2.py

3 Using clustering methods to improve scalability. Here I first tried kmeans then changed to spectral clustering

python test_kmeans.py

You could change the num_group to see how the number of clusters effect on the performance in both speed and accuracy.

################################
Requirement
################################

Python 2.7 

NumPy 1.13

Pandas 0.22

NetworkX 2.1

sklearn 0.19.1

################################
Reference
################################

Multiple Accounts Detection on Facebook Using Semi-Supervised Learning on Graphs

Xiaoyun Wang, Chun-Ming Lai, Yunfeng Hong, Cho-Jui Hsieh, S. Felix Wu

https://arxiv.org/abs/1801.09838






  
