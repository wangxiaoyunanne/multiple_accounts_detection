from numpy import linalg as LA
import pandas as pd
import numpy as np
import random
import math
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import roc_auc_score
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import networkx as nx
import glob
from scipy.stats import itemfreq
from collections import Counter


data = pd.read_table('sample_file/page_user_multi_edgelist.csv',sep =',' , header = None)
graph = nx.from_pandas_dataframe(data, 0,1, create_using=nx.MultiGraph() )
print 'number of nodes ' + str(graph.number_of_nodes())
print 'number of edges ' + str(len(data))
print 'number of edges in graph ' + str(graph.number_of_edges())
random.seed(7)

page_freq = Counter(data.iloc[:,0])
user_freq = Counter(data.iloc[:,1])

page_nodes = list(set(data.iloc[:,0]))
#get 100 most active ID
freq_1000 = user_freq.most_common()
for num in range(1):
    index_n = num
    sample_index = np.asarray(random.sample(range(index_n* 200,index_n* 200+400), 100))
    ID_100 = []
    for i in sample_index:
        ID_100.append(freq_1000[i][0])
    ID_100.extend(page_freq)
# get sub graph of this 100 ID
    subGraph2 = nx.MultiGraph()
    subGraph2 = graph.subgraph(ID_100)
# store subGraph2 to dataframe
#data2 = nx.to_pandas_dataframe(subGraph2)
    graph_file = 'sample_file/shuffled/data2/2_'+str(num)+'.edgelist' 
    nx.write_edgelist(subGraph2, graph_file)




