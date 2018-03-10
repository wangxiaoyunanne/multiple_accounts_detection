#######katz_sim.py
import pandas as pd
import igraph
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from numpy.linalg import pinv
from numpy import linalg as LA
from scipy import stats
from sklearn.preprocessing  import normalize
import math
import random
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import roc_auc_score
from scipy import spatial
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from scipy.linalg import get_blas_funcs
import glob

#bipart = pd.read_table('sample_file/name_id_49070.txt', sep = '\t', header = None)
files = glob.glob('sample_file/pop_name/*.txt')
files.sort()

bipart = pd.read_table('sample_file/pop_name/name0000000000000095_00009384.txt', sep = ' ', header = None)
#G = nx.from_pandas_dataframe(bipart, 0, 1)
#A = nx.adjacency_matrix(G).todense()
# here A is symmetric matrix
def katz (MAT, norm = True):
    alpha = 1/(LA.norm(MAT))*0.99
    I = np.identity(MAT.shape[0])
    distence = pinv(I- alpha*MAT)- I
    if norm == False:
        return distence
    if norm == True:
        distence_norm = normalize(distence, norm = 'l1')
        return distence_norm

# sub_edgelist 
def sub_edgelist( bipart, threshold):
    node_list = []
    drop_node = []
    user_id = list(bipart.ix[:,0])
    G = nx.from_pandas_dataframe(bipart, 0, 1)
    for n in G.nodes() :
        #print G.degree(n)
        #print (n in user_id)
        if n in user_id:
            if G.degree(n) > threshold :
                node_list.append(n)
            else :
                drop_node.append(n)
    index_bi_t=[]
    for n in node_list:
        index_bi = np.where(user_id == n)[0][0]
        index_bi_t.append(index_bi)
    data_bipart = bipart.iloc[index_bi_t]    
    return data_bipart, drop_node

# subgraph using the weight of edges
def  subGraph (Graph,theshold) :  # subgraph with edges > theshold
    G = nx.Graph()
    for u,v in Graph.edges_iter() :
        w = 1
        if G.has_edge(u,v):
            G[u][v] ['weight'] += w
            #if (G[u][v]['weight'] > 1) :
            #     G[u][v] ['weight'] -= w
        else:
            G.add_edge(u,v, weight=w)
    sub_k_proj= nx.Graph()
    for u,v in G.edges_iter() :
        w = G[u][v]['weight']
	if (w > theshold):
            sub_k_proj.add_edge(u,v,weight = w)
    #print sub_k_proj.nodes()
    return sub_k_proj

# truth = pd.read_table('ground_truth/test03000.txt', sep = ',')

def get_diff_pairs(Graph, threshold) :
    thres_new = threshold - 1
    G = nx.Graph()
    for u,v in Graph.edges_iter() :
        w = 1
        if G.has_edge(u,v):
            G[u][v] ['weight'] += w
            #if (G[u][v]['weight'] > 1) :
            #     G[u][v] ['weight'] -= w
        else:
            G.add_edge(u,v, weight=w)
    sub_k_proj= nx.Graph()
    for u,v in G.edges_iter() :
        w = G[u][v]['weight']
        if (w <= thres_new):
            sub_k_proj.add_edge(u,v,weight = w)
    return sub_k_proj



#sub_bipart, droped_nodes = sub_edgelist (bipart, 1)
for single_file in files[:50] : 
#bipart = pd.read_table('sample_file/pop_name/name0000000000000095_00009384.txt', sep = ' ', header = None)
    bipart = pd.read_table(single_file, sep = ' ', header = None)
    G = nx.from_pandas_dataframe(bipart, 0, 1)
    G_proj = bipartite.projected_graph(G,nodes = bipart.ix[:, 0], multigraph = True)
#G_proj = subGraph(G_proj,2)
    A = nx.adjacency_matrix(G_proj).todense()
#K_A = katz(A,False)
    thres = np.percentile(A, 99.99)
    G_proj_new = subGraph(G_proj,thres)
    #mat_nodes_list = G_proj.nodes()
    k_file = 'sample_file/kTruth/pairs' + single_file[38:] 
    g_file = 'sample_file/graphs/graph' + single_file[38:-4] + 'edgelist.gz'
    same_pairs = pd.DataFrame(G_proj_new.edges())
    G_1 = subGraph(G_proj,1)
    same_pairs.to_csv(k_file, index = False)
    nx.write_weighted_edgelist(G_1, g_file)



#print mat_nodes_list
####
### test the performance unsupervised way. : 
'''
l = len(truth)
# x_id are id1 and id2
X_id = []
# ground truth
y = []
labels = []
# a subset of X_id [] 
X_IDs= []
# a subset of y, 1 corresponds to X_train, 2 the complitment of 1. 1+2 = y[]
y_ID_1 = []
y_ID_2 = []
distance = []
label_semi = []
X_train = []
# for semi 
sample_size = int(math.floor(l/4))
random.seed(7)
sample_index = np.asarray(random.sample(range(l), sample_size))
# end of semi sample

truth_id = list(truth['ID'])
truth_truth = list(truth['truth'])

for i in range(l-1) :
    for j in range(i+1, l) :
        X_id.append([truth_id[i], truth_id[j]])
        if (truth_truth[i] == truth_truth[j]):
            y.append(1)
        else :
            y.append(0)
        # get IDs
        ID_1 = X_id[-1][0]
        ID_2 = X_id[-1][1]
        index_1 = np.where(mat_nodes_list == ID_1 )[0][0]
        index_2 = np.where(mat_nodes_list == ID_2 )[0][0]
        #print index_1
        #print index_2
        f = K_A.item((index_1, index_2))
        distance.append(f) 
        if np.in1d(i, sample_index) and np.in1d(j, sample_index):
            label_semi.append(y[-1])
            X_train.append(f)
            y_ID_1.append(y[-1])       
        else :
            X_train.append(f)
            X_IDs.append([ID_1,ID_2])
            label_semi.append(-1)
            y_ID_1.append(y[-1])

X_id = []
K_A = nx.adjacency_matrix(G_proj).todense()
l = len(K_A)
#thres = np.percentile(K_A, 99.99)
#G_proj = subGraph(G_proj,thres)

for i in range(l-1) :
    for j in range(i+1, l) :
        if K_A[i,j] > 0 :
            X_id.append([mat_nodes_list[i], mat_nodes_list[j]])
       

same_pairs = pd.DataFrame(X_id)



#########unsupervised the resultX_id = []
l = len(K_A)
thres = np.percentile(K_A, 99.99)
G_proj = subGraph(G_proj,thres)

for i in range(l-1) :
    for j in range(i+1, l) :
        if K_A[i,j] > thres :
            X_id.append([mat_nodes_list[i], mat_nodes_list[j]])


print np.percentile(distance, [90,95])


labels = []
thres = np.percentile(K_A, 99.99)

fpe=0
fne=0

for i in range(len(distance)):
    if distance[i] > thres:
        labels.append(1)
    else :
        labels.append(0)
    if labels[-1] > y[i]:
        fpe = fpe+1
    if labels[-1] < y[i]:
        fne = fne +1

print "unsupervised"
print fpe
print fne
print sum(y)/float (len(y))
print sum(y)
print (fpe+fne)/float(len(distance))
print "precision recall"
avg_percision =  average_precision_score(y,labels)
print('Average precision-recall score: {0:0.2f}'.format(avg_percision))
recall_un = recall_score(y,labels)
print recall_un
print('recall'.format(recall_un))

print classification_report(y, labels)


############## semi-supervised result
max_elem = max(X_train)
print max_elem
for i in range(len(X_train)):
    X_train[i] = max_elem-X_train[i] + 0.1


X_train= np.asarray(X_train).reshape(-1, 1)
#print len(X_train) 
#print label_semi

#print np.shape(X_train)

# find the cut
cut = float(len(np.where(np.asarray(label_semi)==1 )[0]))/(len(np.where(np.asarray(label_semi)==0 )[0]) + len(np.where(np.asarray(label_semi)==1 )[0]))
print cut
########################build semi-supervied model #########################
label_spread = label_propagation.LabelSpreading(kernel = 'rbf', n_neighbors= 10, alpha = 0.05)
label_spread.fit(X_train, label_semi)

y_fitted = label_spread.predict_proba(X_train)
y_fitted_s = []
score = label_spread.score(X_train, y)
print score

fpe =0
fne =0

for i in range(len(y_fitted)):
    if y_fitted[i][1] > cut* 0.9:
        y_fitted_s.append(1)
    else :
        y_fitted_s.append(0)


for i in range(len(y_fitted_s)):
    if y_fitted_s[i] -y[i] >0:
        fpe +=1 # 0 predict to 1
    if y_fitted_s[i] -y[i] <0:
        fne +=1 # 1 predict to 0


print fpe
print fne


avg_percision =  average_precision_score(y,y_fitted_s)
print('Average precision-recall score: {0:0.2f}'.format(avg_percision))
recall_un = recall_score(y,y_fitted_s)
print recall_un

print classification_report(y, y_fitted_s)
'''
