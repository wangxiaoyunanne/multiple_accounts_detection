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
from sklearn.metrics import precision_recall_fscore_support

#bipart = pd.read_table('sample_file/name_id_49070.txt', sep = '\t', header = None)
bipart = pd.read_table('sample_file/name0000000000000462_00003986.txt', sep = '\t', header = None)
G = nx.from_pandas_dataframe(bipart, 0, 1)
A = nx.adjacency_matrix(G).todense()
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

def  subGraph (Graph,theshold) :  # subgraph with edges > theshold
    sub_k_proj= nx.Graph()
    for u,v in Graph.edges_iter() :
        w = Graph[u][v]['weight']
        if (w > theshold):
            sub_k_proj.add_edge(u,v,weight = w)
    #print sub_k_proj.nodes()
    return sub_k_proj



# row is normalized
#bipart = pd.read_table('sample_file/name_id_49070.txt', sep = '\t', header = None)
#truth = pd.read_table('ground_truth/truth_49070.txt', sep = ',')
truth = pd.read_table('ground_truth/truth_3000.txt', sep = ',')

G = nx.from_pandas_dataframe(bipart, 0, 1)
A = nx.adjacency_matrix(G).todense()

K_A = katz(A,False)

user_id = set(bipart.ix[:,0])
G_proj = bipartite.weighted_projected_graph(G,user_id)

G_new = subGraph (G_proj,3) 
nodes_list = G_new.nodes()
truth_id = truth.iloc[:,0]
truth_truth = truth.iloc[:,1]
node_list = list( set(truth_id).intersection(nodes_list))

mat_nodes_list = G.nodes()
#print mat_nodes_list
####
### test the performance unsupervised way. : 
# x_id are id1 and id2
X_id = []
# ground truth
y = []
labels = []
# a subset of X_id [] 
X_IDs= []
# a subset of y, 1 corresponds to X_train, 2 the complitment of 1. 1+2 = y[]
distance = []
X_train = []

for i in range(len(node_list)-1) :
    for j in range(i+1, len(node_list)):
        ID_1 = np.where(truth_id == node_list[i])[0][0]
        ID_2=  np.where(truth_id == node_list[j])[0][0]
        X_id.append([truth_id[ID_1], truth_id[ID_2]])
        if (truth_truth[ID_1] == truth_truth[ID_2]):
            y.append(1)
        else :
            y.append(0)
        # get IDs
        ID_3 = X_id[-1][0]
        ID_4 = X_id[-1][1]
        index_1 = np.where(mat_nodes_list == ID_3 )[0][0]
        index_2 = np.where(mat_nodes_list == ID_4 )[0][0]
        f = K_A.item((index_1, index_2))
        distance.append(f)
        

#########unsupervised the result
print "y"+str(float(sum(y))/len(y))
print np.percentile(distance, [90,95])


labels = []
thres = np.percentile(distance, 99.5)

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
# get precision and recall
print precision_recall_fscore_support(y,labels,average ='binary')
print roc_auc_score(y,labels)
print('Average precision-recall score: {0:0.2f}'.format(avg_percision))
recall_un = recall_score(y,labels)
print recall_un
print('recall'.format(recall_un))

print classification_report(y, labels, digits =5)

# get alternative ground truth 
labels = np.array(labels)
t0 = np.where(labels == 0)[0]
X_0 = []
for index0 in t0:
    X_0.append(X_id[index0])

X_0_data = pd.DataFrame (X_0)
X_0_data.to_csv('data2X0.txt', index = False)

labels = np.array(labels)
t1 = np.where(labels == 1 )[0]
X_1 = []
for index1 in t1:
    X_1.append(X_id[index1])
    
X_1_data = pd.DataFrame(X_1)
X_1_data.to_csv('data2X1.txt', index = False)





############## semi-supervised result
X_train = distance
max_elem = max(X_train)
print max_elem
for i in range(len(X_train)):
    X_train[i] = max_elem-X_train[i] + 0.1


X_train= np.asarray(X_train).reshape(-1, 1)
#print len(X_train) 
#print label_semi
# sample data
random.seed(7)
l = len(X_train)
sample_size = l /16
sample_index = np.asarray(random.sample(range(l), sample_size))
label_semi = [-1]* l
for i in sample_index:
    label_semi[i] = y[i]
    

# find the cut
cut = float(len(np.where(np.asarray(label_semi)==1 )[0]))/(len(np.where(np.asarray(label_semi)==0 )[0]) + len(np.where(np.asarray(label_semi)==1 )[0]))
print cut
########################build semi-supervied model #########################
label_spread = label_propagation.LabelSpreading(kernel = 'rbf', n_neighbors= 10, alpha = 0.2)
label_spread.fit(X_train, label_semi)

y_fitted = label_spread.predict_proba(X_train)
y_fitted_s = []
score = label_spread.score(X_train, y)
print score

y_fitted_v = y_fitted[:,1]
cut = np.percentile(y_fitted_v,(1- cut)*100)

fpe =0
fne =0

for i in range(len(y_fitted)):
    if y_fitted[i][1] > cut:
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
print roc_auc_score(y,y_fitted_s)

print classification_report(y, y_fitted_s)

