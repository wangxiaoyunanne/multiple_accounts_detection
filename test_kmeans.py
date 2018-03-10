#### test_kmeans.py
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
import networkx as nx
from scipy.stats import itemfreq
from networkx.algorithms import bipartite
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
#########################################################################
######################use semi supervised learning######################
##input pairs of nodes output 0 or 1, whether 2 nodes belongs to the same person

#df =pd.read_table('node2vec/emb/test_edgelist.emb', sep = ' ', header = None)
#mat = np.zeros(shape=(70,70) )
#df_bipart = pd.read_table('node2vec/emb/49070.emb', sep = ' ', header = None)
df_bipart = pd.read_table('sample_file/name0000000000000462_00003986_0.54.emb', sep = ' ', header = None)
bipart = pd.read_table('sample_file/name0000000000000462_00003986.txt', sep = '\t', header = None)
# get ground truth and sample some data
#truth = pd.read_table('ground_truth/truth_49070.txt', sep = ',')
truth = pd.read_table('ground_truth/truth_3000.txt', sep = ',')
#sample_index = np.asarray(random.sample(range(l), sample_size))

truth_id = truth.iloc[:,0]
truth_truth = truth.iloc[:,1]


def  subGraph (Graph,theshold) :  # subgraph with edges > theshold
    sub_k_proj= nx.Graph()
    for u,v in Graph.edges_iter() :
        w = Graph[u][v]['weight']
        if (w > theshold):
            sub_k_proj.add_edge(u,v,weight = w)
    #print sub_k_proj.nodes()
    return sub_k_proj

G = nx.from_pandas_dataframe(bipart, 0, 1)
user_id = set(bipart.ix[:,0])
G_proj = bipartite.weighted_projected_graph(G,user_id)
G_new = subGraph (G_proj,3)
nodes_list = G_new.nodes()
node_list = list( set(truth_id).intersection(nodes_list))

df_bipart_new = []
for i in range(len(node_list)):
    id_1 = np.where(df_bipart.iloc[:,0] == node_list[i] )[0][0]
    df_bipart_new.append(df_bipart.iloc[id_1,:])

df_bipart= pd.DataFrame(df_bipart_new)

def num_cluster (data_bipart):
    random.seed(17)
    num = len(data_bipart) /300 +1
    kmeans_bipart = KMeans(n_clusters=num, random_state=0).fit(data_bipart)
    labels_bipart = kmeans_bipart.labels_
    max_group = max(itemfreq(labels_bipart)[:,1])
    while (max_group >350):
        num += 2
        kmeans_bipart = KMeans(n_clusters=num, random_state=0).fit(data_bipart)
        labels_bipart = kmeans_bipart.labels_
        max_group = max(itemfreq(labels_bipart)[:,1])
    return num


X_id = []
y = []
labels = []
# a subset of X_id [] 
X_IDs= []
# generate X_train using X_id
#ID_list = df.ix[:,0]
ID_bipart = df_bipart.ix[:,0]
data_bipart = df_bipart.ix[:,1:]
# need to determin how do you get num_group
num_group = 3
random.seed(17)
kmeans_bipart = KMeans(n_clusters=num_group, random_state=0).fit(data_bipart)
labels_bipart = kmeans_bipart.labels_
labels_bipart = SpectralClustering( num_group, gamma = 0.5,affinity='rbf').fit(data_bipart).labels_

# add new column group to data
df_bipart ['group'] = labels_bipart
#############
#group data by their 'group'
# df_bipart = df_bipart.sort_values('group')
#divide by group i 
# @i means i is a variable in group
global_y = 0
global_len = 0
global_truth = []
global_fitted = []
print "number of groups" + str(num_group)
for i_group in range(num_group) :
    df_bipart_i = df_bipart.query('group == @i_group ') 
## need to test whether the size of each group is too big, may need a while lo 
    l = len(df_bipart_i)
    print "l is " +str(l)        
    X_train = []
# get abs range testing 
    X_id =[]
    y = []
    labels = []
    # a subset of X_id [] 
    X_IDs= []
###################################
    df_bipart_index = df_bipart_i.iloc[:,0]
    #truth_set = set(truth_id).intersection(set(df_bipart_i.ix[:,0]))
    # sample data
    sample_size = int(math.floor(l/4))
    random.seed(10)
    sample_index = np.asarray(random.sample(range(l), sample_size))
    for i in range(l-1) :
        for j in range(i+1, l) :
            ID_1 = df_bipart_index.iloc[i]
            ID_2 = df_bipart_index.iloc[j]
            X_id.append([ID_1, ID_2])
            index_1 = np.where(truth_id == ID_1 )[0][0]
            index_2 = np.where(truth_id == ID_2 )[0][0]
            if (truth_truth[index_1] == truth_truth[index_2]):
                y.append(1)
            else :
                y.append(0)
#####calculate L1 of vects
########need to change
            index_1 = np.where(df_bipart_index == ID_1 )[0][0]
            index_2 = np.where(df_bipart_index == ID_2 )[0][0]
            f1 = df_bipart_i.iloc[index_1,1:]
            f2 = df_bipart_i.iloc[index_2,1:]
            abs_12 = abs(f1-f2)
            if np.in1d(i, sample_index) and np.in1d(j, sample_index):
                labels.append(y[-1])
                X_train.append(abs_12)
            else :
                X_train.append(abs_12)
                X_IDs.append([ID_1,ID_2])
                labels.append(-1)
    #if group size is smaller than 10, we directly assign 0s
    if l<= 1 or l >300  :
        #labels = np.zeros(len(labels))
        continue
    else :
# find the cut
# small sample cut is no longer useful just let it be 15%
        cut = float(len(np.where(np.asarray(labels)==1 )[0]))/(len(np.where(np.asarray(labels)==0 )[0]) + len(np.where(np.asarray(labels)==1 )[0]))
        print "cut"+ str(cut)
        if cut ==0.0 :
            labels = np.zeros(len(labels))
            print classification_report(y,labels )
            global_truth.extend(y)
            global_fitted.extend(labels)
        elif cut == 1.0 :
            labels = np.ones(len(labels))
            print classification_report(y,labels )
            global_truth.extend(y)
            global_fitted.extend(labels)
        else :
            #cut = 0.15
########################build semi-supervied model #########################
            label_spread = label_propagation.LabelSpreading(kernel = 'rbf', n_neighbors=7, alpha = 0.2)
            label_spread.fit(X_train, labels)
            y_fitted = label_spread.predict_proba(X_train)
            y_fitted_s = []
            #score = label_spread.score(X_train, y_ID_1)
            #print "score"+ str(score)
            #print len(y_fitted)
            fpe =0
            fne =0
           # print y_fitted
            y_v = y_fitted[:,1]
            cut = np.percentile(y_v,100*( 1-cut))
            for i_fit in range(len(y_fitted)):
                if y_fitted[i_fit][1] > cut:
                    y_fitted_s.append(1)
                else :
                    y_fitted_s.append(0)
            for i_fit in range(len(y_fitted_s)):
                if y_fitted_s[i_fit] -y[i_fit] >0:
                    fpe +=1 # 0 predict to 1
                if y_fitted_s[i_fit] -y[i_fit] <0:
                    fne +=1 # 1 predict to 0
            print fpe 
            print fne
            print classification_report(y,y_fitted_s )
            global_truth.extend(y)
            global_fitted.extend(y_fitted_s)
# get inter-group ground truth
    if i_group + 1 < num_group :
        for j_group in range(i_group+1, num_group) : 
            df_bipart_j = df_bipart.query('group == @j_group') 
            l_j  = len(df_bipart_j)
            if l_j <=6 or l_j >300:
                continue
            else:
                y = []
                for i in range(l):
                    for j in range(l_j):
                        if (truth_truth[i] == truth_truth[j]):
                            y.append(1)
                        else :
                            y.append(0)
                #local_rate = float(sum(y)) /len(y)
                #print "local rate" + str(local_rate)
                global_truth.extend(y)
                global_fitted.extend([0]*len(y))
                
roc = roc_auc_score(global_truth, global_fitted )
print roc
print classification_report(global_truth,global_fitted)







        
