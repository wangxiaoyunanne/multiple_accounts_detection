from numpy import linalg as LA
import pandas as pd
import numpy as np
import random
import math
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import roc_auc_score
from scipy import spatial
import networkx as nx
from networkx.algorithms import bipartite
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.cluster import SpectralClustering
#########################################################################
######################use semi supervised learning######################
##input pairs of nodes output 0 or 1, whether 2 nodes belongs to the same person

#df =pd.read_table('node2vec/emb/test_edgelist.emb', sep = ' ', header = None)

#df_bipart = pd.read_table('node2vec/emb/49070.emb', sep = ' ', header = None)
df_bipart = pd.read_table('sample_file/name_id_49070_0.58.emb', sep = ' ', header = None)
bipart = pd.read_table('sample_file/name_id_49070.txt', sep = '\t', header = None)
# get ground truth and sample some data
truth = pd.read_table('ground_truth/truth_49070_2.txt', sep = ',')
# generate a set of y  and X train data and label data. -1 means not labeled.
#bi_g = pd.read_table('sample_file/name_id_49070.txt', header = None)

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

G_new = subGraph (G_proj,1)

nodes_list = G_new.nodes()
truth_id = truth.iloc[:,0]
truth_truth = truth.iloc[:,1]
node_list = list( set(truth_id).intersection(nodes_list))

l = len(node_list)
sample_size = int(math.floor(l/4))
random.seed(7)
sample_index = np.asarray(random.sample(range(l), sample_size))
X_id = []
y = []
labels = []
# a subset of X_id [] 
X_IDs= []
# a subset of y, 1 corresponds to X_train, 2 the complitment of 1. 1+2 = y[]

# generate X_train using X_id
#ID_list = df.ix[:,0]
ID_bipart = df_bipart.ix[:,0]
X_train = []
# get abs range testing 
abs_r = []
###################################
for i in range(len(node_list)-1) :
    for j in range(i+1, len(node_list)) :
        ID_1 = np.where(truth_id == node_list[i])[0][0]
        ID_2=  np.where(truth_id == node_list[j])[0][0]
        X_id.append([truth['ID'][ID_1], truth['ID'][ID_2]])
        if (truth['truth'][ID_1] == truth['truth'][ID_2]):
            y.append(1)
        else :
            y.append(0)
#####calculate L1 of vects
        ID_1 = X_id[-1][0]
        ID_2 = X_id[-1][1]
        index_1 = np.where(ID_bipart == ID_1 )[0][0]
        index_2 = np.where(ID_bipart == ID_2 )[0][0]
        f1 = df_bipart.iloc[index_1][1:]
        f2 = df_bipart.iloc[index_2][1:]
        abs_12 = abs(f1-f2)
        if np.in1d(i, sample_index) and np.in1d(j, sample_index):
            labels.append(y[-1])
            X_train.append(abs_12)
        else :
            X_train.append(abs_12)
            X_IDs.append([ID_1,ID_2])
            labels.append(-1)
              
# generate X_train using X_id
#ID_list = df.ix[:,0]
#ID_bipart = df_bipart.ix[:,0]
#X_train = []
X_train= np.asarray(X_train)
# find the cut
cut = float(len(np.where(np.asarray(labels)==1 )[0]))/(len(np.where(np.asarray(labels)==0 )[0]) + len(np.where(np.asarray(labels)==1 )[0]))

########################build semi-supervied model #########################
label_spread = label_propagation.LabelSpreading(kernel = 'rbf', n_neighbors=7, alpha = 0.2)
label_spread.fit(X_train, labels)

y_fitted = label_spread.predict_proba(X_train)
y_fitted_s = []
#score = label_spread.score(X_train, y_ID_1)
#print score
fpe =0
fne =0

y_fitted_v = y_fitted[:,1]
cut = np.percentile(y_fitted_v,(1- cut)*100)

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


print "fpe and fne"
print fpe 
print fne
#y_fitted
index_test= np.where(np.asarray(labels) == -1)[0]
y_ind_test = [y[i] for i in index_test]
y_ind_fitted = [y_fitted_s[i] for i in index_test]
roc = roc_auc_score(y,y_fitted_s)
precision_recall_fscore_support(y,y_fitted_s, average = 'binary')
sum(y_fitted_s)
print roc
print classification_report(y,y_fitted_s, digits = 5)
############# try to group ID. test whether using groups will influence perf
df_bipart_new = []
for i in range(len(node_list)):
    id_1 = np.where(df_bipart.iloc[:,0] == node_list[i] )[0][0]
    df_bipart_new.append(df_bipart.iloc[id_1,:])

df_bipart= pd.DataFrame(df_bipart_new)

num_group = 2
ID_bipart = df_bipart.ix[:,0]
data_bipart = df_bipart.ix[:,1:]
random.seed(0)
kmeans_bipart = KMeans(n_clusters=num_group, random_state=0).fit(data_bipart)
labels_bipart = kmeans_bipart.labels_
labels_bipart = SpectralClustering(num_group, gamma = 0.2,affinity='rbf').fit(data_bipart).labels_

df_bipart ['group'] = labels_bipart
global_y = 0
global_len = 0
global_truth = []
global_fitted = []

for i_group in range(num_group) :
    df_bipart_i = df_bipart.query('group == @i_group')
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
    sample_size = int(math.floor(l/2))
    random.seed(7)
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
            #print len(y_fitted)
            #print (y)
            y_v = y_fitted[:,1]
            cut = np.percentile(y_v,(1-cut)*100)
            #cut = 0.6
            #cut = 1-cut
            for i_fit in range(len(y_fitted)):
                if y_fitted[i_fit][1] > cut:
                    y_fitted_s.append(1)
                else :
                    y_fitted_s.append(0)
            print len(y_fitted_s)
            for i_fit in range(len(y_fitted_s)):
                if y_fitted_s[i_fit] -y[i_fit] >0:
                    fpe +=1 # 0 predict to 1
                    #print i_fit
                elif y_fitted_s[i_fit] -y[i_fit] <0:
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
print classification_report(global_truth,global_fitted, digits = 5)
######################################################
###########try matrix 
'''
X_mat = np.zeros((sample_size, 128))
A_mat = np.zeros((l,l))
cut
for i in range(l):
    for j in range(l):
        if (truth['truth'][i] == truth['truth'][j]):
            A_mat[i,j] = 1
        else :
            A_mat[i,j] = 0

A = A_mat[sample_index, :][:, sample_index ]
ID_bipart = df_bipart.ix[:,0]

for i in range(sample_size) :
    ID_1 = truth['ID'][sample_index[i]]
    index_1 = np.where(ID_bipart == ID_1 )[0][0]
    X_mat[i,:] = df_bipart.iloc[index_1][1:]
   # for j in range(sample_size) :
   #     A[i,j] =  A_mat[sample_index[i], sample_index[j]]


   
XTX_1 =np.linalg.pinv( np.dot(X_mat.T,X_mat))
XTAX = np.dot(np.dot(X_mat.T,A),X_mat)
M = np.dot(np.dot(XTX_1,XTAX),XTX_1) 

X_new = np.zeros((l-sample_size, 128))
left_index =  list(set(range(l))- set (sample_index))

for i in range(l-sample_size) :
    ID_2 = truth['ID'][left_index[i]]
    index_2 = np.where(ID_bipart == ID_2)[0][0]
    X_new[i,:] = df_bipart.iloc[index_2][1:]

A_2 = A_mat[sample_index,:][:, left_index]
A_3 = A_mat[left_index,:][:, sample_index]
A_4 = A_mat[left_index,:][:, left_index]

def fit_mat (X1,M,X2):
    return np.dot(np.dot(X1,M),X2.T) 

A_f1 = fit_mat(X_mat,M, X_mat)
A_f2 = fit_mat(X_mat,M, X_new) #A_2
A_f3 = fit_mat(X_new,M,X_mat) #A_3
A_f4 = fit_mat(X_new,M,X_new) #A_4

def res_mat(Af,A_true,cut,diag = False) :
    thres = np.percentile(Af.reshape(-1),( 1-cut)*88)
    if diag == True: 
        np.fill_diagonal(Af, thres + 1 )
    Af_p = np.where (Af.reshape(-1) > thres)[0]
    result_true = A_true.reshape(-1)
    A_p = np.where(result_true == 1)[0]
    #print Af_p.diagonal()
    #print len(A_p)
    print thres
    result = np.zeros(len(result_true))
    result[Af_p] = 1
    fpe = len(set(Af_p)- set(A_p))
    fne = len(set(A_p)- set(Af_p))
    roc = roc_auc_score(result_true,result)
    return [fpe, fne,sum(result_true),roc]



res_mat(A_f1,A, cut, True)
res_mat(A_f2, A_2, cut)
res_mat(A_f3, A_3, cut)
res_mat(A_f4, A_4, cut, True)
'''

if False:'''
label_prop_model = LabelSpreading(kernel = 'knn', n_neighbors = 4, alpha = 0.155)
label_prop_model.fit(X_train, labels)
y_fitted = label_prop_model.predict_proba(X_train)
'''




if False :'''
## norm of each pairs of vectors
for i in range (len(df)):
    for j in range(len(df)) :
        a = abs(df.iloc[i][1:128] - df.iloc[j][1:128])
        mat[i,j] = LA.norm(a)

# print pairwised data
for i in  range(70):
    for j in range(i+1,70):
        if abs (mat[i,j]) < 0.3:
            print (df.ix[i,0], df.ix[j,0])


sample = [[1066017246744202,'mostafa.mosad3'],[975611912520299,'mostafa.mosad.10'],[ 1036909843057172,'mostafa.mosad.10' ],[190866118021578,'mostafa.mosad.1804'],[ 1153797658036234,'mostafa.mosad.9889'],[ 1170003323079675,'mostafa.mosad.7' ]]

sample_vec = []
#print sample[5][0]
for i in range (len(sample)):
    sample_vec = sample_vec + ( df[df.ix[:,0]==sample[i][0]].index.tolist())



print sample_vec
for i in  range(70):
    min_dist = []
    for j in range(len(sample)):
        if (mat[i,sample_vec[j]]< 0.3):
            min_dist.append(sample[j][1])
            min_dist.append(mat[i,sample_vec[j]])
    print (df.ix[i,0],min_dist )        
'''









        
