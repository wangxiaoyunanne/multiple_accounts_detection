#### kmeans_shuffle.py
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
from sklearn.cluster import SpectralClustering

#########################################################################
######################use semi supervised learning######################
##input pairs of nodes output 0 or 1, whether 2 nodes belongs to the same person

#df =pd.read_table('node2vec/emb/test_edgelist.emb', sep = ' ', header = None)
#mat = np.zeros(shape=(70,70) )
#df_bipart = pd.read_table('node2vec/emb/49070.emb', sep = ' ', header = None)

df_bipart = pd.read_table('sample_file/shuffled/data2/diff_size/1mi_4_s15.emb', sep = ' ', header = None) 
#df_bipart = df_bipart.sort_values(by = 0 )
#df_bipart = df_bipart.iloc[:500,:]
# kTruth column '0' and '1'
#kTruth = pd.read_table('sample_file/kTruth/pairs100_00009242.txt', sep = ',')
            
# get ground truth and sample some data
#truth = pd.read_table('ground_truth/truth_49070.txt', sep = ',')
#truth = pd.read_table('sample_file/truth_file/truth0000000000000100_00009242.txt', sep = ',')

#subGraph = nx.read_edgelist('sample_file/shuffled/data2_1000_0_5s.edgelist', create_using = nx.MultiGraph())


# generate a set of y  and X train data and label data. -1 means not labeled.
#bi_g = pd.read_table('sample_file/name_id_49070.txt', header = None)

#l = len(truth)
#sample_size = int(math.floor(l/4))
def num_cluster (data_bipart):
    random.seed(17)
    num = len(data_bipart) /50 +1 
    print num
    #kmeans_bipart = KMeans(n_clusters=num, random_state=0).fit(data_bipart)
    #labels_bipart = kmeans_bipart.labels_
    labels_bipart = SpectralClustering( num, gamma = 0.7,affinity='rbf').fit(data_bipart).labels_
    max_group = max(itemfreq(labels_bipart)[:,1]) 
    while (max_group >200):
        num += 2
        print num
       # kmeans_bipart = KMeans(n_clusters=num, random_state=0).fit(data_bipart)
       # labels_bipart = kmeans_bipart.labels_
        labels_bipart = SpectralClustering( num, gamma = 0.7,affinity='rbf').fit(data_bipart).labels_
        max_group = max(itemfreq(labels_bipart)[:,1])
    return num



random.seed(7)
#sample_index = np.asarray(random.sample(range(l), sample_size))
X_id = []
y = []
labels = []
# a subset of X_id [] 
X_IDs= []
# a subset of y, 1 corresponds to X_train, 2 the complitment of 1. 1+2 = y[]
y_ID_1 = []
y_ID_2 = []

# generate X_train using X_id
#ID_list = df.ix[:,0]
ID_bipart = df_bipart.ix[:,0].astype(str)
data_bipart = df_bipart.ix[:,1:]
print 'number of nodes' + str(len(ID_bipart))

# need to determin how do you get num_group
num_group = num_cluster(data_bipart)
#num_group = 5
print 'number of groups' + str(num_group)
#kmeans_bipart = KMeans(n_clusters=num_group, random_state=0).fit(data_bipart)
#labels_bipart = kmeans_bipart.labels
random.seed(17)
labels_bipart = SpectralClustering( num_group, gamma = 0.7,affinity='rbf').fit(data_bipart).labels_
# get ktruth's group
#k_groups = kTruth_groups (ID_bipart, labels_bipart, kTruth)

# add new column group to data
df_bipart ['group'] = labels_bipart
#############
#group data by their 'group'
#df_bipart = df_bipart.sort_values('group')

#divide by group i 
# @i means i is a variable in group
global_y = 0
global_len = 0
global_truth = []
global_fitted = []

y_whole = []
y_fitted_whole = []
for i_group in range(num_group) :
    df_bipart_i = df_bipart.query('group == @i_group ') 
    X_train = []
    labels = []
    y = []
    l = len(df_bipart_i)
    print "l is " +str(l)        
    df_bipart_index = df_bipart_i.iloc[:,0]
    df_bipart_index = list(df_bipart_index)
    df_bipart_index = np.array(df_bipart_index)
    #print df_bipart_index
###################################
    # sample data
    sample_size = int(math.floor(l/4))
    random.seed(10)
    for i in range(l-1) :
        for j in range(i+1, l):
            t1 = df_bipart_index[i]
            t2 = df_bipart_index[j]
            #print t1, t2
            if t1[:-2] == t2[:-2] :
                X_train.append([t1,t2])
                y.append(1)
            else:
                X_train.append([t1, t2])
                y.append(0) 
    # change X_train to dataframe
    X_train = pd.DataFrame(X_train)
    # get embing data
    X_train_id = X_train
    X_train = []
    for id_index in range(len(X_train_id)) :
        id_1 =  X_train_id.iloc[id_index,0 ]
        id_2 = X_train_id.iloc[id_index,1 ]
        ID_1 = np.where ( df_bipart_index == id_1)[0][0]
        ID_2 = np.where ( df_bipart_index == id_2)[0][0]            
        abs_value = abs(df_bipart_i.iloc[ID_1, 1:] - df_bipart_i.iloc[ID_2, 1:]  )
        X_train.append(abs_value)
    X_train =  pd.DataFrame(X_train)
    print X_train.shape
# fit models
#sample data 
    l = len(X_train)
    sample_size = int(math.floor(l/4))
    random.seed(10)
    sample_index = np.asarray(random.sample(range(l), sample_size))
        # just need to sample lables
        # first try only labeled data
    labels = [-1]*len(y)
    labels = np.array(labels)
    y = np.array(y)
    if l <5  :
        continue
    else:      
        labels[sample_index] = y[sample_index] 
# find the cut
# small sample cut is no longer useful just let it be 15%
        cut = float(len(np.where(np.asarray(labels)==1 )[0]))/(len(np.where(np.asarray(labels)==0 )[0]) + len(np.where(np.asarray(labels)==1 )[0]))
        print "cut"+ str(cut)
        if cut == 0.0 :
            labels = np.zeros(len(labels))
            print classification_report(y,labels )
            y_whole.extend (y)
            y_fitted_whole.extend(labels)
        elif cut == 1.0 :
            labels = np.ones(len(labels))
            print classification_report(y,labels )
            y_whole.extend (y)
            y_fitted_whole.extend(labels)
        else :
            #cut = 0.15
########################build semi-supervied model #########################
            label_spread = label_propagation.LabelSpreading(kernel = 'rbf', n_neighbors=7, alpha = 0.2)
            label_spread.fit(X_train, labels)     
            y_fitted = label_spread.predict_proba(X_train)
            y_fitted_s = []
            score = label_spread.score(X_train, y)
            print "score"+ str(score)
            print len(y_fitted)
        #print y_fitted
            fpe =0
            fne =0
           # print y_fitted
            y_v = y_fitted[:,1]
            cut = np.percentile(y_v,100*( 1-cut))
            for i_fit in range(len(y_fitted)):
                if y_fitted[i_fit][1] > cut*0.995:
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
            print f1_score(y,y_fitted_s, average = 'weighted' )
            print precision_score(y,y_fitted_s, average = 'weighted')
            print recall_score(y,y_fitted_s, average = 'weighted')
            y_whole.extend (y)
            y_fitted_whole.extend(y_fitted_s)
    # add inter group result
        if i_group + 1 < num_group :
            for j_group in range(i_group+1, num_group) :
                df_bipart_j = df_bipart.query('group == @j_group')
                l_j  = len(df_bipart_j)
                df_bipart_index_j = df_bipart_j.iloc[:,0]
                df_bipart_index_j = list(df_bipart_index_j)
                df_bipart_index_j = np.array(df_bipart_index_j)
                y = []
           # print l_j
                for i in range(len(df_bipart_index)):
            #    print i
                    for j in range(l_j):
                        if (df_bipart_index[i][:-2] == df_bipart_index_j[j][:-2]):
                            y.append(1)
                        else :
                            y.append(0)
                local_rate = float(sum(y)) /len(y)
            #print "local rate" + str(local_rate)
                y_whole.extend(y)
                y_fitted_whole.extend([0]* len(y))
        
print classification_report(y_whole , y_fitted_whole )
print f1_score(y_whole , y_fitted_whole, average = 'weighted' )
print precision_score(y_whole ,y_fitted_whole, average = 'weighted')  
print recall_score(y_whole , y_fitted_whole, average = 'weighted')
 
print roc_auc_score(y_whole, y_fitted_whole )         
print f1_score(y_whole , y_fitted_whole )
print precision_score(y_whole ,y_fitted_whole)
print recall_score(y_whole , y_fitted_whole)

if False:'''
# get inter-group ground truth
    if i_group + 1 < num_group :
        for j_group in range(i_group+1, num_group) : 
            df_bipart_j = df_bipart.query('group == @j_group') 
            l_j  = len(df_bipart_j)
            y = []
            for i in range(l):
                for j in range(l_j):
                    if (truth_truth[i] == truth_truth[j]):
                        y.append(1)
                    else :
                        y.append(0)
            local_rate = float(sum(y)) /len(y)
            #print "local rate" + str(local_rate)
            global_y += sum(y)
            global_len += len(y)
            #print global_len
global_rate = float(global_y)/global_len   
print "global y" + str(global_y)

print "global rate"+str( global_rate)
roc = roc_auc_score(global_truth, global_fitted )
print roc
#print classification_report(global_truth,global_fitted)



#y_fitted
index_test= np.where(np.asarray(labels) == -1)[0]
y_ind_test = [y[i] for i in index_test]
y_ind_fitted = [y_fitted_s[i] for i in index_test]
roc = roc_auc_score(y_ind_test,y_ind_fitted)
sum(y_fitted_s)
print roc
print score

output_file = 'output_03000.txt'
with open (output_file, "w") as text_file:
    text_file.write("{}".format((cut,score,fpe,fne )))

text_file.close()

######################################################
###########try matrix 
X_mat = np.zeros((sample_size, 128))
A_mat = np.zeros((l,l))

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



label_prop_model = LabelSpreading(kernel = 'knn', n_neighbors = 4, alpha = 0.155)
label_prop_model.fit(X_train, labels)
y_fitted = label_prop_model.predict_proba(X_train)


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









        
