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
from sklearn.metrics import classification_report
#########################################################################
######################use semi supervised learning######################
##input pairs of nodes output 0 or 1, whether 2 nodes belongs to the same person

#df =pd.read_table('node2vec/emb/test_edgelist.emb', sep = ' ', header = None)
#mat = np.zeros(shape=(70,70) )
df_bipart = pd.read_table('node2vec/emb/49070.emb', sep = ' ', header = None)
# get ground truth and sample some data
truth = pd.read_table('ground_truth/truth_49070.txt', sep = ',')
# generate a set of y  and X train data and label data. -1 means not labeled.
#bi_g = pd.read_table('sample_file/name_id_49070.txt', header = None)
bipart = pd.read_table('sample_file/name_id_49070.txt', sep = '\t', header = None)
G = nx.from_pandas_dataframe(bipart, 0, 1)
node_list = []
drop_node = []
user_id = list(bipart.ix[:,0])


for n in G.nodes() :
    print G.degree(n)
    print (n in user_id)
    if n in user_id:
        if G.degree(n) >1 :
            node_list.append(n)
        else :
            drop_node.append(n)



index_in_t = [] 

for n in node_list:
    index_in = np.where(truth['ID'] == n)[0][0]
    index_in_t.append(index_in)

truth =  truth.iloc[index_in_t,:]

l = len(truth)
sample_size = int(math.floor(l/4))
random.seed(7)
sample_index = np.asarray(random.sample(range(l), sample_size))
X_id = []
y = []
labels = []
for i in range(l-1) :
    for j in range(i+1, l) :
        X_id.append([truth['ID'][i], truth['ID'][j]])
        if (truth['truth'][i] == truth['truth'][j]):
            y.append(1)
        else :
            y.append(0)
        if np.in1d(i, sample_index) and np.in1d(j, sample_index):
            labels.append(y[-1])
        else :
            labels.append(-1)

#print len(X_id)
#print df_bipart
# generate X_train using X_id
#ID_list = df.ix[:,0]
ID_bipart = df_bipart.ix[:,0]
X_train = []
X_IDs = []
for i in range(len(X_id)):
    ID_1 = X_id[i][0]
    ID_2 = X_id[i][1]
    index_1 = np.where(ID_bipart == ID_1 )[0][0]
    index_2 = np.where(ID_bipart == ID_2 )[0][0]
    f1 = df_bipart.iloc[index_1][1:]
    f2 = df_bipart.iloc[index_2][1:]
    abs_12 = abs(f1-f2)
    #abs_12 = (f1+f2)/2
    #abs_12 = abs_12*abs_12
    #if sum(abs_12) < 0.5:
    X_train.append(abs_12)
    X_IDs.append([ID_1,ID_2])
    #X_train.append([spatial.distance.cosine(f1, f2),1])

X_train= np.asarray(X_train)
#print len(X_train)

# find the cut
cut = float(len(np.where(np.asarray(labels)==1 )[0]))/(len(np.where(np.asarray(labels)==0 )[0]) + len(np.where(np.asarray(labels)==1 )[0]))

########################build semi-supervied model #########################
label_spread = label_propagation.LabelSpreading(kernel = 'rbf', n_neighbors=10, alpha = 0.01)
label_spread.fit(X_train, labels)

y_fitted = label_spread.predict_proba(X_train)
y_fitted_s = []
score = label_spread.score(X_train, y)
print score
fpe =0
fne =0

for i in range(len(y_fitted)):
    if y_fitted[i][1] >= cut:
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
#y_fitted
index_test= np.where(np.asarray(labels) == -1)[0]
y_ind_test = [y[i] for i in index_test]
y_ind_fitted = [y_fitted_s[i] for i in index_test]
roc = roc_auc_score(y_ind_test,y_ind_fitted)
sum(y_fitted_s)

print classification_report(y, y_fitted_s)
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









        
