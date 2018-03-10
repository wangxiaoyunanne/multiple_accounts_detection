import pandas as pd
import igraph
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from numpy.linalg import pinv
from numpy import linalg as LA
from scipy import stats
from sklearn.preprocessing  import normalize

if False:'''
file_1 = pd.read_table("test_file.txt", sep = '\t', header = None, names = ["user","page"])
g = nx.from_pandas_dataframe(file_1, 'user', 'page')
A = nx.adjacency_matrix(g)
data_user = file_1['user']
#####get away of mulit edges of origen bipartite graph
G = nx.Graph()
for u,v in g.edges_iter() :
    w = 1
    if G.has_edge(u,v):
        G[u][v] ['weight'] += w
        if (G[u][v]['weight'] > 1) :
             G[u][v] ['weight'] -= w
    else:
        G.add_edge(u,v, weight=w)
'''

'''
#####get number of n share pages node

G_sub = nx.Graph()
for u0,v0 in G.edges_iter() :
    user_neighbors = G.neighbors(u0)
    for v1 in user_neighours:
        u1 = G.neighbors(v1)
        
'''

bipart = pd.read_table('sample_file/name_id_49070.txt', sep = '\t', header = None)
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

# row is normalized
K_A = katz(A)
############################################################################
p =1 
q = 0.5
def alpha(G):
    #result = []
    result = np.zeros(A.shape)
    for t,v in G.edges_iter() :
        pv = 1/p # go ti v
        #result.append([v,t, pv])
        result[v-1,t-1] = 1/p
        result[t-1,v-1] = 1/p
        v_neighbors = G.neighbors(v)
        t_neighbors = G.neighbors(t)
        x_1 = [node for node in v_neighbors if node in t_neighbors]
        if len(x_1):
            for node in x_1:
                #result.append([v,node,1])
                result[v-1,node-1] = 1
                result[node-1,v-1] = 1
        x_2 = [node for node in v_neighbors if node not in t_neighbors]
        if len(x_2):
            for node in x_2:
                #result.append([v,node,1/q])
                result[v-1,node-1] = 1/q
                result[node-1,v-1] = 1/q
       # print (t,v)
         # then swith v and t
        result[v-1,t-1] = 1/p
        result[t-1,v-1] = 1/p
        if len(x_1):
            for node in x_1:
                #result.append([v,node,1])
                result[t-1,node-1] = 1
                result[node-1,t-1] = 1
        x_22 = [node for node in t_neighbors if node not in v_neighbors]
        if len(x_22):
            for node in x_22:
                #result.append([v,node,1/q])
                result[t-1,node-1] = 1/q
                result[node-1,t-1] = 1/q
    return result


karate = nx.from_pandas_dataframe(data,0,1)
A = nx.adjacency_matrix(karate).todense()
result = np.zeros(A.shape)
K = alpha(karate)







g_proj =   bipartite.projected_graph(G,data_user , multigraph=True)
G_proj = nx.Graph()
#######change multi edges to weight
for u,v in g_proj.edges_iter() :
    w = 1
    if G_proj.has_edge(u,v):
        G_proj[u][v] ['weight'] += w
        
    else:
        G_proj.add_edge(u,v, weight=w)

######## subset graph of G_proj with edges >k
k = 2
sub_k_proj= nx.Graph()
for u,v in G_proj.edges_iter() :
    w = G_proj[u][v]['weight'] 
    if (w > k):
        sub_k_proj.add_edge(u,v,weight = w)

######## connected components
id_group = sorted(nx.connected_components(sub_k_proj),key = len, reverse = True)
nb_comp = nx.number_connected_components(sub_k_proj)


##################################################################
if False:'''output_route = 'test_result/group'
for i in range(nb_comp):
    output_file = output_route + str(i) +'.txt'
    with open (output_file, "w") as text_file: 
        id_gk = id_group[i]
        text_file.write("{}".format(id_gk))
    text_file.close()

output_others = output_route + 'others.txt'
other_nodes = []
for u in G_proj.nodes_iter():
    if(not sub_k_proj.has_node(u)):
        other_nodes.append(u)
with open (output_others,"w") as text_file:
    text_file.write("{}".format(other_nodes))
text_file.close()

##################################################################
# deal with isolated nodes
k = 0
sub_0_proj= nx.Graph()
for u,v in G_proj.edges_iter() :
    w = G_proj[u][v]['weight']
    if (w > k):
        sub_0_proj.add_edge(u,v,weight = w)


############


if False:'''
g_proj =   bipartite.projected_graph(G,data_user , multigraph=True)
G_proj = nx.Graph()
#######change multi edges to weight
for u,v in g_proj.edges_iter() :
    w = 1
    if G_proj.has_edge(u,v):
        G_proj[u][v] ['weight'] += w
        
    else:
        G_proj.add_edge(u,v, weight=w)

######## subset graph of G_proj with edges >k
k = 2
sub_k_proj= nx.Graph()
for u,v in G_proj.edges_iter() :
    w = G_proj[u][v]['weight'] 
    if (w > k):
        sub_k_proj.add_edge(u,v,weight = w)

######## connected components
id_group = sorted(nx.connected_components(sub_k_proj),key = len, reverse = True)
nb_comp = nx.number_connected_components(sub_k_proj)


##################################################################
output_route = 'test_result/group'
for i in range(nb_comp):
    output_file = output_route + str(i) +'.txt'
    with open (output_file, "w") as text_file: 
        id_gk = id_group[i]
        text_file.write("{}".format(id_gk))
    text_file.close()

output_others = output_route + 'others.txt'
other_nodes = []
for u in G_proj.nodes_iter():
    if(not sub_k_proj.has_node(u)):
        other_nodes.append(u)
with open (output_others,"w") as text_file:
    text_file.write("{}".format(other_nodes))
text_file.close()

##################################################################
# deal with isolated nodes
k = 0
sub_0_proj= nx.Graph()
for u,v in G_proj.edges_iter() :
    w = G_proj[u][v]['weight']
    if (w > k):
        sub_0_proj.add_edge(u,v,weight = w)

nb_comp = nx.number_connected_components(sub_k_proj)


##################################################################
output_route = 'test_result/group'
for i in range(nb_comp):
    output_file = output_route + str(i) +'.txt'
    with open (output_file, "w") as text_file: 
        id_gk = id_group[i]
        text_file.write("{}".format(id_gk))
    text_file.close()

output_others = output_route + 'others.txt'
other_nodes = []
for u in G_proj.nodes_iter():
    if(not sub_k_proj.has_node(u)):
        other_nodes.append(u)
with open (output_others,"w") as text_file:
    text_file.write("{}".format(other_nodes))
text_file.close()

##################################################################
# deal with isolated nodes
k = 0
sub_0_proj= nx.Graph()
for u,v in G_proj.edges_iter() :
    w = G_proj[u][v]['weight']
    if (w > k):
        sub_0_proj.add_edge(u,v,weight = w)

######## connected components
id_group_0 = sorted(nx.connected_components(sub_0_proj),key = len, reverse = True)
nb_comp_0 = nx.number_connected_components(sub_0_proj)
print nb_comp_0
#for i in range(n_comp_0) :
for u in other_nodes:
    for v in other_nodes:
        if  G_proj.has_edge(u,v):
            w = G_proj[u][v]['weight']
            if(w >0):
                sub_0_proj.add_edge(u,v,weight = w)

#print sub_0_proj.nodes()
#print sub_0_proj.number_of_nodes()
id_group_0 = sorted(nx.connected_components(sub_0_proj),key = len, reverse = True)
nb_comp_0 = nx.number_connected_components(sub_0_proj)
#print id_group_0
#print len(other_nodes)
mat_nodes_list = sub_0_proj.nodes()
mat_proj =  nx.adjacency_matrix(sub_0_proj)
#print mat_proj

##################try katz score
graphs = list(nx.connected_component_subgraphs(G_proj))
###################################
##############katz################
##################################
def  subGraph (Graph,theshold) :  # subgraph with edges > theshold
    sub_k_proj= nx.Graph()
    for u,v in Graph.edges_iter() :
        w = Graph[u][v]['weight']
        if (w > theshold):
            sub_k_proj.add_edge(u,v,weight = w)
    #print sub_k_proj.nodes()
    return sub_k_proj

ID_ind = pd.DataFrame (columns = ['ID','ind'])
dtypes = {'ID': 'int', 'ind': 'int'}
ind = 0
for i in range( len(graphs)):
    gi = graphs[i]
#print g0.edges(data= True)
    mat_i = nx.adjacency_matrix(gi).todense()
    alpha = 1/(LA.norm(mat_i))/2
    I = np.identity(mat_i.shape[0])
