import pandas as pd
import igraph
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from numpy.linalg import pinv
from numpy import linalg as LA
from scipy import stats


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
#####get number of n share pages node

G_sub = nx.Graph()
for u0,v0 in G.edges_iter() :
    user_neighbors = G.neighbors(u0)
    for v1 in user_neighours:
        u1 = G.neighbors(v1)
        
'''
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
'''
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
    distence = pinv(I- alpha*mat_i)- I
    nodes_i =  graphs[i].nodes()
    if len(nodes_i) <= 5:
        # treat them as a group
        for v in nodes_i:
            ID_ind.loc[len(ID_ind)] = [v, ind]
        ind = ind + 1
    else :         
# find a cut 
        cut_thres =  3.5
        n,m = np.shape(distence)
        flat_mat_i = (np.asarray(distence)).reshape(-1)        
    #print len(flat_mat_i)
    #print flat_mat_i
   # loc, scale = stats.expon.fit(flat_mat_i)
   # print (loc,scale)
    #print stats.anderson(flat_mat_i,dist= 'expon')
   # print stats.expon.ppf(.90 , loc = loc, scale = scale )
        print np.percentile(flat_mat_i,93)
        cut_thres = np.percentile(flat_mat_i,93)
    ### initalize new graph based on katz score
        dt = [('weight', float)] 
        G_katz = nx.from_numpy_matrix(distence)
    ## generate mapping between nodes index and ID
        mapping = dict(zip(range(n),gi.nodes()))
    ## relable graph
        G_katz = nx.relabel_nodes(G_katz, mapping)
    #print G_katz.edges(data = True)
        G_katz_sub = subGraph(G_katz, cut_thres)
  #  print G_katz_sub.nodes()
        group_i = sorted(nx.connected_components(G_katz_sub),key = len, reverse = True)
        nb_comp = nx.number_connected_components(G_katz_sub)
        #print len(G_katz_sub.nodes())
        #print nb_comp
        for j in range(nb_comp):
            for k in group_i[j] :
                ID_ind.loc[len(ID_ind)] = [k,ind]
            ind = ind +1
    #####get subgraph for other group 
        G_other = G_katz.copy()
        for u in G_katz_sub.nodes_iter():
            G_other.remove_node(u)
         
   # remove self loops
        for u in G_other.nodes_iter():
            G_other.remove_edge(u,u)        
    # remove small weight edges
        G_other_sub = nx.Graph()
        for u,v in G_other.edges_iter():
            w = G_other[u][v]['weight']
            if w >=cut_thres/10 :
                G_other_sub.add_edge(u,v,weight = w)
        print nx.number_connected_components(G_other_sub)
        print sorted(nx.connected_components(G_other_sub),key = len, reverse = True)
    
        for v in G_other.nodes (): 
            ID_ind.loc[len(ID_ind)] = [v, ind]
            ind = ind + 1
        print G_other_sub.neighbors(1438677009494522)
#print ID_ind['ID'].dtype

for c in ID_ind.columns:
    ID_ind[c] = ID_ind[c].astype(dtypes[c])
#print ID_ind 
########################write result
ID_ind.to_csv('test_result/new.txt', index = False)

if False: '''
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
text_file.close()'''
################################
 
