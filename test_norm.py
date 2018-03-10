import numpy as np
import pandas as pd
from numpy import linalg as LA
import networkx as nx
from networkx.algorithms import bipartite
################## process bipartite graph to weighted projection graphs 
################# node2vec use this graph

df = pd.read_table("test_file.txt", sep = '\t', header = None, names = ["user","page"])
#for i in range(len(df)):
    #print df.ix[i,::]
g = nx.from_pandas_dataframe(df, 'user', 'page')
G = nx.Graph()
data_user = df['user']

for u,v in g.edges_iter() :
    w = 1
    if G.has_edge(u,v):
        G[u][v] ['weight'] += w
    else:
        G.add_edge(u,v, weight=w)

g_proj =   bipartite.projected_graph(G,data_user , multigraph=True)
G_proj = nx.Graph()
for u,v in g_proj.edges_iter() :
    w = 1
    if G_proj.has_edge(u,v):
        G_proj[u][v] ['weight'] += w
        
    else:
        G_proj.add_edge(u,v, weight=w)

nx.write_edgelist(G_proj, "test_edgelist.txt", data = ['weight'])
