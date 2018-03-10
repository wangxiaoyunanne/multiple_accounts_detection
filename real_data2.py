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

def shuffle (n_id, num1, num2):
    alpha_set =['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9','0',',','<','.','>',':',';','[','{','}',']','?','/','|','_','-','+','=','!','@','#','$','%','^','&','*','(',')','~','`']
    r_num = random.randint(0,num1-1)
    r_num2 = random.randint(0,num2-1)
    return str(n_id)+alpha_set[r_num] + alpha_set[r_num2]


# read file
sub_files = glob.glob('sample_file/shuffled/data2/diff_size/1mi_4.edgelist')
sub_files.sort()
for fi in sub_files:
    subGraph = nx.read_edgelist(fi, create_using = nx.MultiGraph())
# get different nodes
    node = subGraph.nodes()
    node.sort()
    page_nodes = node[100:]
    user_nodes = node[:100]
    shuffle_num1 = 15
    shuffle_num2 = 1
    shuffle_rate = 0.5
# shuffled graph
    s_graph = nx.MultiGraph()
    for u,v in subGraph.edges_iter():
        if u in user_nodes:
            u_2 = shuffle(u,shuffle_num1,shuffle_num2)
            s_graph.add_edge(v,u_2)
        elif v in user_nodes:
            v_2 = shuffle(v, shuffle_num1, shuffle_num2)
            s_graph.add_edge(u,v_2)
# to edgelist
    s_file = fi[:32]+'_s15.edgelist'
    nx.write_edgelist(s_graph,s_file)

