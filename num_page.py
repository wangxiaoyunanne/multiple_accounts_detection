import glob
import numpy as np 
import pandas as pd

files = glob.glob('sample_file/pop_name/*.txt')
files.sort()
whole_set = set()
for fi in files[4:]:
    data = pd.read_csv(fi, header = None, sep = ' ')
    pages = data.iloc[:,1]
    page_set = set(pages)
    whole_set = whole_set.union(page_set)
    print len(page_set)
    print len(whole_set)



