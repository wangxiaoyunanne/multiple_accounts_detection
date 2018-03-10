from urllib2 import urlopen
#from robobrowser import RoboBrowser
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import numpy as np
import re
import glob

############# input is str not int
# primary_id_w is for wang, primary_id is for pan primary_id_f
def primary_id_p(fb_id):
	driver = webdriver.Firefox()
	driver.get("https://www.facebook.com")
	elem = driver.find_element_by_name("email")
        elem.send_keys("sincere.crawl.cindy@gmail.com")#facebook username
        #elem.send_keys("cmlaih2@gmail.com")
        #elem.clear()
	elem = driver.find_element_by_name("pass")#facebook password
	#elem2.clear()
        elem.send_keys("cindycrawford")
        #elem.send_keys("Zxcv1234")
	#driver.implicitly_wait(5)
	elem.send_keys(Keys.RETURN)
	#elem = driver.find_element_by_css_selector(".selected")
	#print ("test")
	elem.click()
	time.sleep(5) # make a deley
	#url1 = '100002065534410'
	url = "https://www.facebook.com/" + fb_id
        #print url
        driver.get(url)
        #time.sleep(3) 
	r_id=driver.current_url#.replace("https://www.facebook.com/", "")
	#print (url)
	print (r_id)
	driver.quit()
	return r_id[25: ]
####################################################
# when the input is a weighed projected graph
result = pd.read_table("Downloads/new.txt",sep = ',')
result['truth'] = pd.Series (np.zeros(len(result)))
#i = '964255970322560'
result['user'] = result['user'].astype('str')
for i in range (len(result)): 
    result['truth'][i] = primary_id(result['user'][i])
#print isblocked(i)
result.to_csv('Desktop/sincere data/id_cons.txt', index = False)
#####################################################
# when the input is a bipartite graph
result = pd.read_table("name50_120146.txt",sep = '\t', header = None)
# get the unique users IDs
######## read mult files

def read_list(raw_list):
    names_pos = [m.start() for m in re.finditer(',', raw_list)]
    l = len(raw_list)
    ID_li = []
    ID_li.append(raw_list [1: names_pos[0] ] )
    for i in range(len( names_pos)-1):
        id_s = str(raw_list[names_pos[i]+2:names_pos[i+1]])
        ID_li.append(id_s)
#
    ID_li.append(raw_list[names_pos[-1]+2: l-1])
    print ID_li
    print len(ID_li)
    return ID_li

files = glob.glob('sample_name/*.txt')
files.sort()
whole_list = []
for single_file in files[17:25]:
    print single_file
    truth_file = 'truth_file/truth'+ single_file[25:]
    print truth_file
    with open(single_file,'r') as input_file:
        ID_list =input_file.read() 
    ID_list =read_list(ID_list)
    ID_list = np.asarray(ID_list)
    ID_list = ID_list.astype('str') 
    outputnew = []
    for i in ID_list[0:50]:
        truth = primary_id_w(i)
        outputnew.append([i,truth])
    for i in ID_list[50:]:
        truth =  primary_id_w(i)
        outputnew.append([i,truth])
    # append to whole list
    whole_list.append(outputnew)
    output_1 = pd.DataFrame(outputnew)
    #print ID_list
    output_1.to_csv(truth_file, index = False)

ID_list = np.asarray(ID_list)
ID_list = ID_list.astype('str')
outputnew = []
for i in ID_list[50:]:
    truth = primary_id_w(i)
    outputnew.append([i,truth])

output =  pd.DataFrame(outputnew)
output.to_csv('Desktop/sincere data/test03000.txt', index = False)
