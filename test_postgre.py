# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:03:00 2018

@author: Robbie
"""
import sys
sys.path.append(r"C:\Users\Robbie\Desktop\MIT\text_classification")
import postgreRESTful
import prexcel3
import pandas as pd, numpy as np, matplotlib.pyplot as plt


import psycopg2
conn = psycopg2.connect("host=localhost dbname=demosql user=robbie")
cur = conn.cursor()
cur.execute("SELECT * FROM advertising")
advertising_raw = cur.fetchall()
for i in range(3):
    print(advertising_raw[i])
    
cur.execute("SELECT * FROM stock")
stock_raw = cur.fetchall()
for i in range(3):
    print(stock_raw [i])
    
cur.execute("SELECT * FROM gre")
gre_raw = cur.fetchall()
for i in range(3):
    print(gre_raw [i])

conn.close()

# test with regression data    
postgreRESTful.build_model(advertising_raw)
temp_test1 = list(advertising_raw[99])
temp_test1 = temp_test1[0:3]
print(temp_test1)
postgreRESTful.predict(temp_test1,col=3)
postgreRESTful.anomaly(23,temp_test1,col=3)

# test with classification data   
postgreRESTful.build_model(gre_raw)
temp_test2 = list(gre_raw[99])
temp_test2 = temp_test2[0:3]
print(temp_test2)
postgreRESTful.predict(temp_test2,col=3)
postgreRESTful.anomaly('Accepted',temp_test2,col=3)

temp_test3 = list(gre_raw[99])
temp_test3 = [temp_test3[i] for i in [0,1,3]]
print(temp_test3) 
postgreRESTful.predict(temp_test3,col=2)
postgreRESTful.anomaly('4',temp_test3,col=2)

# test with timeseries data 
postgreRESTful.build_model(stock_raw)
print(stock_raw)
temp_test4 = list(stock_raw[19])
print(temp_test4)
postgreRESTful.predict(temp_test4,col=1)
postgreRESTful.anomaly(17,temp_test4,col=1)

#test text-classification

### load train and test_data 
src_train = pd.read_csv('train_src.csv')
src_train = [ src_train.iloc[i,:].tolist() for i in range(src_train.shape[0])]
print(src_train)
src_test = pd.read_csv('test_src.csv')
src_test.shape

postgreRESTful.build_model(src_train)
temp_test5 =src_test.iloc[50,:].tolist()
print(temp_test5[0])

postgreRESTful.predict(temp_test5[0],col=1)
postgreRESTful.anomaly('arxiv',temp_test5[0],col=1)


# test with abstraction source data   
ptg_reg_model = prexcel3.Model()
ptg_reg_model.data('postgreSQL', advertising_raw)
ptg_reg_model.predict(temp_test,col=3)
ptg_reg_model.anomaly(20,temp_test,col=3)

#test regression
ptg_clas_model = prexcel3.Model()
ptg_clas_model.data('postgreSQL', gre_raw)
ptg_clas_model.predict(temp_test,col=3)
ptg_clas_model.anomaly('Accepted',temp_test,col=3)

#test classification
ptg_clas_model = prexcel3.Model()
ptg_clas_model.data('postgreSQL', gre_raw)
ptg_clas_model.predict(temp_test,col=2)
ptg_clas_model.anomaly('3',temp_test,col=2)

#test time-series
ptg_time_model = prexcel3.Model()
ptg_time_model.data('postgreSQL', stock_raw) 
temp_test = list(stock_raw[19])
print(temp_test)
ptg_time_model.predict(temp_test,col=0)
ptg_time_model.anomaly(17,temp_test,col=1)

###############################################################################
#test text classification
import psycopg2
conn = psycopg2.connect("host=localhost dbname=demosql user=robbie")
cur = conn.cursor()
cur.execute("SELECT abstraction FROM absArxiv")
data_arxiv = cur.fetchall()

cur.execute("SELECT abstraction FROM absJdm")
data_jdm = cur.fetchall()
        
cur.execute("SELECT abstraction FROM absPlos")
data_plos = cur.fetchall()

for i in range(0,3):  
    print(data_arxiv[i])
    print(data_jdm[i])
    print(data_plos[i])
    
conn.close()

# extract train and test data from Postgre TABLE
# make a train data
tmp_arxiv = [list(sen)[0] for sen in data_arxiv[0:250]]
tmp_jdm = [list(sen)[0] for sen in data_jdm[0:250]]
tmp_plos = [list(sen)[0] for sen in data_plos[0:250]]

header = pd.DataFrame({'abs':['text'],'tag':['label']})
df_arxiv = pd.DataFrame({"abs":tmp_arxiv, "tag":['arxiv']*250})
df_jdm = pd.DataFrame({"abs":tmp_jdm, "tag":['jdm']*250})
df_plos = pd.DataFrame({"abs":tmp_plos, "tag":['plos']*250})

frame = [header,df_arxiv, df_jdm,df_plos]
src_train = pd.concat(frame)
src_train.to_csv('train_src.csv', sep=',',index=False)
### load train_data
src_train = pd.read_csv('train_src.csv')
src_train.tolist()

#make a test data
tmp_arxiv = [list(sen)[0] for sen in data_arxiv[250:280]]
tmp_jdm = [list(sen)[0] for sen in data_jdm[250:280]]
tmp_plos = [list(sen)[0] for sen in data_plos[250:280]]

header = pd.DataFrame({'abs':['text'],'tag':['label']})
df_arxiv = pd.DataFrame({"abs":tmp_arxiv, "tag":['arxiv']*30})
df_jdm = pd.DataFrame({"abs":tmp_jdm, "tag":['jdm']*30})
df_plos = pd.DataFrame({"abs":tmp_plos, "tag":['plos']*30})

frame = [header,df_arxiv, df_jdm,df_plos]
src_test = pd.concat(frame)
src_test.shape

src_test.to_csv('test_src.csv', sep=',',index=False)
### load test_data
src_test = pd.read_csv('test_src.csv')

# change data to list of lists to unify the data format
src_train = pd.read_csv('train_src.csv')
src_train = [ src_train.iloc[i,:].tolist() for i in range(src_train.shape[0])]
print(src_train)
src_test = pd.read_csv('test_src.csv')
src_test.shape
src_train.shape
# test with abstraction source data
   
ptg_model = prexcel3.Model()
ptg_model.data('postgreSQL', src_train)
print(src_test[50])
ptg_model.predict(src_test[50][0],col=1)
ptg_model.anomaly('arxiv',src_test[50][0],col=1)
            

