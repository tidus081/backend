# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:15:07 2018

@author: Robbie
"""
import csv
import sys
sys.path.append("C:\\Users\Robbie\Desktop\MIT\postgre")
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))

'''
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, ForeignKey
db_string = "postgres://admin:donotusethispassword@aws-us-east-1-portal.19.dblayer.com:15813/compose"
SQLALCHEMY_DATABASE_URI = 'postgresql://robbie:rob6083@localhost/demosql'
db = create_engine(SQLALCHEMY_DATABASE_URI)
'''
###############################################################################
# pull the zendesk data
from zenpy import Zenpy
import datetime
creds = {
    'email' : 'zlf123890@gmail.com',
    'token' : 'wyZXHVc3upaxZaJ9FU4UvNFJ9MsCu2YXTJXPJUja',
    'subdomain': 'robbie'
}

# Default
zenpy_client = Zenpy(**creds)

past = datetime.datetime.now() - datetime.timedelta(days=300)
result_generator = zenpy_client.tickets.incremental(start_time=past)
colname=['Date', 'Requester', 'Subject', 'Description', 'Priority', 'Tags']
myData=[]
for ticket in result_generator:
    print(ticket.created,',',ticket.requester,',',ticket.subject)
    print(ticket.priority,',',ticket.tags,',',ticket.description)
    myData.append([ticket.created, ticket.requester, ticket.subject,
     ticket.priority, ticket.tags, ticket.description])
df = pd.DataFrame(myData, columns=colname)
df
df.to_csv('test1.csv', sep=',')

################################################################################
# pull the text classification data
# 2nd version would be save all 

def pull_src_text(src, j =0):
    text_list =[]
    while (j <10000):
        j+=1
        file_location = "C:\\Users\\Robbie\\Desktop\\MIT\\zendesk\\unlabeled\\"
        file_location += str(src) +"\\" +str(j) +".txt"
        try:
            text = pd.read_fwf(file_location,index_col=False)
            tmp =[]
            for r in range(text.shape[0]):
                if text.iloc[r,0] == "### introduction ###":
                    
                    text_list.append("".join(tmp))
                    break
                tmp.append(text.iloc[r,0])
        except:
            pass
    return text_list

def NLPpreprocess(data):
    data = pd.DataFrame(data)
    textcorpus, filtered_list = data.iloc[1:,0].tolist(), []
    for text in textcorpus:
        text = text.lower()
        words = tokenizer.tokenize(text)
        words = [word for word in words if word not in stopWords]
        sen = " ".join(words)
        filtered_list.append(sen)
    return filtered_list
# make data frame
src_list = ["arxiv","jdm","plos"]
src_d={}
for i in range(3):
    tmp_t= pull_src_text(src_list[i])
    src_d[src_list[i]] = NLPpreprocess(tmp_t)
    
data_arxiv = pd.DataFrame({'text':src_d[src_list[0]],
                            'tag':["arxiv"]*len(src_d[src_list[0]])})
data_jdm = pd.DataFrame({'text':src_d[src_list[1]],
                          'tag':["jdm"]*len(src_d[src_list[1]])})
data_plos = pd.DataFrame({'text':src_d[src_list[2]],
                           'tag':["plos"]*len(src_d[src_list[2]])})

data_arxiv.to_csv('abstraction_arxiv.csv', sep=',', index=False)
data_jdm.to_csv('abstraction_jdm.csv', sep=',',index=False)
data_plos.to_csv('abstraction_plos.csv', sep=',',index=False)

frame = [data_arxiv,data_jdm,data_plos]
data_src = pd.concat(frame)
data_src.shape

data_src.to_csv('data_src.csv', sep=',')

################################################################################
# store abstraction source data in postgreSQL
    
import psycopg2
conn = psycopg2.connect("host=localhost dbname=demosql user=robbie")
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS absArxiv')
cur.execute("""
           CREATE TABLE absArxiv(
           Abstraction text, 
           label text)
           """)
    
cur.execute('DROP TABLE IF EXISTS absJdm')
cur.execute("""
           CREATE TABLE absJdm(
           Abstraction text, 
           label text)
           """)
           
cur.execute('DROP TABLE IF EXISTS absPlos')
cur.execute("""
           CREATE TABLE absPlos(
           Abstraction text, 
           label text)
           """)
conn.commit()          
           
cur = conn.cursor()
with open('abstraction_arxiv.csv','r') as f:
    next(f)
    cur.copy_from(f, 'absArxiv', sep=',')
    
with open('abstraction_jdm.csv','r') as f:
    next(f)
    cur.copy_from(f, 'absJdm', sep=',')
    
with open('abstraction_plos.csv','r') as f:
    next(f)
    cur.copy_from(f, 'absPlos', sep=',')
conn.commit()           

conn.commit()
cur = conn.cursor()
# Read
cur.execute("SELECT * FROM absArxiv")
rows = cur.fetchall()
for i in range(0,6):  
    print(rows[i])
    
cur.execute("SELECT * FROM absJdm")
rows = cur.fetchall()
for i in range(0,6):  
    print(rows[i])
    
cur.execute("SELECT * FROM absPlos")
rows = cur.fetchall()
for i in range(0,6):  
    print(rows[i])
################################################################################  
# store zendesk data in postgreSQL
    
import psycopg2
conn = psycopg2.connect("host=localhost dbname=demosql user=robbie")
cur = conn.cursor()
         
cur.execute('DROP TABLE IF EXISTS zendesktest')

cur.execute("""
           CREATE TABLE zendesktest(
           Date text, 
           Requester text, 
           Subject text, 
           Description text,
           Priority text,
           Tags text)
           """)
conn.commit()
           
cur = conn.cursor()
with open('test2.csv','r') as f:
    next(f)
    cur.copy_from(f, 'zendesktest', sep=',')
           
    
cur.execute('DROP TABLE IF EXISTS advertising')

cur.execute("""
           CREATE TABLE IF NOT EXISTS advertising(
           TV text, 
           Radio text, 
           Newspaper text, 
           Sales text
           )
           """)
           
cur.execute('DROP TABLE IF EXISTS "stock";')
cur.execute("""
           CREATE TABLE stock(
           time text, 
           close text
           )
           """)
           
cur.execute('DROP TABLE IF EXISTS "gre";')
cur.execute("""
           CREATE TABLE gre(
           gre text, 
           gpa text, 
           rank text, 
           admit text
           )
           """)
           
conn.commit()
cur = conn.cursor()
with open('advertising.csv','r') as f:
    next(f)
    cur.copy_from(f, 'advertising', sep=',')
        
with open('stock.csv','r') as ff:
    next(ff)
    cur.copy_from(ff, 'stock', sep=',')
    
with open('gre.csv','r') as f2:
    next(f2)
    cur.copy_from(f2, 'gre', sep=',')
    
conn.commit()
cur = conn.cursor()
# Read
cur.execute("SELECT * FROM advertising")
rows = cur.fetchall()
for i in range(0,6):  
    print(rows[i])
    
cur.execute("SELECT * FROM stock")
rows = cur.fetchall()
for i in range(0,6):  
    print(rows[i])
    
cur.execute("SELECT * FROM gre")
rows = cur.fetchall()
for i in range(0,6):  
    print(rows[i])
    
    
conn.close()
    