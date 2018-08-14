# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:23:19 2018

@author: Robbie

7/11
add anomaly, score function for reg, class, time series
haven't done Matrix estimation

"""

import numpy as np, pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse

from gensim.models import doc2vec
from collections import namedtuple
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
from datetime import datetime

class Model(object):
    def __init__(self):
        pass
    
    def data(self, src, raw_data, window=False, factor=False):
        # save build_type, lm, mse
        self.model, self.src ={} ,src
        self.encode_d, self.decode_d = {}, {} 
        #When we set a model, it will train data and built a model
        raw_data = pd.DataFrame(raw_data)
        self.header = raw_data.iloc[0,:]
        self.rawdata = self.empty_filter(raw_data)
        self.window, self.timecol = 3, None # for time 
        self.factor = 3 # for ME
        self.model_filter()
    
    def empty_filter(self, data):
        ginput = pd.DataFrame(data)
        ginput = ginput.drop(0)            
        droplist =[]
        # remove all empty data
        for r in range(ginput.shape[0]):
            for c in range(ginput.shape[1]):
                if ginput.iloc[r,c] == "" :
                    droplist.append(r)
        filtered = ginput.drop([r+1 for r in droplist])
        return filtered   
    
    
                
    def model_filter(self):
        # check header contain text or time
        text_col = [i for i in range(len(self.header)) if 'text' == self.header[i]]
        
        if len(text_col) >0 :
            text_col, class_col= text_col[0], abs(1-text_col[0])
            self.model[class_col] =['text clas'] + self.train(self.rawdata, class_col, build_type ='text clas')
            print('text classification model is saved')
        else:
            data = self.rawdata
            col = data.shape[1]
            # check whether data has time variable
            for i in range(col):
                try:
                    if (len(data.iloc[1,i].split('/')) ==3) or (len(data.iloc[1,i].split('-')) ==3):
                        self.timecol = i
                        for j in range(col):
                            self.model[j] = ['time']+ self.train(data, j, build_type ='time')
                            print('time series column {} is saved'.format(j))
                        return 
                except:
                    pass
                
            if not self.timecol:    
            # check whether a column has string or a few values
                for i in range(col):
                    y_raw=[data.iloc[r,i] for r in range(data.shape[0]) if data.iloc[r,i] != ("" or "#ERROR!") ]
                    if len(set(y_raw)) < 6 :
                        self.model[i] =['clas'] + self.train(data, i, build_type ='clas')
                        print('classification column {} is saved'.format(i))
                    else:
                        self.model[i] = ['reg'] + self.train(data, i, build_type ='reg')
                        print('regression column {} is saved'.format(i))
    
    def train(self, filtered, ycol, build_type = False):
        lm, x_train, y_train = self.preprocess_train(filtered, ycol, build_type = build_type)
        print('build_type :',build_type)
        lm.fit(x_train, y_train)
        y_fit = lm.predict(x_train)
            
        if (build_type =='reg') or (build_type =='time'):     
            mse = mean_squared_error(y_train, y_fit)
            
        elif (build_type == 'clas') or (build_type =='text clas'):
            y_fit,res = y_fit.astype('int'), 0
            for i in range(len(y_fit)):
                if y_fit[i] != y_train.iloc[i,0]:
                    res+=1
            mse = res/len(y_train)
            
        print("check mean error :",mse)    
        return [lm, mse]
    
    def preprocess_train(self, filtered, ycol, build_type = False):
        lm=linear_model.LinearRegression()
        # train data contain x, y data
        x_filtered = filtered.drop(filtered.columns[ycol], axis=1)
        if build_type=='time':
            filtered.iloc[:,self.timecol] = pd.to_datetime(filtered.iloc[:,self.timecol])
            filtered = filtered.sort_values(self.timecol)
            #check we are going to train time column
            if ycol !=self.timecol:
                time_values, new_feature = filtered.iloc[:,ycol], {}
                y_train = pd.DataFrame(time_values[self.window:time_values.shape[self.timecol]])
            else:
                time_values, new_feature = filtered.iloc[:,ycol].values.astype(float), {}
                y_train = pd.DataFrame(time_values[self.window:time_values.shape[self.timecol]])
                
            for i in range(self.window):
                new_feature[i] = time_values[i:time_values.shape[0]-(self.window-i)].tolist()
            x_train = pd.DataFrame(new_feature)
            print(x_train.head())
            
        elif build_type=='reg':
            # convert discrete column to dummy variables 
            X = self.dummyProcess(x_filtered)
            x_train, y_train = preprocessing.normalize(X), pd.DataFrame(filtered.iloc[:,ycol])
            # in case, a sklearn algorithm doesn't understand list or panda dataframe
            
        elif build_type=='clas':
            x_train = preprocessing.normalize(self.dummyProcess(x_filtered) )
            y_train = self.classConvert(filtered.iloc[:,ycol], ycol)
            #x_train, x_dummy, y_train, y_dummy = train_test_split(X, y, test_size=0, random_state=0)
            lm = linear_model.LogisticRegression()
            
        elif build_type=='text clas':
            x_train = self.text2vec(x_filtered)
            #preprocess class column
            y_train = self.classConvert(filtered.iloc[:,ycol], ycol)
            lm = linear_model.LogisticRegression(penalty='l2', C=.5)
            
        else:
            return "Don't have this build type yet"
        print(build_type,'train preprocess is done')
        return lm, x_train, y_train
    
    def text2vec(self,data):
        # preprocess raw data and convert text data to feature vector
        # it has one column
        data = pd.DataFrame(data)
        print(data.shape)
        tokenizer = RegexpTokenizer(r'\w+')
        stopWords, docs = set(stopwords.words('english')), []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for i  in range(1, data.shape[0]):
            text = data.iloc[i,0].lower()
            words = tokenizer.tokenize(text)
            words = [word for word in words if word not in stopWords]
            tags = [i]
            docs.append(analyzedDocument(words, tags))
        self.vecmodel = doc2vec.Doc2Vec(docs, vector_size = 100, alpha = 0.025, window = 300, 
                                       min_count = 1, min_alpha = 0.00025, dm =1)
        x_train = [self.vecmodel.docvecs[i] for i in range(len(self.vecmodel.docvecs))]
        return pd.DataFrame(x_train)
                
    def classConvert(self, raw_class, ycol):
        if ycol not in self.encode_d.keys():
            self.encode_d[ycol], self.decode_d[ycol] = {}, {}
        raw_class = raw_class.tolist()
        ori_value = list(set(raw_class))
        for i in range(len(ori_value)):
            self.encode_d[ycol][ori_value[i]], self.decode_d[ycol][i] = i, ori_value[i]
        label = [ self.encode_d[ycol][raw_class[i]] for i in range(len(raw_class))]        
        y_train = pd.DataFrame(label)
        return y_train
                
    def dummyProcess(self, filtered, ycol=False, predict=False):
        # when features have a categorical variable
        if ycol:
            # for google sheet
            X = filtered.drop(filtered.columns[ycol], axis=1)
        else:
            #for postgreSQL
            X = pd.DataFrame(filtered)
            
        X.columns = [i for i in range(X.shape[1])]
        dummylist ={}
        #check X has categorical variables
        for col in range(X.shape[1]):
            if len(set(X.iloc[:,col])) <6:
                dummylist[col]=pd.get_dummies(X.iloc[:,col], drop_first=True)
                
        X = X.drop(X.columns[list(dummylist.keys())], axis=1) 
        for dummy in dummylist:
            X = pd.concat([X, dummylist[dummy]], axis=1)
            
        return X
        
#####################################################################################################        
    def predict(self, x_raw, col=False, x_test=False, row=False):
        if col is False:
            row, col = self.find_googleSheet(x_raw)
        predict_type, self.lm, self.mse = self.model[col]
        print('predict model :',predict_type)
        
        if self.src == 'google_sheet':
            x_test = self.preprocess_predict(x_raw, col, row= row, build_type = predict_type)
        elif self.src == 'postgreSQL':
            x_test = self.preprocess_predict(x_raw, col, build_type = predict_type)
        else:
            return "error"
        
        print('test input:')
        print(x_test)
        
        if predict_type=='time':
            y_predicted = self.lm.predict(x_test)
            y_predicted=y_predicted.astype('float')
            if col !=self.timecol:
                return round(y_predicted[0][0],4)
            else:
                return str(pd.to_datetime(y_predicted[0][0]))
        
        elif predict_type=='reg':
            y_predicted = self.lm.predict(x_test)
            print(y_predicted[0])
            if self.src == 'google_sheet':
                return (y_predicted[0][0]//0.01)/100
            elif self.src == 'postgreSQL':
                return (y_predicted[0]//0.01)/100
            else:
                return "error"
        
        elif (predict_type=='clas') or (predict_type=='text clas'):
            y_predicted = self.lm.predict(x_test)
            y_predicted = y_predicted[0]
            y_predicted=self.decode_d[col][int(y_predicted)]
            return y_predicted
        
        else:
            return "error"
        
        '''
        elif predict_type=='text clas':
            y_predicted = self.lm.predict(x_test)
            y_predicted = y_predicted[0]
            print(y_predicted)
            print(self.decode_d)
            y_predicted=self.decode_d[col][int(y_predicted)]
            return y_predicted
        '''
        
        
    def find_googleSheet(self, x_raw):
        # find which column we want to predict
        filtered = self.empty_filter(x_raw)
        for row in range(filtered.shape[0]):
            for col in range(filtered.shape[1]):
                if filtered.iloc[row,col] == "#ERROR!":
                    return row, col
                
    def preprocess_predict(self, x_raw, ycol, row=False, build_type = False):
        # check empty cell and save it in test data and remove this row from training data
        if self.src == 'google_sheet':
            filtered = self.empty_filter(x_raw)
            if build_type=='time':
                x_test = [filtered.iloc[row-j,ycol] for j in range(self.window,0,-1)] # from 3 to 1 not current cell
                if ycol == self.timecol:
                    ndt = pd.to_datetime(x_test)
                    x_test=ndt.values.astype(float)
                                            
                x_test = pd.DataFrame(x_test).T
                
            elif (build_type == 'reg') or (build_type == 'clas'):
                X = self.dummyProcess(filtered, ycol=ycol)
                x_test = preprocessing.normalize(pd.DataFrame(X.iloc[row,:]).T)
                
            elif build_type == 'text clas':
                row, col = self.find_googleSheet(x_raw)
                test_raw = self.rawdata.iloc[:,abs(1-col)]
                x_total_text = self.text2vec(test_raw)
                x_test = pd.DataFrame(x_total_text.iloc[row,:]).T
            else:
                return "build type error"
                
        elif self.src == 'postgreSQL':
            if build_type=='time':
                x_test = self.findTimeLocation(x_raw, ycol)
                x_test = pd.DataFrame(x_test.iloc[:,ycol]).T
                if ycol == self.timecol:
                    x_test = x_test.values.astype(float)
                
            elif (build_type == 'reg') or (build_type == 'clas'):
                tmp, rawData = self.rawdata, pd.DataFrame(x_raw).T
                x_filtered = tmp.drop(tmp.columns[ycol], axis=1).append(rawData) 
                X= self.dummyProcess(x_filtered, predict=1)
                x_test = preprocessing.normalize(X.iloc[[-1]])
                
            elif build_type == 'text clas':
                test_raw = self.rawdata.iloc[:,abs(1-ycol)].append(pd.Series(x_raw))
                x_total_text = self.text2vec(test_raw)
                x_test= pd.DataFrame(x_total_text.iloc[x_total_text.shape[0]-1,:]).T
            else:
                return "build type error"
                
        else:
            return "this source is not ready"
                
        return x_test
     
    def anomaly(self, y_real, data, col=False, row=False):
        if self.src =='postgreSQL':
            build_type, lm, mse = self.model[col]
        elif self.src =='google_sheet':
            row, col = self.find_googleSheet(data)
            build_type, lm, mse = self.model[col]
        else:
            return "not ready"
        
        #postgreSQL anomaly data would be 
        '''
        if build_type == 'text clas':
            test_raw = self.rawdata.iloc[:,abs(1-col)].append(pd.Series(data))
            x_test = self.text2vec(test_raw)
            x_test = pd.DataFrame(x_test.iloc[x_test.shape[0]-1,:]).T
            idx = self.encode_d[col][y_real] 
            return (1-lm.predict_proba(x_test)[0][idx])/(len(self.encode_d[col].keys())-1 )
        '''    
        if (build_type == 'reg') or (build_type == 'time') or (build_type == 'ME'):
            if self.src == 'google_sheet':
                y_predicted = self.predict(data)
            elif self.src =='postgreSQL':
                y_predicted = self.predict(data, col=col)
                
            
            if mse < 0.00000001:
                temp = 10
            else:
                temp = abs((y_predicted - y_real)/mse)    
            print("Inside Predicted: %s %s" % (build_type,y_predicted))
            # anomaly score
            result = norm.cdf(abs(temp)) * 100.0
            return round(( (result -50)*2),4)
        
        elif (build_type == 'clas') or (build_type == 'text clas'):
            x_test = self.preprocess_predict(data, col, row = row, build_type = build_type)
            idx = self.encode_d[col][y_real] 
            res = (1 - lm.predict_proba(x_test)[0][idx])*100
            if build_type =='text clas':
                return round((res)/(len(self.encode_d[col].keys())-1 ),4)
            return round(res,4)
        
        else:
            return'error'
        
    def score(self):
        if self.src == 'google_sheet':
            if self.build_type in ['reg', 'clas', 'time', 'ME']:
                return self.mse
            
    def findTimeLocation(self,test,ycol):
        data_time, tcol = self.rawdata.iloc[:,self.timecol].tolist(), self.timecol
        #test = pd.to_datetime(test[self.timecol])
        if test[tcol] > str(max(data_time)):
            return self.rawdata.iloc[self.rawdata.shape[0]-3:self.rawdata.shape[0],:] 
        
        for i in range(data_time.shape[0]):
            if str(data_time[i]) < test[tcol] < str(data_time[i+1]):
                return self.rawdata.iloc[self.rawdata.iloc[i-3,:]:self.rawdata.iloc[i,:],:] 
            
   


        
    
    


    
    

    
    
        
    
        
        