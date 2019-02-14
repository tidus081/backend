# -*- coding: utf-8 -*-
"""
Edited on Wed Aug 22 2018

@author: Robbie

10/10
Sources : 1. Google sheet 2. PostegreSQL
Predict Type : 1. Regression 2. Classification 3. Time Series Prediction 4. Text Classification
Function Type : 1. Prediction 2. Anomaly detection

Next step would be Image and Video classification.

Process
  save_model : store train data which is used for training and making x features for test
      |
 model_filter : check predict type and save it in dictionary,
      |
    train : preprocess data and result scikit model and MSE #save model dictionary using Pickle
      |
  1.predict : load scikit model, preprocess test data and predict
  2.anomaly : load predict funtion and MSE, and calculate anomaly score
load

issues
1. when we request post, we need Password
2. When we save a model, How can we seperately save each model?
3. How te detect whether text or text classification
4. In text classification case, What if we have more than 2 columns

install library lists
pip3 install numpy pandas sklearn pillow pytesseract gensim DateTime nltk flask pickle
sudo apt-get install tesseract-ocr
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

#time
from DateTime import DateTime
from dateutil.parser import parse

#OCR
from PIL import Image
from pytesseract import image_to_string
import urllib

#vec
from gensim.models import doc2vec
from collections import namedtuple
from nltk.tokenize import RegexpTokenizer

from multiprocessing import Process
from gensim.parsing.preprocessing import remove_stopwords

import ins_controller as ic
import collections 

class Model(object):
    def __init__(self):
        pass

    def save_model(self, src, raw_data, window=False, factor=False):
        # save build_type, lm, mse
        self.model, self.src ={} ,src
        self.encode_d, self.decode_d = {}, {}
        #When we set a model, it will train data and built a model
        raw_data = pd.DataFrame(raw_data)
        #self.headDrop =[]
        self.header, self.rawdata= [str(i) for i in raw_data.iloc[0,:]], self.empty_filter(raw_data)
        self.window, self.timecol = 3, -1 # for time series
        self.textcol = -1 # for text classification
        self.ocrcol = -1 # for ocr classification
        self.model_filter() # save model for each columns
        return self.model

    def empty_filter(self, data):
        # remove rows that contain empty values
        ginput = pd.DataFrame(data)
        # remove header
        ginput = ginput.drop(0)
        emptylist =[]
        # remove all empty data
        for r in range(ginput.shape[0]):
            for c in range(ginput.shape[1]):
                if (ginput.iloc[r,c] == ""):
                    if isinstance(ginput.iloc[2, c], str) or len(set(ginput.iloc[:,c])) <10:
                        ginput.iloc[r, c] = ginput.iloc[:,c].mode().tolist()[0]
                        while ginput.iloc[r, c] =="": # if above or below cell is empty, doesn't work
                            if r-1 >0:
                                ginput.iloc[r, c] = ginput.iloc[r-1, c]
                            else:
                                ginput.iloc[r, c] = ginput.iloc[r+1, c] 

                        print("fill string out:", ginput.iloc[r, c])
                    else:
                        #ginput.iloc[r, c] = np.median(ginput.iloc[:,c])
                        ginput.iloc[r, c] = 0
                        emptylist.append((r,c))
        for r,c in emptylist:
            ginput.iloc[r, c] = ginput.iloc[:,c].mean()
            print("fill numeric out:", ginput.iloc[r, c])
        return ginput

    def is_date(self,string):
        try:
            parse(string)
            return True
        except:
            return False
    
    def is_text(self, sample):
        try:
            sample = sample.split(" ")
            if len(sample) > 5:
                return True
        except:
            pass
        '''
        if len(set(df.iloc[:,c]))>100:
                type_list.append('text classification')
                break
        '''
        return False
        
    def type_classifier(self, df):
        # expect dataframe
        type_list = []
        for c in range(df.shape[1]):
            type_count = collections.Counter()
            for r in range(df.shape[0]):
                if self.is_text(df.iloc[r,c]):
                    type_count['text classification'] +=1
                elif self.is_date(df.iloc[r,c]):
                    type_count['time'] +=1
                elif isinstance(df.iloc[r,c], int) or isinstance(df.iloc[r,c], float):
                    type_count['regression'] +=1
                else:
                    type_count['classification'] +=1
            type_list.append(type_count.most_common(1)[0][0]) # col type
        return type_list
        
    def model_filter(self):
        # check header contain text or data format is time
        col = self.rawdata.shape[1]
        # right now, we should use header for OCR
        if 'OCR' in self.header:
            self.ocrcol= self.header.index('OCR')
            class_col= abs(1-self.ocrcol) # assume we have two columns
            self.model[class_col] = ['OCR']+ self.train(self.rawdata, class_col, build_type ='OCR')
        else:
            type_list = self.type_classifier(self.rawdata) # data filled out by empty_filter 
            if 'time' in type_list:
                self.timecol = type_list.index('time')
                for j in range(col):
                    self.model[j] = ['time']+ self.train(self.rawdata, j, build_type ='time')
            elif 'text classification' in type_list:
                self.textcol=type_list.index('text classification')
                class_col= abs(1-self.textcol) # assume we have two columns
                self.model[class_col] =['text classification'] + self.train(self.rawdata, class_col, build_type ='text classification')
            else:
                # regression and classification left
                for i, train_type in enumerate(type_list):
                    self.model[i] =[train_type] + self.train(self.rawdata, i, build_type =train_type)
                
        

    def train(self, filtered, ycol, build_type = False):
        lm, x_train, y_train = self.preprocess_train(filtered, ycol, build_type = build_type)
        lm.fit(x_train, y_train)
        y_fit = lm.predict(x_train)

        if (build_type =='regression') or (build_type =='time'):
            mse = mean_squared_error(y_train, y_fit)

        elif (build_type == 'classification') or (build_type =='text classification') or (build_type =='OCR'):
            y_fit,res = y_fit.astype('int'), 0
            for i in range(len(y_fit)):
                if y_fit[i] != y_train[i]:
                    res+=1
            mse = res/len(y_train)

        return [lm, mse]


    def preprocess_train(self, filtered, ycol, build_type = False, lm=None):
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
            lm=linear_model.LinearRegression()

        elif build_type=='regression':
            # convert discrete column to dummy variables
            x_train, y_train = self.dummyProcess(x_filtered), pd.Series(filtered.iloc[:,ycol])
            lm=linear_model.LinearRegression()
            # in case, a sklearn algorithm doesn't understand list or panda dataframe

        elif build_type=='classification':
            x_train, y_train = self.dummyProcess(x_filtered), self.classConvert(filtered.iloc[:,ycol], ycol)
            lm = linear_model.LogisticRegression()
            #lm = SVC(kernel="linear", C=0.025, random_state=101)

        elif build_type=='text classification':
            x_train, y_train = self.text2vec(x_filtered), self.classConvert(filtered.iloc[:,ycol], ycol)
            #lm = SVC(kernel="linear", C=0.025, random_state=101)
            lm = RandomForestClassifier(n_estimators=70, random_state = 101)

        elif build_type=='OCR':
            x_train, y_train = self.image2doc(x_filtered), self.classConvert(filtered.iloc[:,ycol], ycol)
            self.rawdata.iloc[:,self.ocrcol] = x_train
            x_train = self.text2vec(x_train)
            #lm = SVC(kernel="linear", C=0.025, random_state=101)
            lm = RandomForestClassifier(n_estimators=70, random_state = 101)
        else:
            return "Don't have this build type yet"
        try:
            print(pd.DataFrame(x_train).iloc[0:3,:])
        except:
            pass
        print(build_type,'train preprocess is done')
        return lm, x_train, y_train

    def image2doc(self,data):
        x_train, data =[], pd.DataFrame(data)
        for url in data.iloc[:,0]:
            urllib.request.urlretrieve(url, "test1.webp")
            Image.open('test1.webp').convert('RGB').save('new.jpeg')
            x_train.append( image_to_string(Image.open('new.jpeg')) )
        return x_train

    #without google's pre-trained model
    def text2vec(self,data):
        print("text2vec processing")
        '''
        data_list = data.values.tolist()
        print(data_list[0:2])
        x_train = ic.client_vec(data = "%s"%(data_list))
        try:
            x_train = eval(x_train) # vector case
        except:
            pass
        '''
        try:
            # text2vec(self,data):
            # use google pre-trained model for word to vector
            # preprocess raw data and convert text data to feature vector
            # it has one column
            data, docs = pd.DataFrame(data), []
            print(data)
            tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
            analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
            for i in range(data.shape[0]):
                try:
                    text = data.iloc[i,0].lower()
                except:
                    pass
                text = remove_stopwords(text)
                words = tokenizer.tokenize(text)
                tags = [i]
                docs.append(analyzedDocument(words, tags))
            vecmodel = doc2vec.Doc2Vec(docs, vector_size = 100, window = 300,
                                           min_count = 3, worker=10)
            x_train = [vecmodel.docvecs[i].tolist() for i in range(len(vecmodel.docvecs))]
            print(vecmodel.docvecs[i])
            print ('***Vecterization process finished')
            return pd.DataFrame(x_train)
        except:
            return "Vectorization process fail"

    def classConvert(self, raw_class, ycol):
        if ycol not in self.encode_d.keys():
            self.encode_d[ycol], self.decode_d[ycol] = {}, {}
        raw_class = raw_class.tolist()
        ori_value = list(set(raw_class))
        for i in range(len(ori_value)):
            self.encode_d[ycol][ori_value[i]], self.decode_d[ycol][i] = i, ori_value[i]
        label = [ self.encode_d[ycol][raw_class[i]] for i in range(len(raw_class))]
        y_train = pd.Series(label).reset_index(drop=True)
        return y_train

    def dummyProcess(self, filtered, ycol=-1, predict=False):
        # check feature has a categorical variable and normalize uncategorical variables
        if ycol > -1 :
            # for google sheet
            X = filtered.drop(filtered.columns[[ycol]], axis=1)
        else:
            #for postgreSQL
            X = pd.DataFrame(filtered)
        X.columns = [i for i in range(X.shape[1])]
        dummy_d ={}
        #check X has categorical variables
        for col in range(X.shape[1]):
            if isinstance(X.iloc[2,col], str):
                dummy_d[col]=pd.get_dummies(X.iloc[:,col], drop_first=True).reset_index(drop=True)

        X = X.drop(X.columns[list(dummy_d.keys())], axis=1).reset_index(drop=True)
        try:
            X = pd.DataFrame(preprocessing.scale(X)).reset_index(drop=True)
        except:
            pass
        for dummy in dummy_d:
            X = pd.concat([X, dummy_d[dummy]], axis=1).reset_index(drop=True)

        return X

    def predict(self, predict_raw, col=False, x_test=False, row=False):
        if col is False:
            # google sheet send a whole data
            predict_raw, col = self.find_googleSheet(predict_raw)
        # Test 1. In OCR and text classification case, error occur when predicted column is not label.
        try:
            predict_type, lm, mse = self.model[col]
            print("build type : ", predict_type)
        except:
            return "Unstructured column, Can't predict this column"
        
        if col == self.textcol or col == self.ocrcol:
            return "Unstructured column, Can't predict this column"
        
        if (self.src == 'google_sheet') or (self.src == 'postgreSQL'):
            # Test 2. preprocess error such as dimension doesn't match or mixed data type
            try:
                x_test = self.preprocess_predict(predict_raw, col, build_type = predict_type)
                print('test input:')
                print(x_test)
            except:
                return "Preprocess error, Check test data structure again"
        else:
            return "unsupported source"
        
        # Test 3. test data structure doesn't match train data structure
        try:
            if predict_type=='time':
                '''
                checker=False
                if col !=self.timecol:
                    for node in x_test:
                        if self.is_date(node):
                            checker = True
                    if not checker:
                        return "Test data type doesn't match trian data type"
                '''
                
                if sum(isinstance(i,str)*1 for i in x_test.iloc[0,:]) >0:
                    return "Can't predict this cell"
                
                y_predicted = lm.predict(x_test)
                y_predicted=y_predicted.astype('float')
                if col !=self.timecol:
                    return round(y_predicted[0][0],2)
                else:
                    longTime=str(pd.to_datetime(y_predicted[0][0]))
                    longTime = longTime.split(" ")
                    return longTime[0]
    
            elif predict_type=='regression':
                y_predicted = lm.predict(x_test)
                return (y_predicted[0]//0.01)/100
    
            elif (predict_type=='classification') or (predict_type=='text classification') or (predict_type=='OCR'):
                y_predicted = lm.predict(x_test)[0]
                return self.decode_d[col][int(y_predicted)]
            else:
                return "Can't predict this data type"
        except:
            return "test data structure should match train data structure"


    def find_googleSheet(self, filtered):
        # find which column we want to predict
        # when time series case it return t-3, t-2, t-1 values
        filtered = pd.DataFrame(filtered)
        for row in range(filtered.shape[0]):
            for col in range(filtered.shape[1]):
                if filtered.iloc[row,col] == "#ERROR!":
                    testRaw = filtered.iloc[row,:].tolist()
                    testRaw.pop(col)
                    '''
                    if len(self.headDrop) > 0 :
                        testRaw = [testRaw[i] for i in range(len(testRaw)) if i not in self.headDrop]
                    '''
                    if self.timecol >-1:
                        testRaw_time, j=[] , 1
                        while len(testRaw_time) <self.window:
                            if filtered.iloc[row-j,col] != "":
                               testRaw_time.append(filtered.iloc[row-j,col])
                            j+=1
                        testRaw_time.reverse()
                        testRaw = testRaw_time

                    return testRaw, col


    def testEmptyFilter(self, test_raw, ycol):
        # fill out the missing values in test data
        temp = self.rawdata.drop(self.rawdata.columns[[ycol]], axis=1)
        for i in range(len(test_raw)):
            if test_raw[i]=="":
                #check it is categorical
                if isinstance(temp.iloc[2,i], str):
                    test_raw[i] = temp.iloc[:,i].mode().tolist()[0]

                else:
                    test_raw[i] = round(temp.iloc[:,i].mean() ,2)
        return test_raw


    def preprocess_predict(self, xraw_test, ycol, row=False, build_type = False):
        if (self.src == 'postgreSQL') or (self.src == 'google_sheet'):
            if build_type=='time':
                if ycol == self.timecol:
                    filtered = pd.to_datetime(xraw_test)
                    xraw_test = filtered.values.astype(float)
                x_test = pd.DataFrame(xraw_test).T
                return x_test

            xraw_test = self.testEmptyFilter(xraw_test, ycol)
            if (build_type == 'regression') or (build_type == 'classification'):
                tmp = self.rawdata
                x_filtered = tmp.drop(tmp.columns[ycol], axis=1)
                x_filtered = x_filtered.append(pd.DataFrame(xraw_test, index=list(x_filtered.columns.values)).T )
                X= self.dummyProcess(x_filtered)
                x_test = pd.DataFrame(X.iloc[-1,:]).T

            elif build_type == 'text classification':

                xraw_test = self.rawdata.iloc[:,abs(1-ycol)].append(pd.Series(xraw_test))
                x_total_text = self.text2vec(xraw_test)
                x_test= pd.DataFrame(x_total_text.iloc[x_total_text.shape[0]-1,:]).T

            elif build_type == 'OCR':
                xrawImage = self.image2doc(xraw_test)

                test_raw = self.rawdata.iloc[:,abs(1-ycol)].append(pd.Series(xrawImage))
                x_total_text = self.text2vec(test_raw)
                x_test = pd.DataFrame(x_total_text.iloc[x_total_text.shape[0] - 1, :]).T

            else:
                return "build type error"

        else:
            return "this source is not ready"

        return x_test

    def anomaly(self, y_real, raw_data, col=False, row=False):
        if col is False:
            # google sheet send a whole data
            raw_data, col = self.find_googleSheet(raw_data)
        # Test 1. In OCR and text classification case, error occur when predicted column is not label.
        try:
            build_type, lm, mse = self.model[col]
            print("build type : ", build_type)
        except:
            return "Unstructured column, Can't detect anomaly this column"
        
        
        
        if (build_type == 'regression') or (build_type == 'time'):
            y_predicted = self.predict(raw_data, col=col)

            if self.timecol >-1 and col == self.timecol:
                y_real = DateTime(y_real).timeTime()
                # Test 3.  predict function return error message
                try:
                    y_predicted = DateTime(y_predicted).timeTime()
                    mse = mse/10**23
                except:
                    return "Prediction issue, "+y_predicted
            
            # Test 3. predict function return error message
            # predicted value for regression should be float
            if isinstance(y_predicted, str):
                return "Prediction issue, "+y_predicted
            
            if mse < 0.00001:
                temp = 10
            else:
                temp = abs((y_predicted - y_real)/mse)
            # anomaly score
            result = norm.cdf(abs(temp)) * 100.0
            return round(( (result -50)*2),2)

        elif build_type in ['classification','text classification','OCR']:
            # Test 4. preprocess error such as dimension doesn't match or mixed data type
            try:
                x_test = self.preprocess_predict(raw_data, col, row = row, build_type = build_type)
            except:
                return "Preprocess error, Check test data structure again"
            # Test 5. If real value is not in label set, return 100  
            if y_real in self.encode_d[col]:
                idx = self.encode_d[col][y_real]
                res = (1 - lm.predict_proba(x_test)[0][idx])*100
                if build_type =='text classification':
                    return round((res)/(len(self.encode_d[col].keys())-1 ),2)
                return round(res,2)
            else:
                return 100

        else:
            return'error'


    def findTimeLocation(self,test,ycol):
        data_time = self.rawdata.iloc[:,self.timecol].tolist()
        #test = pd.to_datetime(test[self.timecol])
        '''
        if ycol ==self.timecol:
            test.values.astype(float)
            time_values, new_feature = filtered.iloc[:,ycol].values.astype(float), {}
            y_train = pd.DataFrame(time_values[self.window:time_values.shape[self.timecol]])

            for i in range(self.window):
                new_feature[i] = time_values[i:time_values.shape[0]-(self.window-i)].tolist()
            x_train = pd.DataFrame(new_feature)
            lm=linear_model.LinearRegression()
        '''
        if test[0] > str(max(data_time)):
            return self.rawdata.iloc[self.rawdata.shape[0]-3:self.rawdata.shape[0],ycol]

        for i in range(len(data_time)):
            if str(data_time[i]) < test[0] < str(data_time[i+1]):
                return self.rawdata.iloc[i-2:i+1,ycol]


    def score(self):
        if self.src == 'google_sheet':
            if self.build_type in ['regression', 'classification', 'time', 'ME']:
                return self.mse
