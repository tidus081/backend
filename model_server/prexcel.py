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

# time
from DateTime import DateTime
from dateutil.parser import parse

# OCR
from PIL import Image
from pytesseract import image_to_string
import urllib

# Vec
from gensim.models import doc2vec
from collections import namedtuple
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import remove_stopwords
from multiprocessing import Process

import ins_controller as ic
import collections


class Model(object):
    def __init__(self):
        self.model = {}
        self.encode_d, self.decode_d = {}, {}  # For classification label
        self.window, self.timecol = 3, -1  # For time series
        self.textcol, self.text_d = [], {}  # For text vector
        self.ocrcol = -1  # For OCR classification
        self.column_type_list = []

    def save_model(self, src, raw_data, window=False, factor=False):
        # Save build_type, lm, mse
        self.src = src
        self.header = [str(i) for i in raw_data.iloc[0, :]]
        self.rawdata = self.empty_filter(pd.DataFrame(raw_data))
        self.get_column_type_list(self.rawdata)
        self.model_filter()  # Run function and save model for each columns
        return self.model

    def empty_filter(self, data):
        # remove rows that contain empty values
        ginput = pd.DataFrame(data)
        # remove header
        ginput = ginput.drop(0)
        emptylist = []
        # remove all empty data
        for r in range(ginput.shape[0]):
            for c in range(ginput.shape[1]):
                if (ginput.iloc[r, c] == ""):
                    if isinstance(ginput.iloc[2, c], str) or len(set(ginput.iloc[:, c])) < 10:
                        ginput.iloc[r, c] = ginput.iloc[:, c].mode().tolist()[
                            0]
                        # if above or below cell is empty, doesn't work
                        while ginput.iloc[r, c] == "":
                            if r-1 > 0:
                                ginput.iloc[r, c] = ginput.iloc[r-1, c]
                            else:
                                ginput.iloc[r, c] = ginput.iloc[r+1, c]

                        print("fill string out:", ginput.iloc[r, c])
                    else:
                        #ginput.iloc[r, c] = np.median(ginput.iloc[:,c])
                        ginput.iloc[r, c] = 0
                        emptylist.append((r, c))
        for r, c in emptylist:
            ginput.iloc[r, c] = ginput.iloc[:, c].mean()
            print("fill numeric out:", ginput.iloc[r, c])
        return ginput

    def get_column_type_list(self, df):
        # expect dataframe
        type_list = []
        categorical_threshold = 0.1
        text_threshold = 0.8
        for c in range(df.shape[1]):
            type_count = collections.Counter()
            number_unique_c = len(set(df.iloc[:, c]))
            # If unique values is less than categorical_threshold, it is classification
            if number_unique_c == 1:
                type_list.append('only')

            elif number_unique_c < 15 or float(number_unique_c)/df.shape[1] < categorical_threshold:
                type_list.append('categorical')

            else:
                for r in range(df.shape[0]):
                    if self.is_date(df.iloc[r, c]):
                        type_count['time'] += 1
                    elif isinstance(df.iloc[r, c], int) or isinstance(df.iloc[r, c], float):
                        type_count['numeric'] += 1
                    elif self.is_text(df.iloc[r, c]) and float(number_unique_c)/df.shape[1] > text_threshold:
                        type_count['text'] += 1
                    else:
                        type_count['categorical'] += 1
                type_list.append(type_count.most_common(1)[0][0])  # col type
        if 'text' in type_list:
            self.textcol = [i for i, v in enumerate(type_list) if v == 'text']
        if 'time' in type_list:
            self.timecol = type_list.index('time')

        self.column_type_list = type_list
        print(self.column_type_list)

    def is_date(self, string):
        try:
            parse(string)
            return True
        except:
            return False

    def is_text(self, sample):
        try:
            if len(sample.split(" ")) > 3:
                return True
        except:
            pass
        return False

    def model_filter(self):
        """ Check each column's type

        1. Check OCR or time type is in the data
        2. Except that, All cases belong to classification or regression.
        3. We don't support to predict text
        """
        col_n = self.rawdata.shape[1]
        # Let's forget about OCR for a moment
        if 'OCR' in self.header:  # For time being, we should use header for OCR
            self.ocrcol = self.header.index('OCR')
            # Assume we have two columns for OCR
            class_col = abs(1-self.ocrcol)
            self.model[class_col] = ['OCR'] + \
                self.train(self.rawdata, class_col, column_type='OCR')

        elif self.timecol > -1:
            print("time modeling")
            for c in range(col_n):
                self.model[c] = ['time'] + \
                    self.train(self.rawdata, c, column_type='time')
        else:
            print("standard modeling")
            for c, column_type in enumerate(self.column_type_list):
                if column_type == 'only':
                    self.model[c] = [column_type] + \
                        [self.rawdata.iloc[0, c], 1]
                elif column_type != 'text':
                    self.model[c] = [column_type] + \
                        self.train(self.rawdata, c, column_type=column_type)
                else:
                    pass
        print(self.model.keys())

    def train(self, filtered, ycol, column_type=False):
        lm, x_train, y_train = self.preprocess_train(
            filtered, ycol, column_type=column_type)
        lm.fit(x_train, y_train)
        y_fit = lm.predict(x_train)

        if (column_type == 'numeric') or (column_type == 'time'):
            mse = mean_squared_error(y_train, y_fit)

        elif (column_type == 'categorical'):
            y_fit, res = y_fit.astype('int'), 0
            for i in range(len(y_fit)):
                if y_fit[i] != y_train[i]:
                    res += 1
            mse = res/len(y_train)
        else:
            mse = 1  # To avoid errors
        print(ycol, "train process done")
        return [lm, mse]

    def preprocess_train(self, filtered, ycol, column_type=False, lm=None):
        # train data contain x, y data
        print("preprocess start")
        if column_type == 'OCR':
            x_filtered = filtered.drop(filtered.columns[ycol], axis=1)
            x_train, y_train = self.image2doc(
                x_filtered), self.class_convert(filtered.iloc[:, ycol], ycol)
            self.rawdata.iloc[:, self.ocrcol] = x_train
            x_train = self.text2vec(x_train)
            #lm = SVC(kernel="linear", C=0.025, random_state=101)
            lm = RandomForestClassifier(n_estimators=70, random_state=101)

        elif column_type == 'time':
            filtered.iloc[:, self.timecol] = pd.to_datetime(
                filtered.iloc[:, self.timecol])
            filtered = filtered.sort_values(self.timecol)
            # check we are going to train time column
            if ycol != self.timecol:
                time_values, new_feature = filtered.iloc[:, ycol], {}
                y_train = pd.DataFrame(
                    time_values[self.window:time_values.shape[self.timecol]])
            else:
                time_values, new_feature = filtered.iloc[:, ycol].values.astype(float), {
                }
                y_train = pd.DataFrame(
                    time_values[self.window:time_values.shape[self.timecol]])

            for i in range(self.window):
                new_feature[i] = time_values[i:time_values.shape[0] -
                                             (self.window-i)].tolist()
            x_train = pd.DataFrame(new_feature)
            lm = linear_model.LinearRegression()

        else:
            x_train = self.feature_process(filtered, ycol)  # Return vectors
            if column_type == 'numeric':
                y_train = pd.Series(filtered.iloc[:, ycol])
                lm = linear_model.LinearRegression()

            elif column_type == 'categorical':
                y_train = self.class_convert(filtered.iloc[:, ycol], ycol)
                lm = linear_model.LogisticRegression()
                # lm = SVC(kernel="linear", C=0.025, random_state=101)
                # RandomForestClassifier(n_estimators=70, random_state = 101)

            else:
                return "Don't have this build type yet"
        try:
            print(x_train.iloc[0:3, :])
        except:
            pass

        print(column_type, 'train preprocess done')
        return lm, x_train, y_train

    def feature_process(self, X, ycol, predict=False):
        # Transform text to vector and categorical value to dummy variables
        if not predict:
            X = X.drop(X.columns[[ycol]], axis=1)
        else:
            pass

        feature_type_list = self.column_type_list[:ycol] + \
            self.column_type_list[ycol+1:]
        print("feature process:", feature_type_list)
        X.columns = [i for i in range(X.shape[1])]
        feature_dict = {}
        drop_list = []
        # Input pd.DataFrame
        for c, col_type in enumerate(feature_type_list):
            if col_type == 'text':
                print(X.iloc[:, c].values.tolist()[0:2])
                feature_dict[c] = pd.Series(self.text2vec(X.iloc[:, c].values.tolist()))
                drop_list.append(c)

            elif col_type == 'only':
                feature_dict[c] = pd.Series([1]*X.shape[0])
                drop_list.append(c)

            elif col_type == 'categorical':
                feature_dict[c] = pd.get_dummies(
                    X.iloc[:, c], drop_first=True).reset_index(drop=True)
                drop_list.append(c)
            else:
                pass

        X = X.drop(X.columns[drop_list], axis=1).reset_index(drop=True)
        for dummy_c in feature_dict:
            X = pd.concat([X, feature_dict[dummy_c]],
                          axis=1).reset_index(drop=True)
        print("feature_process done")
        return X

    def image2doc(self, data):
        x_train, data = [], pd.DataFrame(data)
        for url in data.iloc[:, 0]:
            urllib.request.urlretrieve(url, "test1.webp")
            Image.open('test1.webp').convert('RGB').save('new.jpeg')
            x_train.append(image_to_string(Image.open('new.jpeg')))
        return x_train

    # without google's pre-trained model
    def text2vec(self, data_list):
        try:
            # text2vec(self,data):
            # use google pre-trained model for word to vector
            # preprocess raw data and convert text data to feature vector
            # it has one column
            print(data_list[0:2])
            x_train = ic.client_vec(data="%s" % (data_list))
            x_train = eval(x_train)  # vector case
        except:
            return "Vectorization process fail"

    def class_convert(self, raw_class, ycol):
        if ycol not in self.encode_d.keys():
            self.encode_d[ycol], self.decode_d[ycol] = {}, {}
        raw_class = raw_class.tolist()
        ori_value = list(set(raw_class))
        for i in range(len(ori_value)):
            self.encode_d[ycol][ori_value[i]
                                ], self.decode_d[ycol][i] = i, ori_value[i]
        label = [self.encode_d[ycol][raw_class[i]]
                 for i in range(len(raw_class))]
        y_train = pd.Series(label).reset_index(drop=True)
        return y_train

    def predict(self, predict_raw, col=False, x_test=False, row=False):
        if (self.src == 'google_sheet') or (self.src == 'postgreSQL'):
            if col is False:
                # google sheet send a whole data
                predict_raw, col = self.find_google_sheet(predict_raw)

            # Test 1. In OCR and text classification case, error occur when predicted column is not label.
            if col in self.textcol or col == self.ocrcol:
                return "Unstructured column, Can't predict this column"

            try:
                predict_type, lm, mse = self.model[col]
                print("build type : ", predict_type)
            except:
                return "Unstructured column, Can't predict this column"

            if predict_type == 'only':  # Don't need preprocessing
                return lm  # only value in one type

            # Test 2. preprocess error such as dimension doesn't match or mixed data type
            x_test = self.preprocess_predict(
                predict_raw, col, build_type=predict_type)
            print('test input:')
            print(x_test.iloc[0:3, :])

        else:
            return "unsupported source"

        # Test 3. test data structure doesn't match train data structure
        if predict_type == 'time':
            if sum(isinstance(i, str)*1 for i in x_test.iloc[0, :]) > 0:
                return "Can't predict this cell"

            y_predicted = lm.predict(x_test)
            y_predicted = y_predicted.astype('float')
            if col != self.timecol:
                return round(y_predicted[0][0], 2)
            else:
                longTime = str(pd.to_datetime(y_predicted[0][0]))
                longTime = longTime.split(" ")
                return longTime[0]

        elif predict_type == 'numeric':
            y_predicted = lm.predict(x_test)
            return (y_predicted[0]//0.01)/100

        elif (predict_type == 'categorical') or (predict_type == 'OCR'):
            y_predicted = lm.predict(x_test)[0]
            return self.decode_d[col][int(y_predicted)]

    def find_google_sheet(self, filtered):
        # find which column we want to predict
        # when time series case it return t-3, t-2, t-1 values
        filtered = pd.DataFrame(filtered)
        for row in range(filtered.shape[0]):
            for col in range(filtered.shape[1]):
                if filtered.iloc[row, col] == "#ERROR!":
                    testRaw = filtered.iloc[row, :].tolist()
                    testRaw.pop(col)
                    if self.timecol > -1:
                        testRaw_time, j = [], 1
                        while len(testRaw_time) < self.window:
                            if filtered.iloc[row-j, col] != "":
                                testRaw_time.append(filtered.iloc[row-j, col])
                            j += 1
                        testRaw_time.reverse()
                        testRaw = testRaw_time

                    return testRaw, col

    def preprocess_predict(self, xraw_test, ycol, row=False, build_type=False):
        if (self.src == 'postgreSQL') or (self.src == 'google_sheet'):
            if build_type == 'time':
                if ycol == self.timecol:
                    filtered = pd.to_datetime(xraw_test)
                    xraw_test = filtered.values.astype(float)
                x_test = pd.DataFrame(xraw_test).T
                return x_test

            xraw_test = self.test_empty_filter(xraw_test, ycol)
            if (build_type == 'numeric') or (build_type == 'categorical'):
                tmp = self.rawdata
                x_filtered = tmp.drop(tmp.columns[ycol], axis=1)
                x_filtered = x_filtered.append(pd.DataFrame(
                    xraw_test, index=list(x_filtered.columns.values)).T)
                X = self.feature_process(x_filtered, ycol, predict=True)
                x_test = pd.DataFrame(X.iloc[-1, :]).T

            elif build_type == 'OCR':
                xrawImage = self.image2doc(xraw_test)
                test_raw = self.rawdata.iloc[:, abs(
                    1-ycol)].append(pd.Series(xrawImage))
                x_total_text = self.text2vec(test_raw, ycol)
                x_test = pd.DataFrame(
                    x_total_text.iloc[x_total_text.shape[0] - 1, :]).T

            else:
                return "build type error"

        else:
            return "this source is not ready"
        print("test preprocess done")
        return x_test

    def test_empty_filter(self, test_raw, ycol):
        # fill out the missing values in test data
        temp = self.rawdata.drop(self.rawdata.columns[[ycol]], axis=1)
        for i in range(len(test_raw)):
            if test_raw[i] == "":
                # check it is categorical
                if isinstance(temp.iloc[2, i], str):
                    test_raw[i] = temp.iloc[:, i].mode().tolist()[0]

                else:
                    test_raw[i] = round(temp.iloc[:, i].mean(), 2)
        return test_raw

    def anomaly(self, y_real, raw_data, col=False, row=False):
        if col is False:
            # google sheet send a whole data
            raw_data, col = self.find_google_sheet(raw_data)
        # Test 1. In OCR and text classification case, error occur when predicted column is not label.
        if col in self.textcol or col == self.ocrcol:
            return "Unstructured column, Can't get anomaly score on this column"
        try:
            build_type, lm, mse = self.model[col]
            print("build type : ", build_type)
        except:
            return "Unstructured column, Can't detect anomaly on this column"

        if build_type == 'one':
            if y_real == lm:
                return 0
            else:
                return 100

        elif (build_type == 'numeric') or (build_type == 'time'):
            y_predicted = self.predict(raw_data, col=col)

            if self.timecol > -1 and col == self.timecol:
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
            return round(((result - 50)*2), 2)

        elif build_type in ['categorical', 'OCR']:
            # Test 4. preprocess error such as dimension doesn't match or mixed data type
            try:
                x_test = self.preprocess_predict(
                    raw_data, col, row=row, build_type=build_type)
            except:
                return "Preprocess error, Check test data structure again"
            # Test 5. If real value is not in label set, return 100
            if y_real in self.encode_d[col]:
                idx = self.encode_d[col][y_real]
                res = (1 - lm.predict_proba(x_test)[0][idx])*100
                if len(self.encode_d[col].keys()) > 2:
                    res = (res) / (len(self.encode_d[col].keys()) - 1)
                return round(res, 2)
            else:
                return 100

        else:
            return'error'

    def findTimeLocation(self, test, ycol):
        data_time = self.rawdata.iloc[:, self.timecol].tolist()
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
            return self.rawdata.iloc[self.rawdata.shape[0]-3:self.rawdata.shape[0], ycol]

        for i in range(len(data_time)):
            if str(data_time[i]) < test[0] < str(data_time[i+1]):
                return self.rawdata.iloc[i-2:i+1, ycol]
