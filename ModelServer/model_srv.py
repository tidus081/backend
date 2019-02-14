# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC gPrexcel.Greeter server."""
from concurrent import futures
import time
import pandas as pd
import prexcel
import grpc
import preModel_pb2
import preModel_pb2_grpc
import ins_controller as ic
import pickle
import os
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(preModel_pb2_grpc.GreeterServicer):

    def build_model(self, request, context):
        #Parameters don't return errors. At least there are None.
        src, pname, sname, user = request.src, request.pname, request.sname, request.user

        #Get data from store instance
        print("******get data")
        #Test1. csv file doesn't exist
        try:
            ic.client_store(type="download", src=src, pname=pname, sname=sname, user=user, extension='csv')
        except:
            return preModel_pb2.gReply(message="update data first before build model")
        
        #Test2. Loading csv file fails or building data frame(such as mixed float, str) error occur
        try:
            fname = "%s_%s_%s.csv"%(user, pname, sname)
            res = pd.read_csv(fname)
            os.remove(fname)
            res.fillna("", inplace=True)
            header = pd.DataFrame(list(res)).T
            res.columns = [i for i in range(res.shape[1])]
            res = pd.concat([header, res], ignore_index=True)
            if res.shape[0] > 2:
                print(res.iloc[0:3,:])
        except:
            return preModel_pb2.gReply(message="unstructured data")

        #Test3. Error occurs in prexcel script 
        try:
            prx_model = prexcel.Model() #Call class in prexcel
            prx_model.save_model(src, res) #Prexcel train a model and save it as an object
        except:
            return preModel_pb2.gReply(message="Can't build a model")
        
        #Send object to store instance
        print("******send object")
        try:
            fname = "%s_%s_%s"%(user, pname, sname)
            #Test4. Uploading object fails
            with open(fname, "wb") as f:
                pickle.dump(prx_model, f)
                f.close()
            ic.client_store(type="pass_data", src=src, pname=pname, sname=sname, user=user, extension='object') #Send parameters
            resData = ic.client_store(type="upload", pname=pname, sname=sname, user=user, extension='object') # Send object file
            print("Client store-upload : ", resData)
        except:
            return preModel_pb2.gReply( message="can't upload object" )

        print ('***build_model process finished')
        ic.delete_file(fname)
        return preModel_pb2.gReply( message="Build model")

    def predict(self, request, context):
        print ('***Predict process started')
        #Test1. Receiving data from flask instance fails
        try:
            src, user, pname, sname, data = request.src, request.user, request.pname, request.sname, eval(request.data)
        except:
            return preModel_pb2.gReply(message="Internal network problem")
       
        #Load obejct from store instance
        #Test 2. Downloading object file fail 
        #If error occured when update data&build model or didn't click update data of build model
        fname = "%s_%s_%s"%(user, pname, sname)
        try:
            ic.client_store(type="download", src=src, pname=pname, sname=sname, user=user, extension='object')
            with open(fname, "rb") as f:
                prediction_model = pickle.load(f)
                f.close()
            os.remove(fname)
        except:
            return preModel_pb2.gReply(message="Check data structure and Update data&Build model again")
        
        #Prexcel return right result or error message
        #Insert try&except in predict function in Prexcel script
        try:
            result = prediction_model.predict(data)
        except:
            result = "can't predict value"
        print(result)
        print ('***Predict process finished')
        ic.delete_file(fname)
        return preModel_pb2.gReply(message=str(result))

    def anomaly(self, request, context):
        print ('***Anomaly process started')
        #Test1. Receiving data from flask instance fail
        try:
            src,user,pname,sname,data= request.src,request.user,request.pname,request.sname,eval(request.data)
        except:
            return preModel_pb2.gReply( message="Internal network problem" )
        
        try:
            real = float(request.real)
        except:
            #At least request.real is None
            real = str(request.real)
            
        #load obejct from store instance
        #Test2. Downloading object file fails
        #If error occured when update data&build model or didn't click update data of build model
        fname = "%s_%s_%s"%(user, pname, sname)
        try:
            ic.client_store(type="download", src=src, pname=pname, sname=sname, user=user, extension='object')
            with open(fname, "rb") as f:
                prx_model = pickle.load(f)
                f.close()
        except:
            return preModel_pb2.gReply(message="Check data structure and Update data&Build model again")
               
        #Prexcel return right result or error message
        #Insert try&except in predict function in Prexcel script
        try:
            result = prx_model.anomaly(real,data)
        except:
            result = 100
        print ('***Anomaly process finished')
        ic.delete_file(fname)
        return preModel_pb2.gReply(message=str(result))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    preModel_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

print("g-Model server is running")

if __name__ == '__main__':
    serve()
