import pandas as pd
import os
import yaml
from concurrent import futures
import time
import grpc

import preModel_pb2, preModel_pb2_grpc
import ins_controller as ic
import s3

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

ENVIRONMENT = ic.load_environment("environment.yaml")

TMP_CSV_LOCATION = ENVIRONMENT["temporary_location"]["csv"]

class Greeter(preModel_pb2_grpc.GreeterServicer):
    #When we upload object, we need user, file and sheet name
    #Pass_data send those parameter and self. variables are used in upload_object
    def __init__(self):   
        self.pname = self.sname = self.user = self.src = self.extension = None

    #Store instance receive data if extension is csv and receive parameters
    def pass_data(self, request, context):
        #Parameters don't return errors. At least there are None.
        self.src, self.user, self.pname, self.sname = request.src, request.user, request.pname, request.sname
        if request.extension == 'csv':
            try:
                data = eval(request.data)
                fname = "%s_%s_%s.csv"%(self.user, self.pname, self.sname)
                print(fname)
                ic.delete_file(fname) #To avoid errors
                temp = pd.DataFrame(data)
                temp.fillna("", inplace=True)
                temp.to_csv(fname, sep=',', index=False, header=False)
                s3.ins2s3(self.user, self.pname, self.sname, request.extension)
                os.remove(fname)
                print("******%s saved"%(fname))
            except:
                return preModel_pb2.gReply(message="Can't save the data, Check data structure")
        elif request.extension == 'object':
            pass
        else:
            return preModel_pb2.gReply(message="Extension error")
        return preModel_pb2.gReply(message="Saved current data")

    #After build model, model instance ask upload_object request to store instance
    def upload_object(self, request, context):
        fname = "%s_%s_%s"%(self.user, self.pname, self.sname) #Already assign variables in pass_data
        ic.delete_file(fname)
        #Test1. connection error from model instance
        try:    
            ic.respon2file(request, fname) #Convert response to file. It downloads file. 
        except:
            return preModel_pb2.gReply(message="didn't receive object")
        
        #Test2. s3 connection problem
        try:
            s3.ins2s3(self.user, self.pname, self.sname, 'object') #upload object from instance to S3
        except:
            return preModel_pb2.gReply(message="Internal storage connection error")
        
        ic.delete_file(fname)
        print("******%s saved"%(fname))
        return preModel_pb2.gReply(message="%s saved"%fname)

    #download csv file from S3. It's used in build_model in model instance
    def download_csv(self, request, context):
        print('loading csv')
        pname, sname, user = request.pname, request.sname, request.user
        fname = "%s_%s_%s.csv"%(user,pname,sname)
        s3.s32ins(user,pname,sname, 'csv')
        print("******%s loaded"%(fname))
        return ic.file2respon(fname)

    #download csv file from S3. It's used in model instance
    def download_object(self, request, context):
        print('loading object')
        pname, sname, user = request.pname, request.sname, request.user
        fname = "%s_%s_%s"%(user,pname,sname)
        s3.s32ins(user,pname,sname,'object')
        print("******%s loaded"%(fname))
        return ic.file2respon(fname)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    preModel_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

print("Test GRPc-store server is running")

if __name__ == '__main__':
    serve()
