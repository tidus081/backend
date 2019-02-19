# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""grpc controller sends proto request to each EC2 instances"""
from concurrent import futures
import time
import pandas as pd
import os
import os.path
import grpc
import preModel_pb2
import preModel_pb2_grpc
import csv
import yaml

import s3

def load_environment(path):
    """

    Load environment file from yaml file into python dictionary

    Args:
        path (str): Relative Path for environment.yaml file

    Returns:
        (dict): environment in python dict format

    """
    with open(path, 'r') as f:
        return yaml.load(f)

ENVIRONMENT = load_environment("environment.yaml")

STORE_IP = ENVIRONMENT["URL_dev"]["store"]
MODEL_IP = ENVIRONMENT["URL_dev"]["model"]
VEC_IP = ENVIRONMENT["URL_dev"]["vec"]

def delete_file(fname):
    if os.path.isfile(fname):
        os.remove(fname)
    else:
        pass
    return

# Check whether an user email is on the ikigai's userlist
def auth(email):
    delete_file('ikigai_userList.csv') 
    s3.s32auth() # check list file comes from S3 and dowonload it in EC2
    user_list = pd.read_csv('ikigai_userList.csv')
    print(user_list)
    if email in list(user_list.iloc[:,1]):
        return True
    else:
        return False
    
    return True
        
# Convert file to response to send proto request
def file2respon(fname):
    CHUNK_SIZE = 1024 * 1024  # 1MB
    with open(fname, 'rb') as f:
        delete_file(fname)
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield preModel_pb2.fileReply(buffer=piece)
        f.close()
    
# Convert response to file to receive proto request
def respon2file(response, fname):
    with open(fname, 'wb') as f:
        for resp in response:
            f.write(resp.buffer)
        f.close()

# Send proto request to a store instance
def client_store(type=None, src=None, pname=None, sname=None, user=None, data=None, extension=None):
    try:
        channel = grpc.insecure_channel(STORE_IP)
        stub = preModel_pb2_grpc.GreeterStub(channel)
    except:
        return "gRPC Connection Error"

    if type == 'upload':
        # data from flask ins, model from model ins
        fname = "%s_%s_%s"%(user,pname,sname)
        try:
            respon_generator = file2respon(fname)
            response = stub.upload_object(respon_generator)
        except:
            return "Upload Error"
        return response.message

    elif type =='pass_data':
        try:
            response = stub.pass_data(preModel_pb2.gRequest(src=src, pname=pname, sname=sname, user=user, data=data, extension=extension))
        except:
            return "can't reach to store instance"
        return response.message

    elif type == 'download':
        # model, predict, anomaly from model ins
        if extension == 'csv':
            fname ="%s_%s_%s.csv"%(user,pname,sname)
            response = stub.download_csv(preModel_pb2.gRequest(src=src, pname=pname, sname=sname, user=user))
        elif extension == 'object':
            fname ="%s_%s_%s"%(user,pname,sname)
            response = stub.download_object(preModel_pb2.gRequest(src=src, pname=pname, sname=sname, user=user))
        else:
            return "extension error"

        if os.path.isfile(fname):
            os.remove(fname)
        return respon2file(response, fname)

    else:
        return "store error"

# Send proto request to a model instance
def client_model(type=None, src=None, pname=None, sname=None, user=None, data=None, real=None):
    try:
        channel = grpc.insecure_channel(MODEL_IP)
        stub = preModel_pb2_grpc.GreeterStub(channel)
    except:
        return "gRPC Connection Error"
    if type == 'model':
        response = stub.build_model(preModel_pb2.gRequest(src=src,pname=pname,sname=sname,user=user))

    elif type == 'predict':
        response = stub.predict(preModel_pb2.gRequest(src=src,pname=pname,sname=sname, user=user, data=data))

    elif type == 'anomaly':
        response = stub.anomaly(preModel_pb2.gRequest(src=src,pname=pname,sname=sname, user=user, data=data, real = real))

    else:
        return "model error"

    return response.message

# Send proto request to a vec instance
def client_vec(data=None):
    try:
        channel = grpc.insecure_channel(VEC_IP)
        stub = preModel_pb2_grpc.GreeterStub(channel)
    except:
        return "gRPC Connection Error"
    response = stub.vectorize(preModel_pb2.gReply(message=data))
    return response.message
