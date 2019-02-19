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
import grpc
import preModel_pb2
import preModel_pb2_grpc

from multiprocessing import Process
#text classification
from gensim.models import doc2vec
from collections import namedtuple
from nltk.tokenize import RegexpTokenizer

from multiprocessing import Process
from gensim.parsing.preprocessing import remove_stopwords

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(preModel_pb2_grpc.GreeterServicer):

    def vectorize(self, request, context):
        tes=request.message
        print(tes)
        data = eval(tes)
        print ('***Vecterization process started')
        # text2vec(self,data):
        # use google pre-trained model for word to vector
        # preprocess raw data and convert text data to feature vector
        # it has one column
        data, docs = pd.DataFrame(data), []
        tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for i  in range(1, data.shape[0]):
            try:
                text = data.iloc[i,0].lower()
            except:
                pass
            text = remove_stopwords(text)
            words = tokenizer.tokenize(text)
            tags = [i]
            docs.append(analyzedDocument(words, tags))
        vecmodel = doc2vec.Doc2Vec(docs, vector_size = 50, window = 150,
                                    min_count = 3, worker=10)
        x_train = [vecmodel.docvecs[i].tolist() for i in range(len(vecmodel.docvecs))]
        print(vecmodel.docvecs[i])
        print ('***Vecterization process finished')
        return preModel_pb2.gReply( message=str(x_train) )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    preModel_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

print("Test g-vec server is running")

if __name__ == '__main__':
    p = Process(target=serve)
    p.start()
    p.join()
