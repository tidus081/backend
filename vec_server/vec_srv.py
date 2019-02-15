"""

Dev vectorization server

Copyright 2019, Ikigai Labs.
@author: Robbie

"""
from concurrent import futures
import time
import pandas as pd

#text classification
from gensim.models import doc2vec
from collections import namedtuple
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import remove_stopwords
from multiprocessing import Process

import grpc
from utils import preModel_pb2, preModel_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(preModel_pb2_grpc.GreeterServicer):

    def vectorize(self, request, context):
        tes = request.message
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
        for i in range(data.shape[0]):
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
        x_train = [vecmodel.docvecs[i].tolist() for i in range(data.shape[0])]
        return preModel_pb2.gReply(message=str(x_train))

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

print("g-vec server is running")

if __name__ == '__main__':
    p = Process(target=serve)
    p.start()
    p.join()
