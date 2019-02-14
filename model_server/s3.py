import boto3
from boto3.s3.transfer import S3Transfer

# ins2s3("robbie", "csv")
ACCESS_KEY = 'AKIAIIX7TJ24TJ7PGFLQ'
SECRET_KEY = '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'
client = boto3.client('s3',aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY )
transfer = S3Transfer(client)
bucket = "ikigai-demo"

def filter(user,pname,sname, extension):
    if extension == 'csv':
        fname = "%s_%s_%s.csv"%(user,pname,sname)
        s3folder = "/%s/%s/%s/data/"%(user,pname,sname)
    else:
        fname = "%s_%s_%s"%(user,pname,sname)
        s3folder = "/%s/%s/%s/model/"%(user,pname,sname)
        
    return fname, s3folder

def ins2s3(user,pname,sname, extension):
    fname, s3folder = filter(user,pname,sname, extension)
    filepath = "/home/ubuntu/store/" + fname
    #filepath = "/home/robbie/git/test-grpc/" + fname
    s3path = s3folder+fname
    transfer.upload_file(filepath, bucket, s3path)

def s32ins(user,pname,sname, extension):
    fname, s3folder = filter(user,pname,sname, extension)
    s3path = s3folder+fname
    transfer.download_file(bucket, s3path, fname)

def s32auth():
    transfer.download_file(bucket, "/ikigai_auth/ikigai_userList.csv", "ikigai_userList.csv")
