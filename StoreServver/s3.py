import boto3
from boto3.s3.transfer import S3Transfer

#Access S3 on AWS
client = boto3.client('s3')
transfer = S3Transfer(client)
bucket = "ikigai-demo"

#Find file name and folder path
def filter(user, pname, sname, extension):
    if extension == 'csv':
        fname = "%s_%s_%s.csv"%(user, pname, sname)
        s3folder = "/%s/%s/%s/data/"%(user, pname, sname)
    else:
        fname = "%s_%s_%s"%(user, pname, sname)
        s3folder = "/%s/%s/%s/model/"%(user, pname, sname)
    return fname, s3folder

#Upload file from EC2 to S3
def ins2s3(user, pname ,sname, extension):
    fname, s3folder = filter(user, pname, sname, extension)
    filepath = "/home/ubuntu/store/" + fname
    s3path = s3folder+fname
    transfer.upload_file(filepath, bucket, s3path)

#Download file from S3 to EC2
def s32ins(user, pname, sname, extension):
    fname, s3folder = filter(user, pname, sname, extension)
    s3path = s3folder+fname
    transfer.download_file(bucket, s3path, fname)

#Download ikigai user list from S3 to EC2
def s32auth():
    transfer.download_file(bucket, "/ikigai_auth/ikigai_userList.csv", "ikigai_userList.csv")
