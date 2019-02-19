import boto3
from boto3.s3.transfer import S3Transfer
import yaml
import ins_controller as ic

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
ACCESS_KEY = ENVIRONMENT["S3"]["aws_access_key_id"]
SECRET_KEY = ENVIRONMENT["S3"]["aws_secret_access_key"]
# Access to S3 on AWS
client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

transfer = S3Transfer(client)
bucket = ENVIRONMENT["S3"]["bucket_dev"]

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
