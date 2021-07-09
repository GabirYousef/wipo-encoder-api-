import boto3
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import s3fs

fs = s3fs.S3FileSystem()

class S3Utils(object):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        
    def list_bucket_files(self, end):
        count = 0
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(self.bucket_name)
        for my_bucket_object in my_bucket.objects.all():
            print(my_bucket_object.key)
            count += 1
            if count == end:
                break
                
    def im_read_from_bucket(self, file_keys_in_bucket):
        s3 = boto3.resource('s3')
        obj = s3.Object(self.bucket_name, file_keys_in_bucket.strip())  # get object
        body = obj.get()['Body'].read()                 # read binary stream
        arr = np.fromstring(body, dtype=np.uint8)    # convert binary stream to uint8
        img = cv2.imdecode(arr, 1)           # decode to image array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def download_img(self, img_name, out_dir):
        s3_client = boto3.client('s3')
        s3_client.download_file(self.bucket_name, img_name, out_dir)
        
    def load_img_directly(self, img_name):
        with fs.open('s3://{}/{}'.format(self.bucket_name, img_name)) as f:
            img = Image.open(f)
        return img
    
    def display_img(self, pillow_img):
        display(pillow_img)
        
        
    def write_image_to_s3(img_array, bucket, key, region_name='us-west-2'):
        s3 = boto3.resource('s3', region_name)
        bucket = s3.Bucket(bucket)
        object = bucket.Object(key)
        file_stream = BytesIO()
        im = Image.fromarray(img_array)
        im.save(file_stream, format='jpeg')
        object.put(Body=file_stream.getvalue())
        

# if __name__ == '__main__':
#     S = S3Utils('image-uspto')
#     S.download_img('85932339.png')
#     img = S.load_img_directly('85932339.png')