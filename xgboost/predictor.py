# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd
import xgboost

from openpyxl import load_workbook
import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import shutil
from PIL import Image
import time
import argparse
from s3_utils import S3Utils
import matplotlib.pyplot as plt

RETRIEVING_MODEL_BATCH_SIZE = 64
CUDA = False

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

import configs


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ImageReader(Dataset):
    def __init__(self, images):
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), normalize
        ])
        self.images = images

    def __getitem__(self, index):
        img = self.images[index]
        img = self.transform(img)
        return img, 1

    def __len__(self):
        return len(self.images)


def get_feature_vectors(net, images_list):
    test_data_set = ImageReader(images_list)
    test_data_loader = DataLoader(test_data_set, 1, shuffle=False, num_workers=1)
    data_dict = {'test': {'data_loader': test_data_loader}}

    net.eval()
    with torch.no_grad():
        for key in data_dict.keys():
            data_dict[key]['features'] = []
            for inputs, labels in data_dict[key]['data_loader']:
                if CUDA:
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs, labels
                features, classes = net(inputs)
                data_dict[key]['features'].append(features)
            data_dict[key]['features'] = torch.cat(data_dict[key]['features'], dim=0)

    return data_dict


def plot_retrievals(images):
    """Images: list of images (query image at index = 0, and the retrievals)"""
    fig = plt.figure(figsize=(20, 10))
    columns = len(images)
    rows = 1

    plots_path = 'plots/'
    if os.path.exists(plots_path):
        shutil.rmtree(plots_path)
    os.makedirs(plots_path)

    fig_name = 'plots'
    for i in range(1, len(images) + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.savefig(plots_path + fig_name + '.png')


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            # with open(os.path.join(model_path, 'weights.pt'), 'rb') as inp:
            #     cls.model = torch.load(inp, map_location=torch.device('cpu'))
            cls.model = torch.load(os.path.join(model_path, 'weights.pt'), map_location=torch.device('cpu'))
        return cls.model

    @classmethod
    def predict_original(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)

    @classmethod
    def predict(cls, input_img):
        """input: is the directory of the saved file from the request TODO change it later"""
        logo_img = Image.open(input_img).convert('RGB')
        print("Apply inference on the input image ...")
        query_features = get_feature_vectors(cls.get_model(), [logo_img])

        print("loading database features ...")
        data_base = torch.load(configs.ENCODED_DB_DIR)
        gallery_images = data_base['db_images']
        gallery_features = data_base['db_features']

        query_feature = query_features['test']['features'][0]

        print("searching for the image in the encoded database ...")
        dist_matrix = torch.cdist(query_feature.unsqueeze(0).unsqueeze(0), gallery_features.unsqueeze(0)).squeeze()

        idx = dist_matrix.topk(k=configs.RETRIEVAL_NUM, dim=-1, largest=False)[1]

        similar_images = []
        similar_images_names = []
        distances = []

        for num, index in enumerate(idx):
            similar_image_name = gallery_images[index.item()]
            similar_images_names.append(similar_image_name)
            retrieval_image = Image.fromarray(S3Utils('image-uspto').im_read_from_bucket(similar_image_name)).convert(
                'RGB').resize((224, 224), resample=Image.BILINEAR)
            similar_images.append(retrieval_image)

            distances.append(dist_matrix[index.item()].item())

        if distances[0] < 0.001:
            matched_img = similar_images_names[0].split('.')[0]
            similar_imgs = [name.split('.')[0] for name in similar_images_names]
            return {'is_match': True, 'similar_imgs': similar_imgs}
        else:
            matched_img = None
            similar_imgs = [name.split('.')[0] for name in similar_images_names]
            return {'is_match': False, 'similar_imgs': similar_imgs}


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # 
    if flask.request.content_type == 'file':
        input_json = flask.request.get_json()
        # todo
        # the request to the api should look like:
        # 1.) image_file(file);
        # 2.) max_similar_results(integer);
        # 3.) match_likeness_percentage(integer / double);

        image = flask.request.files['image_file']
        image.save('input.jpg')  # TODO YOU NEED TO UPDATE THIS IN CASE OF PNG OR OTHER FORMATS
        # image = flask.request.data.decode('utf-8')
        # s = StringIO(data)
        # data = pd.read_csv(s, header=None)
        # data = data.ix[:, 1:]
    else:
        return flask.Response(response='This predictor supports only images as files', status=415, mimetype='text/plain')

    # print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict('input.jpg')

    # Convert from numpy back to CSV
    # out = StringIO()
    # pd.DataFrame({'results': predictions}).to_csv(out, header=False, index=False)
    # result = out.getvalue()

    # todo
    # the response format should be look like
    # {
    #     is_match: True or False,
    #     similar_imgs: [img_id1, img_id2, img_id3, ….]
    # }

    return flask.Response(predictions, status=200, mimetype='text/csv')
