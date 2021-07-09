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
model = torch.load('assets/weights.pt', map_location=torch.device('cpu'))


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
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.savefig(plots_path + fig_name + '.png')


def search(logo_img, encoded_db_dir, retrieval_num=1, net=model):
    logo_img = Image.open(logo_img).convert('RGB')
    print("Apply inference on the input image ...")
    query_features = get_feature_vectors(net, [logo_img])

    print("loading database features ...")
    data_base = torch.load(encoded_db_dir)
    gallery_images = data_base['db_images']
    gallery_features = data_base['db_features']

    query_feature = query_features['test']['features'][0]

    print("searching for the image in the encoded database ...")
    dist_matrix = torch.cdist(query_feature.unsqueeze(0).unsqueeze(0), gallery_features.unsqueeze(0)).squeeze()

    idx = dist_matrix.topk(k=retrieval_num, dim=-1, largest=False)[1]

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
        print(distances)

    if distances[0] < 0.001:
        print("\nTrade Mark Existed! - serial number: {}".format(similar_images_names[0].split('.')[0]))
        print('\nretrieving and plotting similar trade marks ...')
        plot_retrievals(similar_images)
        print("\nSerial Numbers of similar images: {}".format([name.split('.')[0] for name in similar_images_names]))
    else:
        print("\nTrade Mark Not Existed!")
        print('\nretrieving and plotting similar trade marks ...')
        plot_retrievals(similar_images)
        print("\nSerial Numbers of similar images: {}".format([name.split('.')[0] for name in similar_images_names]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing wipo search pipeline')
    parser.add_argument('--input-img', default='', type=str, help='input image directory directory')
    parser.add_argument('--retrievals-num', default='', type=int,
                        help='number of most similar images to the input images you want to query')
    parser.add_argument('--encoded-db-dir', default='assets/db.pth', help='')

    args = parser.parse_args()

    search(args.input_img, args.encoded_db_dir, args.retrievals_num, net=model)
