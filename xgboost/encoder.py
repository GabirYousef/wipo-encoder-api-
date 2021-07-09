import os
import tempfile

from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
import boto3
from s3_utils import S3Utils
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CUDA = False
batch_size = 64


model = torch.load('assets/weights.pt', map_location=torch.device('cpu'))


class ImageReader():
    def __init__(self, data_dict):

        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket('image-uspto')
        
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), normalize
        ])
        self.images, self.labels = [], []
        for label, image_list in data_dict.items():
            self.images += image_list
            self.labels += [self.class_to_idx[label]] * len(image_list)

    def __getitem__(self, index):
        img_name, target = self.images[index], self.labels[index]
#         print("img_name: ", img_name)
#         print(target)
#         img_name = path.split('/')[-1]
#         # img = S3Utils('image-uspto').load_img_directly(path + '.png').convert('RGB')
        
        img = Image.open("dataset/correct_data/" + img_name).convert('RGB')

#         # we need to download the file from S3 to a temporary file locally
#         # we need to create the local file name
#         obj = self.bucket.Object(img_name)
#         tmp = tempfile.NamedTemporaryFile()
#         tmp_name = tmp.name

#         # now we can actually download from S3 to a local place
#         with open(tmp_name, 'wb') as f:
#             obj.download_fileobj(f)
#             f.flush()
#             f.close()
#             img = Image.open(tmp_name).convert('RGB')

# #         if self.transform:
# #             image = self.transform(image)

        
            
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)




class ImageDataset(Dataset):
    def __init__(self, path='image-uspto', transform=None):
        self.path = path
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(path)
        # self.files = [obj.key for obj in self.bucket.objects.all()]
        self.files = self.get_files()

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((124, 124)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        # we may infer the label from the filename
        dash_idx = img_name.rfind('-')
        dot_idx = img_name.rfind('.')
        label = int(img_name[dash_idx + 1:dot_idx])

        # we need to download the file from S3 to a temporary file locally
        # we need to create the local file name
        obj = self.bucket.Object(img_name)
        tmp = tempfile.NamedTemporaryFile()
        tmp_name = tmp.name

        # now we can actually download from S3 to a local place
        with open(tmp_name, 'wb') as f:
            obj.download_fileobj(f)
            f.flush()
            f.close()
            image = Image.open(tmp_name)

        if self.transform:
            image = self.transform(image)

        print(image.size)
        print(label)
            
        return image, label

    def get_files(self, txt_file='dataset/img_names.txt'):
        with open(txt_file) as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        files = lines[:128]
        return files


def process_data(annotation_file, count):
    db_images = {}

    # annotation_file = os.path.join(data_path, annotation_file)
    with open(annotation_file) as f:
        annotations = f.readlines()
    annotations = [line.strip() for line in annotations]

    for frame_info in annotations:

        img_name, img_label = frame_info.split(",", 1)[0], frame_info.split(",", 1)[0]
        # img_dir = os.path.join(data_path, images_folder, img_name)

        if img_label in db_images:
            db_images[img_label].append(img_name)
        else:
            db_images[img_label] = [img_name]

    torch.save({'db': db_images}, 'dicts/db_dict_{}.pth'.format(count))
    return db_images


def get_feature_vectors(net, data_dict):   
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in data_dict.keys():
            data_dict[key]['features'] = []
            for inputs, labels in tqdm(data_dict[key]['data_loader'], desc='processing {} data'.format(key)):
#             for inputs, labels in data_dict[key]['data_loader']:

                if CUDA:
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs, labels
                features, classes = net(inputs)
                data_dict[key]['features'].append(features)
            data_dict[key]['features'] = torch.cat(data_dict[key]['features'], dim=0)

    return data_dict


def create_database(data_dict, count):

    data_base = {}

    test_data_set = ImageReader(data_dict)
#     test_data_set = ImageDataset(path='image-uspto')
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=4)
    data_dict = {'db': {'data_loader': test_data_loader}}

    data_dict = get_feature_vectors(model, data_dict)

    data_base['db_images'] = test_data_set.images
    data_base['db_labels'] = test_data_set.labels
    data_base['db_features'] = data_dict['db']['features']

    torch.save(data_base, 'features/db_features_{}.pth'.format(count))


if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Test CGD')
#     parser.add_argument('--txtfile', default='', type=str,
#                         help='query image name')
#     parser.add_argument('--count', default='',
#                         type=str, help='queried database')


#     opt = parser.parse_args()
    done = os.listdir('features')
    done = [d.split("_")[2].split(".")[0] + ".txt" for d in done]
    for txt in os.listdir("txt_files/"):
        if txt in done:
            continue
        count = txt.split('.')[0]
        print(count)
        txt_file = "txt_files/" + txt
        data_dict = process_data(txt_file, count)

        create_database(data_dict, count)
