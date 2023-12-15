#  -*- coding: utf-8 -*-
#  Author : Subin Lee
#  e-mail : subin.lee@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.
'''
테스트 코드
'''
import random
import pandas as pd
import numpy as np
import os
import glob
import cv2
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

# params 정의
params = {
    'IMG_SIZE': 224,
    'BATCH_SIZE':32,
    'SEED':2023
}

# random seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(params['SEED'])

# test 이미지 경로 설정
path = './test_images/'
os.chdir(path)

# device 정의
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


test_img_list = glob.glob('./*')
test_df = pd.DataFrame(columns=['img_path','label'])
test_df['img_path'] = test_img_list
test_df['label'] = test_df['img_path'].apply(lambda x:str(x).split('_')[2]) # label 생성


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)


test_transform = A.Compose([
    A.Resize(params['IMG_SIZE'],params['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
])
test_dataset = CustomDataset(test_df['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=params['BATCH_SIZE'], shuffle=False, num_workers=0)


class BaseModel(nn.Module):
    def __init__(self, num_classes=5):
        super(BaseModel, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0')
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.sigmoid(self.classifier(x))
        return x


# model 저장 경로 설정 및 load
model = torch.load('../best_model.pt', map_location=device)


# test
def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            pred = model(imgs)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    return preds


preds = inference(model, test_loader, device)

# label 변환
label_dict = {0:'C1',
              1:'G1',
              2:'H1',
              3:'I1',
              4:'T1'}
test_preds = [label_dict[label] for label in preds]

print(classification_report(test_df['label'], test_preds))

