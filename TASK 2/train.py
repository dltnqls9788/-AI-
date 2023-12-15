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
os.chdir("./S63_DATA1/")

# params 정의
params = {
    'IMG_SIZE': 224,
    'EPOCHS': 20,
    'LEARNING_RATE':3e-4,
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

# device 정의
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


### 데이터 변환 ###
class DataProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def process_files(self):
        file_list = os.listdir(self.folder_path)

        image_files = [f for f in file_list if f.endswith(".jpg")]
        image_files.sort()

        train_image_files = image_files[:4800]
        # valid와 test는 shuffle 후 분리
        shuffle_images = image_files[4800:]
        random.shuffle(shuffle_images)
        valid_image_files = shuffle_images[:600]
        test_image_files = shuffle_images[600:]

        # 이미지를 저장할 폴더 생성
        train_folder_path = "./%s_train_images/"%self.folder_path
        os.makedirs(train_folder_path, exist_ok=True)
        valid_folder_path = "./%s_valid_images/"%self.folder_path
        os.makedirs(valid_folder_path, exist_ok=True)
        test_folder_path = "./%s_test_images/"%self.folder_path
        os.makedirs(test_folder_path, exist_ok=True)

        for image_file in train_image_files:
            source_image_path = os.path.join(self.folder_path, image_file)
            save_image_path = os.path.join(train_folder_path, image_file)
            shutil.copyfile(source_image_path, save_image_path)

        for image_file in valid_image_files:
            source_image_path = os.path.join(self.folder_path, image_file)
            save_image_path = os.path.join(valid_folder_path, image_file)
            shutil.copyfile(source_image_path, save_image_path)

        for image_file in test_image_files:
            source_image_path = os.path.join(self.folder_path, image_file)
            save_image_path = os.path.join(test_folder_path, image_file)
            shutil.copyfile(source_image_path, save_image_path)


input_folders_path = ['C1','T1','I1','G1','H1']
for folder_path in input_folders_path:
    processor = DataProcessor(folder_path)
    processor.process_files()


class ImageDataOrganizer:
    def __init__(self, inputs_path):
        self.inputs_path = inputs_path

    def image_merge(self, output_images_path):
        images = []
        for input_path in self.inputs_path:
            for filename in os.listdir(input_path):
                source_image_path = os.path.join(input_path, filename)
                save_image_path = os.path.join(output_images_path, filename)
                shutil.copyfile(source_image_path, save_image_path)
                images.append(save_image_path)


train_folder_paths = ['C1_train_images',
                      'T1_train_images',
                      'I1_train_images',
                      'G1_train_images',
                      'H1_train_images']
train_images_path = './total_train_images'
os.makedirs(train_images_path, exist_ok=True)
processor = ImageDataOrganizer(train_folder_paths)
processor.image_merge(train_images_path)

valid_folder_paths = ['C1_valid_images',
                      'T1_valid_images',
                      'I1_valid_images',
                      'G1_valid_images',
                      'H1_valid_images']
valid_images_path = './total_valid_images'
os.makedirs(valid_images_path, exist_ok=True)
processor = ImageDataOrganizer(valid_folder_paths)
processor.image_merge(valid_images_path)

test_folder_paths = ['C1_test_images',
                     'T1_test_images',
                     'I1_test_images',
                     'G1_test_images',
                     'H1_test_images']
test_images_path = './total_test_images'
os.makedirs(test_images_path, exist_ok=True)
processor = ImageDataOrganizer(test_folder_paths)
processor.image_merge(test_images_path)

print(len(os.listdir('./total_train_images'))) # 24,000
print(len(os.listdir('./total_valid_images'))) # 3,000
print(len(os.listdir('./total_test_images'))) # 3,000


### 데이터 전처리 ###
train_img_list = glob.glob('./total_train_images/*')
train_df = pd.DataFrame(columns=['img_path','label'])
train_df['img_path'] = train_img_list
train_df['label'] = train_df['img_path'].apply(lambda x:str(x).split('_')[4])

valid_img_list = glob.glob('./total_valid_images/*')
valid_df = pd.DataFrame(columns=['img_path','label'])
valid_df['img_path'] = valid_img_list
valid_df['label'] = valid_df['img_path'].apply(lambda x:str(x).split('_')[4])

test_img_list = glob.glob('./total_test_images/*')
test_df = pd.DataFrame(columns=['img_path','label'])
test_df['img_path'] = test_img_list
test_df['label'] = test_df['img_path'].apply(lambda x:str(x).split('_')[4])


le = preprocessing.LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
valid_df['label'] = le.transform(valid_df['label'])



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


train_transform = A.Compose([
    A.Resize(params['IMG_SIZE'],params['IMG_SIZE']),
    A.RandomBrightnessContrast(brightness_limit=0.2,
                               contrast_limit=0.2, p=0.3),
    A.VerticalFlip(p = 0.2),
    A.HorizontalFlip(p = 0.5),
    A.ShiftScaleRotate(
        shift_limit = 0.1,
        scale_limit = 0.2,
        rotate_limit = 30, p = 0.3),
    A.OneOf([A.Emboss(p = 1),
             A.Sharpen(p = 1),
             A.Blur(p = 1)], p = 0.3),
    A.PiecewiseAffine(p = 0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(params['IMG_SIZE'],params['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
])


train_dataset = CustomDataset(train_df['img_path'].values, train_df['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=False, num_workers=0)

val_dataset = CustomDataset(valid_df['img_path'].values, valid_df['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=params['BATCH_SIZE'], shuffle=False, num_workers=0)



### 모델 훈련 ###
class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0')
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.sigmoid(self.classifier(x))
        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_score = 0
    best_model = None

    for epoch in range(1, params['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Accuracy Score : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score < _val_score:
            best_score = _val_score
            best_model = model

        # best 모델 저장
        torch.save(best_model, "../best_model.pt")
    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)

            loss = criterion(pred, labels)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()

            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_score = accuracy_score(true_labels, preds)

    return _val_loss, _val_score



### train ###
model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = params["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)



### test ###
test_dataset = CustomDataset(test_df['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=params['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            pred = model(imgs)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()

    preds = le.inverse_transform(preds)
    return preds


preds = inference(infer_model, test_loader, device)
print(classification_report(test_df['label'], preds))


