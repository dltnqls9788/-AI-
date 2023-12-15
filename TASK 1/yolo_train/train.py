from ultralytics import YOLO
import cv2
import seaborn 
import argparse 
from pytz import timezone
from datetime import datetime
import torch
import random
import os 
import numpy as np 
import json
import shutil
import yaml
print(torch.__version__)
print(torch.cuda.is_available())
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

current_directory = os.getcwd()
#######################  폴더 생성  #######################
if os.path.exists("./yolo_data/"):
    shutil.rmtree("./yolo_data/")

if not os.path.exists("./yolo_data/train"):
    os.makedirs("./yolo_data/train")
    os.makedirs("./yolo_data/train/images")
    os.makedirs("./yolo_data/train/labels")
    
if not os.path.exists("./yolo_data/valid"):
    os.makedirs("./yolo_data/valid")
    os.makedirs("./yolo_data/valid/images")
    os.makedirs("./yolo_data/valid/labels")
    
if not os.path.exists("./yolo_data/test"):
    os.makedirs("./yolo_data/test")    
    os.makedirs("./yolo_data/test/images")
    os.makedirs("./yolo_data/test/labels")

if not os.path.exists("./results"):
    os.makedirs("./results")

##### 데이터 
def parse_args():
    parser = argparse.ArgumentParser()
    # datasets
    parser.add_argument("--data", type=str, default="./S63_1_DATA3/")
   
    
    args = parser.parse_args()
    
    return args 

args = parse_args()
folder_path = args.data
json_files = [f for f in os.listdir(folder_path)]
print(json_files)

# 좌표를 voc 형태에서 yolo 형태로 변환 함수
def voc_to_yolo(x_min, y_min, x_max, y_max, image_width, image_height):
    
    # VOC 좌표는 좌상단(x_min, y_min)과 우하단(x_max, y_max)을 기준으로 주어짐
    x = (x_min + x_max) / (2 * image_width)
    y = (y_min + y_max) / (2 * image_height)
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return x, y, width, height

def jsontotxt(data, object_class, bbox):
    if  len(data['annotations'][0]['bbox']) == 4: 
            x_min, y_min, x_max, y_max  = data['annotations'][0]['bbox']
            x, y, width, height = voc_to_yolo(x_min, y_min, x_max, y_max, data['images']['width'], data['images']['height'])
            bbox.extend([object_class,x, y, width, height])
    elif len(data['annotations'][1]['bbox']) == 4: 
        x_min, y_min, x_max, y_max  = data['annotations'][1]['bbox']
        x, y, width, height = voc_to_yolo(x_min, y_min, x_max, y_max, data['images']['width'], data['images']['height'])
        bbox.extend([object_class,x, y, width, height])
    return bbox

####################### yolo형태 txt 변환 후 저장 #######################
class Voc_to_Yolo:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
    def precess_files(self):
        file_list = os.listdir(self.folder_path)
        
        for i in file_list:
            json_files = [f for f in os.listdir(folder_path + i+'/') if f.endswith('.json')]
            for j in json_files:
                with open (folder_path + i + '/'+ j , 'r',encoding='utf-8') as file:
                    data = json.load(file)
                    bbox = []
                    if i == 'B0':
                        object_class = 0 
                        result_bbox = jsontotxt(data,object_class, bbox)
                    elif i == 'B1':
                        object_class = 1 
                        result_bbox = jsontotxt(data,object_class, bbox)
                        
                    elif i == 'H0':
                        object_class = 2 
                        result_bbox = jsontotxt(data,object_class, bbox)
                        
                    elif i == 'H1':
                        object_class = 3
                        result_bbox = jsontotxt(data,object_class, bbox)
                        
                    elif i == 'M0':
                        object_class = 4 
                        result_bbox = jsontotxt(data,object_class, bbox)
                        
                    elif i == 'M1':
                        object_class = 5
                        result_bbox = jsontotxt(data,object_class, bbox)      
                        
                with open(folder_path + i + '/'+ j[:-5] +'.txt', 'w', encoding='utf-8') as txt_file:
                    for number in result_bbox:
                        txt_file.write(str(number)+' ')     
                        
processor= Voc_to_Yolo(folder_path)
processor.precess_files()

####################### Train/Valid/Test #######################

path = './yolo_data/'
tvt = ['train','valid','test']

class TVT:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
    def precess_files(self):
        file_list = os.listdir(self.folder_path)
        
        for i in file_list:
            image_files = [f for f in os.listdir(folder_path + i+'/') if f.endswith('.jpg')]
            txt_files = [f for f in os.listdir(folder_path + i+'/') if f.endswith('.txt')]
            image_files.sort()
            txt_files.sort()
            
            train_image_files,train_label_files = image_files[:4000], txt_files[:4000] 
            valid_image_files,valid_label_files = image_files[4000:4500], txt_files[4000:4500] 
            test_image_files,test_label_files = image_files[4500:], txt_files[4500:] 
            
            for image_file, txt_file in zip(train_image_files,train_label_files):
                source_image_path = os.path.join(self.folder_path,i, image_file)
                save_image_path = os.path.join(path+tvt[0]+'/images/', image_file)
                shutil.copyfile(source_image_path, save_image_path)

                source_txt_path = os.path.join(self.folder_path,i, txt_file)
                save_txt_path = os.path.join(path+tvt[0]+'/labels/', txt_file)
                shutil.copyfile(source_txt_path, save_txt_path)
            
            for image_file, txt_file in zip(valid_image_files, valid_label_files):
                source_image_path = os.path.join(self.folder_path,i, image_file)
                save_image_path = os.path.join(path+tvt[1]+'/images/', image_file)
                shutil.copyfile(source_image_path, save_image_path)

                source_txt_path = os.path.join(self.folder_path,i, txt_file)
                save_txt_path = os.path.join(path+tvt[1]+'/labels/', txt_file)
                shutil.copyfile(source_txt_path, save_txt_path)
            
            for image_file, txt_file in zip(test_image_files, test_label_files):
                source_image_path = os.path.join(self.folder_path,i, image_file)
                save_image_path = os.path.join(path+tvt[2]+'/images/', image_file)
                shutil.copyfile(source_image_path, save_image_path)
                
                source_txt_path = os.path.join(self.folder_path,i, txt_file)
                save_txt_path = os.path.join(path+tvt[2]+'/labels/', txt_file)
                shutil.copyfile(source_txt_path, save_txt_path)
            
processor = TVT(folder_path)
processor.precess_files()       

import yaml

    
################ Model Run ######################   
if __name__ == '__main__':
        # Load a model
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='./task1.yaml',
            project = "results",
            name = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d_%H%M%S"),
            imgsz = 640,
            epochs=3 ,
            patience=50,
            max_det=20,
            optimizer='AdamW',
            label_smoothing=0.1,
            cos_lr=True ,
            lr0=0.001,
            batch=32)
    