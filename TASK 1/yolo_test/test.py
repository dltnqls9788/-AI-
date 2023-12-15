from ultralytics import YOLO
import cv2
import os 

if __name__ == '__main__':
    #args = parse_args()
    
    test_image = './test_images/'
    
    model = YOLO('./yolo_best.pt')
    
    file_list = os.listdir(test_image)
    results = model.predict( [test_image + f for f in file_list ])

b0,b1,h0,h1,m0,m1 = list(),list(),list(),list(),list(),list()
for i in results:
    if i.boxes.cls[0] == 0:
        b0.append(i.boxes.conf[0])
    elif i.boxes.cls[0] == 1:
        b1.append(i.boxes.conf[0])
    elif i.boxes.cls[0] == 2:
        h0.append(i.boxes.conf[0])
    elif i.boxes.cls[0] == 3:
        h1.append(i.boxes.conf[0])
    elif i.boxes.cls[0] == 4:
        m0.append(i.boxes.conf[0])
    elif i.boxes.cls[0] == 5:
        m1.append(i.boxes.conf[0])
        
if len(b0) != 0 :
    print('B0 mAP : ', sum(b0)/len(b0))
    
if len(b1) != 0 :
    print('B0 mAP : ', sum(b1)/len(b1))
    
if len(h0) != 0 :
    print('B0 mAP : ', sum(h0)/len(h0))
    
if len(h1) != 0 :
    print('B0 mAP : ', sum(h1)/len(h1))
    
if len(m0) != 0 :
    print('B0 mAP : ', sum(m0)/len(m0))
    
if len(m1) != 0 :
    print('B0 mAP : ', sum(m1)/len(m1))

all_map = list()
for i in results:
    all_map.append(i.boxes.conf[0])
    
print('All mAP : ', sum(all_map)/len(all_map))