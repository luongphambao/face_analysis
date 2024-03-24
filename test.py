from PIL import Image
import torch 
from torchvision.models import resnet18, resnet34, resnet50, resnet101,efficientnet_b0,efficientnet_b1
import numpy as np
import cv2
import json
import torchvision
import pandas as pd
import torch.nn.functional as F
import argparse 
import os 
from yolov5_face.detector import Yolov5Face
print("hello")
import sys
sys.path.insert(0,"/")
from src.model import HydraNetModified
from src.util import convert,mapping,get_model_analysis,transform



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_detect_path', type=str, default='weights/yolov5s-face.pt')
    parser.add_argument('--model_name', type=str, default='efficientnet')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--weights_path', type=str, default='weights/model.pth')
    parser.add_argument('--json_path', type=str, default='labels.json')
    args = parser.parse_args()
    return args
def main():
    args=parse_args()
    weights_detect_path=args.weights_detect_path
    model_name=args.model_name
    img_size=args.img_size
    weights_path=args.weights_path
    json_path=args.json_path
    #print("hello")
    detector = Yolov5Face(model_file=weights_detect_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=get_model_analysis(name=model_name,weight_path=weights_path)
    #model.to(device=device)
    age_mapping,gender_mapping,race_mapping,emotion_mapping,skintone_mapping,masked_mapping=mapping(json_path=json_path)
    img_path="images/2363834.jpg"
    
    img_name=os.path.basename(img_path)
    img=cv2.imread(img_path)
    height,width,_=img.shape
    bboxes, landmarks=detector.detect(image=img)
    for bbox in bboxes[:]:
        x1,y1,x2,y2,conf=bbox
        x,y,w,h=convert(x1,y1,x2,y2)
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        img_crop=img[y1:y2,x1:x2]
        image=Image.fromarray(img_crop)
        height, width = image.size
        image_norm = transform(image)
        image_norm = image_norm.view(1, 3, image_norm.shape[1], image_norm.shape[2])
        print(image_norm.shape)
        print(model(image_norm.to(device)))
        age, gender, race,masked,skintone,emotion = model(image_norm.to(device))
        age=torch.argmax(F.softmax(age, dim=1), dim=1).item()
        gender=torch.argmax(F.softmax(gender, dim=1), dim=1).item()
        race=torch.argmax(F.softmax(race, dim=1), dim=1).item()
        masked=torch.argmax(F.softmax(masked, dim=1), dim=1).item()
        skintone=torch.argmax(F.softmax(skintone, dim=1), dim=1).item()
        emotion=torch.argmax(F.softmax(emotion, dim=1), dim=1).item()
        age=age_mapping[age]
        gender=gender_mapping[gender]
        race=race_mapping[race]
        emotion=emotion_mapping[emotion]
        skintone=skintone_mapping[skintone]
        masked=masked_mapping[masked]
        bbox_str=str([x,y,w,h])
        print("age:",age)
        print("emotion:",emotion)
        print("race:",race)
        print("gender",gender)
        print("skintone",skintone)
        print("masked",masked)
if __name__ == "__main__":
    main()