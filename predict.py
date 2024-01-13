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
from models import HydraNetModified
from utils import convert,mapping,get_model_analysis,transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_detect_path', type=str, default='weights/yolov5s-face.pt')
    parser.add_argument('--json_path', type=str, default='labels.json')
    parser.add_argument('--data_folder_path', type=str, default='public_test')
    parser.add_argument('--data_img2id_path', type=str, default='public_test_and_submission_guidelines/file_name_to_image_id.json')
    parser.add_argument('--csv_path', type=str, default='public_test_and_submission_guidelines/answer.csv')
    parser.add_argument('--save_path', type=str, default='answer.csv')
    parser.add_argument('--model_name', type=str, default='efficientnet')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--weights_path', type=str, default='weights_2/model_15.pth')
    args = parser.parse_args()
    return args
def main():
    args=parse_args()
    weights_detect_path=args.weights_detect_path
    json_path=args.json_path
    data_folder_path=args.data_folder_path
    save_path=args.save_path
    model_name=args.model_name
    img_size=args.img_size
    weights_path=args.weights_path
    data_img2id_path=args.data_img2id_path
    csv_path=args.csv_path
    detector = Yolov5Face(model_file=weights_detect_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=get_model_analysis(name=model_name,weight_path=weights_path)
    age_mapping,gender_mapping,race_mapping,emotion_mapping,skintone_mapping,masked_mapping=mapping(json_path=json_path)
    df_answer=pd.read_csv(csv_path)
    mapping_id=json.load(open(data_img2id_path))
    list_submit=[]
    for i in range(len(df_answer)):
        dict_answer=df_answer.iloc[i].to_dict()
        img_name=dict_answer["file_name"]
        img_path=os.path.join(data_folder_path,img_name)
        img=cv2.imread(img_path)
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
            image_id=mapping_id[img_name]
            row=[img_name,bbox_str,image_id,race,age,emotion,gender,skintone,masked]
            list_submit.append(row)
    df_submit=pd.DataFrame()
    df_submit["file_name"]=[row[0] for row in list_submit]
    df_submit["bbox"]=[row[1] for row in list_submit]
    df_submit["image_id"]=[row[2] for row in list_submit]
    df_submit["race"]=[row[3] for row in list_submit]
    df_submit["age"]=[row[4] for row in list_submit]
    df_submit["emotion"]=[row[5] for row in list_submit]
    df_submit["gender"]=[row[6] for row in list_submit]
    df_submit["skintone"]=[row[7] for row in list_submit]
    df_submit["masked"]=[row[8] for row in list_submit]
    df_submit.to_csv(save_path,index=False)
