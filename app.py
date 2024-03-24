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
from fastapi import Depends, File, Form, Request, UploadFile, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from yolov5_face.detector import Yolov5Face
logger.info("import libraries done")
import sys
sys.path.insert(0,"/")
import src.config as cfg
from src.model import HydraNetModified
from src.util import convert,mapping,get_model_analysis,transform


#load config
weights_detect_path=cfg.weights_detect_path
model_name=cfg.model_name
img_size=cfg.img_size
weights_path=cfg.weights_path
json_path=cfg.json_path

device=cfg.device
#load model
face_detector=Yolov5Face(model_file=weights_detect_path)
logger.info("Face detector loaded")
model=get_model_analysis(name=model_name,weight_path=weights_path)
logger.info("Face analysis model loaded")
age_mapping,gender_mapping,race_mapping,emotion_mapping,skintone_mapping,masked_mapping=mapping(json_path=json_path)
#face_detector.to(device)
model.to(device)

app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(name="index.html",request=request,upload=False)
@app.post("/")
async def face_analysis(file: UploadFile=File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logger.info("Image loaded")
    height,width,_=img.shape
    bboxes, landmarks=face_detector.detect(image=img)
    logger.info("Face detected")
    for bbox in bboxes[:]:
        x1,y1,x2,y2,conf=bbox
        x,y,w,h=convert(x1,y1,x2,y2)
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        img_crop=img[y1:y2,x1:x2]
        cv2.imwrite("static/crop.jpg",img_crop)
        image=Image.fromarray(img_crop)
        height, width = image.size
        image_norm = transform(image)
        image_norm = image_norm.view(1, 3, image_norm.shape[1], image_norm.shape[2])
        #print(image_norm.shape)
        age, gender, race,masked,skintone,emotion =model(image_norm.to(device))
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
        logger.info("Face analysis done")
        result={"age":age,
                "gender":gender,
                "race":race,
                "emotion":emotion,
                "skintone":skintone,
                "masked":masked
                }
        return JSONResponse(content=result)

