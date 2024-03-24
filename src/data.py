import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
class FaceChallenge(Dataset):
        def __init__(self,df,crop_path="crop",img_size=(224,224)):
            # Define the Transforms
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ])
            self.images = []
            self.ages = []
            self.genders = []
            self.races = []
            self.maskeds = []
            self.skintones = []
            self.emotions = []
            self.names=[]
            for i in range(len(df)):
                dict=df.iloc[i].to_dict()
                file_name=dict["crop_name"]
                self.names.append(dict["file_name"])
                self.images.append(os.path.join(crop_path,file_name))
                self.ages.append(dict["age"])
                self.races.append(dict["race"])
                self.genders.append(dict["gender"])
                self.maskeds.append(dict["masked"])
                self.skintones.append(dict["skintone"])
                self.emotions.append(dict["emotion"])
        def __len__(self):
            return len(self.images)
        def __getitem__(self, index):
            # Load an Image
            img = Image.open(self.images[index]).convert('RGB')
            # Transform it
            img = self.transform(img)
            # Get the Labels
            age = self.ages[index]
            gender = self.genders[index]
            race = self.races[index]
            masked = self.maskeds[index]
            skintone = self.skintones[index]
            emotion = self.emotions[index]
            return {'image':img, 'age': age, 'gender': gender,"race":race,"masked":masked,"skintone":skintone,"emotion":emotion,"name":self.names[index],"crop_name":self.images[index]}