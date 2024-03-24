import cv2
import os
import shutil
import pandas as pd
import albumentations as A
import random


for file in os.listdir("data_aug"):
    #data_aug is folder for data augmentation
    img_path="data_aug/"+file
    img=cv2.imread(img_path)
    num_aug=7
    img_aug=img
    for i in range(num_aug):
        transform1=A.HorizontalFlip(p=1)
        transform2=A.RandomBrightnessContrast()
        transform3=A.ColorJitter(contrast=0.3,brightness=0.3)
        transform4=A.Blur()
        transform5=A.GaussNoise()
        transform6=A.RandomFog()
        transform7=A.RandomRain()
        a=i+1
        if a==1:
            img_aug=transform1(image=img)["image"]
        if a==2:
            img_aug=transform2(image=img)["image"]
        if a==3:
            img_aug=transform3(image=img)["image"]
        if a==4:
            img_aug=transform4(image=img)["image"]
        if a==5:
            img_aug=transform5(image=img)["image"]
        if a==6:
            img_aug=transform6(image=img)["image"]
        if a==7:
            img_aug=transform7(image=img)["image"]
        #print(img_aug)
        cv2.imwrite("crop/"+file[:-4]+"_"+str(i)+".jpg",img_aug)


# transform1=A.HorizontalFlip(p=1)
# transform2=A.RandomBrightnessContrast()
# transform3=A.ColorJitter(contrast=0.8)
# transform4=A.Blur()
# transform5=A.GaussNoise()
# transform6=A.RandomFog()
# transform7=A.RandomRain()

# img=cv2.imread("data_aug/154.jpg")
# img1=transform1(image=img)["image"]
# img2=transform2(image=img)["image"]
# img3=transform3(image=img)["image"]
# img4=transform4(image=img)["image"]
# img5=transform5(image=img)["image"]
# img6=transform6(image=img)["image"]
# img7=transform7(image=img)["image"]
# cv2.imwrite("data1/154.jpg",img)
# cv2.imwrite("data1/154_1.jpg",img1)
# cv2.imwrite("data1/154_2.jpg",img2)
# cv2.imwrite("data1/154_3.jpg",img3)
# cv2.imwrite("data1/154_4.jpg",img4)
# cv2.imwrite("data1/154_5.jpg",img5)
# cv2.imwrite("data1/154_6.jpg",img6)
# cv2.imwrite("data1/154_7.jpg",img7)
