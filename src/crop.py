import os 
import cv2 
import pandas as pd 
data_path = 'data'
crop_path = 'crop'
labels_path="labels1.csv"
if not os.path.exists(crop_path):
    os.makedirs(crop_path)
df = pd.read_csv(labels_path)
index=0
for i in range(len(df)):

    dict=df.iloc[i].to_dict()
    #print(dict)
    #exit()
    file_name=dict["file_name"]
    height=dict["height"]
    width=dict["width"]
    bbox=dict["bbox"]
    #convert string to list
    bbox=bbox.strip("[]")
    bbox=bbox.split(",")
    bbox=[int(float(i)) for i in bbox]
    #crop bbox with same name
    new_file_name=dict["crop_name"]

    new_img_path=os.path.join(crop_path,new_file_name)
    img_path=os.path.join(data_path,file_name)
    img=cv2.imread(img_path)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x,y,w,h=bbox
    crop_img=img[y:y+h,x:x+w]
    cv2.imwrite(new_img_path,crop_img)
    print("saved",new_img_path)
    index+=1