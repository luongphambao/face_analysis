import pandas as pd 
import cv2
import os
import shutil
df=pd.read_csv("labels1.csv")
#label=str(df["emotion"])+str(df["skintone"])+str(df["age"])
df["label"]=df["emotion"].astype(str)+df["skintone"].astype(str)+df["age"].astype(str)+df["race"].astype(str)
distribution=df["label"].value_counts()
#convert to dict
print(len(distribution))
distribution=distribution.to_dict()
#percentage of top 20
top20=0
index=0
# for key,value in distribution.items():
#         top20+=value
#         index+=1
#         if index==30:
#                 break

# print(top20/len(df))
# for key,value in distribution.items():
#         print(key,value)
#get key of top 30
top30=[]
index=0
for key,value in distribution.items():
        if index>30:
              top30.append(key)
        index+=1
list_data=[]
for key in top30:
    df_find=df[df["label"]==key]
    crop_name=df_find["crop_name"].tolist()
    for name in crop_name:
        row=df[df["crop_name"]==name].values.tolist()[0]
        aug_name1=name[:-4]+"_1.jpg"
        aug_name2=name[:-4]+"_2.jpg"
        aug_name3=name[:-4]+"_3.jpg"
        aug_name4=name[:-4]+"_4.jpg"
        aug_name5=name[:-4]+"_5.jpg"
        aug_name6=name[:-4]+"_6.jpg"
        aug_name7=name[:-4]+"_7.jpg"
        list_aug=[aug_name1,aug_name2,aug_name3,aug_name4,aug_name5,aug_name6,aug_name7]
        for aug in list_aug:
            #if os.path.exists("data1/"+aug)==True:
                #print(row)
                #exit()
            print(row)
           
            new_row=[row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],aug,row[11]]
            print(new_row)
            #@exit()
            #print(row)
            list_data.append(new_row)
                #print(list_data)
                #exit()
        #exit()
df_new=pd.DataFrame(list_data,columns=df.columns)
df_new.to_csv("labels_add.csv",index=False)
        
  