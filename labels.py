import pandas as pd 
import numpy as np
import os

df=pd.read_csv("labels.csv")
#list columns
print(df.columns)
list_age_label=df["age"].unique()
list_race_label=df["race"].unique()
list_masked_label=df["masked"].unique()
list_skintone_label=df["skintone"].unique()
list_emotion_label=df["emotion"].unique()
list_gender_label=df["gender"].unique()
#convert to dict number
age_dict={}
print((list_age_label))
for i in range(len(list_age_label)):
    print(list_age_label[i])
    print(i)
    age_dict[list_age_label[i]]=i
race_dict={}
for i in range(len(list_race_label)):
    race_dict[list_race_label[i]]=i
masked_dict={}
for i in range(len(list_masked_label)):
    masked_dict[list_masked_label[i]]=i
skintone_dict={}
for i in range(len(list_skintone_label)):
    skintone_dict[list_skintone_label[i]]=i
emotion_dict={}
for i in range(len(list_emotion_label)):
    emotion_dict[list_emotion_label[i]]=i
gender_dict={}
for i in range(len(list_gender_label)):
    gender_dict[list_gender_label[i]]=i
dict_labels={
    "age":age_dict,
    "race":race_dict,
    "masked":masked_dict,
    "skintone":skintone_dict,
    "emotion":emotion_dict,
    "gender":gender_dict
}
import json 
with open("labels.json", "w") as outfile: 
    json.dump(dict_labels, outfile)
df["age"]=df["age"].map(age_dict)
df["race"]=df["race"].map(race_dict)
df["masked"]=df["masked"].map(masked_dict)
df["skintone"]=df["skintone"].map(skintone_dict)
df["emotion"]=df["emotion"].map(emotion_dict)
df["gender"]=df["gender"].map(gender_dict)
df.to_csv("labels1.csv",index=False)
#list_race_label