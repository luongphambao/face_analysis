import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
from torchvision.models import resnet18, resnet34, resnet50, resnet101,efficientnet_b0,efficientnet_b1
from torch import nn
from model import HydraNetModified
from data import FaceChallenge
from util import convert,mapping,get_model_analysis,transform
def main():
    os.makedirs("weights",exist_ok=True)
    train_df=pd.read_csv("labels_train.csv")
    valid_df=pd.read_csv("labels_valid.csv")
    
    add_df=pd.read_csv("labels_add.csv")
    add_df=add_df.drop(columns=["label"])
    add_df=add_df.sample(n=8000,random_state=42)
    train_df=pd.concat([train_df,add_df],ignore_index=True)
    BATCH_SIZE =16
    train_dataset = FaceChallenge(train_df)
    valid_dataset = FaceChallenge(valid_df)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE)

    train_steps = len(train_dataloader.dataset) // BATCH_SIZE
    val_steps = len(val_dataloader.dataset) // BATCH_SIZE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=get_model_analysis(name="efficientnet",weight_path=None)
    loss_fn_1 = nn.CrossEntropyLoss() 
    loss_fn_2 = nn.CrossEntropyLoss() 
    loss_fn_3 = nn.CrossEntropyLoss() 
    loss_fn_4 = nn.CrossEntropyLoss() 
    loss_fn_5 = nn.CrossEntropyLoss()
    loss_fn_6 = nn.CrossEntropyLoss()

    n_epochs = 100
    lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logger = {"train_loss": [],
            "validation_loss": [],
            "train_gender_loss": [],
            "train_race_loss": [],
            "train_age_loss": [],
            "validation_gender_loss": [],
            "validation_race_loss": [],
            "validation_age_loss": []
            }
    sig = nn.Sigmoid()

    for epoch in range(n_epochs):
        model.train()

        total_training_loss = 0
        total_validation_loss = 0
        training_gender_loss = 0
        training_race_loss = 0
        training_age_loss = 0
        training_masked_loss = 0
        training_skintone_loss = 0
        training_emotion_loss = 0
        validation_gender_loss = 0
        validation_race_loss = 0
        validation_age_loss = 0
        validation_masked_loss = 0
        validation_skintone_loss = 0
        validation_emotion_loss = 0

        index=1
        for data in train_dataloader:
            print(f"{round(index*100/len(train_dataloader),3)}%")
            inputs = data["image"].to(device=device)
            age_label = data["age"].to(device=device)     
            gender_label = data["gender"].to(device=device)
            race_label = data["race"].to(device=device)
            masked_label = data["masked"].to(device=device)
            skintone_label = data["skintone"].to(device=device)
            emotion_label = data["emotion"].to(device=device)
            optimizer.zero_grad()

            age_output, gender_output, race_output,masked_output,skintone_output,emotion_output = model(inputs)
            loss_3 = loss_fn_3(race_output, race_label)
            loss_2 = loss_fn_2(gender_output, gender_label)
            loss_1 = loss_fn_1(age_output, age_label)
            loss_4 = loss_fn_4(masked_output, masked_label)
            loss_5 = loss_fn_5(skintone_output, skintone_label)
            loss_6 = loss_fn_6(emotion_output, emotion_label)
            loss = loss_1*3 + loss_2*0.5+ loss_3*2.5+loss_4*0.5+loss_5*3+loss_6*3.5
            #print(loss)
            loss.backward()
            optimizer.step()
            total_training_loss += loss

            training_race_loss += loss_3.item()
            training_gender_loss += loss_2.item()
            training_age_loss += loss_1.item()
            training_masked_loss += loss_4.item()
            training_skintone_loss += loss_5.item()
            training_emotion_loss += loss_6.item()
            index+=1
        print('EPOCH ', epoch+1)
        print(f"Training Losses: Race: {loss_1}, Gender: {loss_2}, Age: {loss_3}, Masked: {loss_4}, Skintone: {loss_5}, Emotion: {loss_6}")

        with torch.no_grad():
            model.eval()

            for data in val_dataloader:
                inputs = data["image"].to(device=device)
                age_label = data["age"].to(device=device)
                gender_label = data["gender"].to(device=device)
                race_label =  data["race"].to(device=device)
                masked_label = data["masked"].to(device=device)
                skintone_label = data["skintone"].to(device=device)
                emotion_label = data["emotion"].to(device=device)
                age_output, gender_output, race_output,masked_output,skintone_output,emotion_output = model(inputs)

                loss_3 = loss_fn_3(race_output, race_label)
                loss_2 = loss_fn_2(gender_output, gender_label)
                loss_1 = loss_fn_1(age_output, age_label)
                loss_4 = loss_fn_4(masked_output, masked_label)
                loss_5 = loss_fn_5(skintone_output, skintone_label)
                loss_6 = loss_fn_6(emotion_output, emotion_label)
                loss = loss_1*3 + loss_2*0.5+ loss_3*2.5+loss_4*0.5+loss_5*3+loss_6*3.5
                #print(loss)    
                total_validation_loss += loss

                validation_race_loss += loss_3.item()
                validation_gender_loss += loss_2.item()
                validation_age_loss += loss_1.item()
                validation_masked_loss += loss_4.item()
                validation_skintone_loss += loss_5.item()
                validation_emotion_loss += loss_6.item()
            print(f"Validation Losses: Race: {loss_1}, Gender: {loss_2}, Age: {loss_3}, Masked: {loss_4}, Skintone: {loss_5}, Emotion: {loss_6}")

        avgTrainLoss = total_training_loss / train_steps
        avgValLoss = total_validation_loss / val_steps

        print(f'Average Losses — Training: {avgTrainLoss} | Validation {avgValLoss}')
        with open("log.txt","a+") as f:
           f.write(f'EPOCH {epoch+1}: Average Losses — Training: {avgTrainLoss} | Validation {avgValLoss}\n')
        print()
        avgTrainGenderLoss = training_gender_loss/len(train_dataloader.dataset)
        avgTrainRaceLoss = training_race_loss/len(train_dataloader.dataset)
        avgTrainAgeLoss = training_age_loss/len(train_dataloader.dataset)

        avgValGenderLoss = validation_gender_loss/len(val_dataloader.dataset)
        avgValRaceLoss = validation_race_loss/len(val_dataloader.dataset)
        avgValAgeLoss = validation_age_loss/len(val_dataloader.dataset)
        if epoch%5==0:
            torch.save(model.state_dict(), "weights/model_"+str(epoch)+".pth")
        logger["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        logger["train_gender_loss"].append(avgTrainGenderLoss)
        logger["train_race_loss"].append(avgTrainRaceLoss)
        logger["train_age_loss"].append(avgTrainAgeLoss)

        logger["validation_loss"].append(avgValLoss.cpu().detach().numpy())
        logger["validation_gender_loss"].append(avgValGenderLoss)
        logger["validation_race_loss"].append(avgValRaceLoss)
        logger["validation_age_loss"].append(avgValAgeLoss)
    torch.save(model.state_dict(), "best_model.pth")
if __name__ == "__main__":
    main()