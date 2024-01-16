import torch
import torchvision
from torchvision import transforms
import json 
from torchvision.models import resnet18, resnet34, resnet50, resnet101,efficientnet_b0,efficientnet_b1
from model import HydraNetModified
def convert(x1,y1,x2,y2):
    "return x,y,w,h"
    x=x1
    y=y1
    w=x2-x1
    h=y2-y1
    return x,y,w,h
transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ])
def mapping(json_path):
    mapping_class = json.load(open(json_path))
    age_mapping = mapping_class['age']
    age_mapping = {v: k for k, v in age_mapping.items()}
    gender_mapping=mapping_class["gender"]
    gender_mapping = {v: k for k, v in gender_mapping.items()}
    race_mapping=mapping_class["race"]
    race_mapping={v: k for k, v in race_mapping.items()}
    emotion_mapping=mapping_class["emotion"]
    emotion_mapping={v: k for k, v in emotion_mapping.items()}
    skintone_mapping=mapping_class["skintone"]
    skintone_mapping={v: k for k, v in skintone_mapping.items()}
    masked_mapping=mapping_class["masked"]
    masked_mapping={v: k for k, v in masked_mapping.items()}
    return age_mapping,gender_mapping,race_mapping,emotion_mapping,skintone_mapping,masked_mapping
def get_model_analysis(name="efficientnet",weight_path="weights_2/model_15.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name=="efficientnet":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        net=torchvision.models.efficientnet_b0(weights=weights)
        model = HydraNetModified(net,backbone="efficientnet")
    elif name=="resnet":
        net = resnet34(pretrained=True)
        model = HydraNetModified(net,backbone="resnet")
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    model.to(device=device)
    model.eval()
    return model