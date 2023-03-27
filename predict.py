import numpy as np
from PIL import Image, ImageOps
from fastai.vision import open_image, load_learner, image, torch
import torch

def infer(img):
    model = cnn_learner(data, models.resnet50, metrics=[accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
    pred_class = model.predict(img)[0]
    pred_prob = round(torch.max(model.predict(img)[2]).item()*100)
    
    return (pred_class, pred_prob)
