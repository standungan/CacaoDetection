import math
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm, notebook

def test(model, loader, device):
    model.to(device)
    model.eval()
    
    for images, targets in tqdm(loader, desc=f"Testing"):
        
        images = list(image.to(device) for image in images)
        
        targets = [{k:v.to(device) for k, v in t.items()} for t in targets]
        
        predicts = model(images)
        
        
        
    