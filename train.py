import math
import sys
import os
import yaml
import torch
import albumentations as Albus
import torch.nn as nn
import torchvision
from model import cnnLayer
from tqdm import tqdm, notebook
from dataset import KakaoDetection
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torchvision.utils import draw_bounding_boxes
from torchvision.models import detection
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from train_one_epoch import train_one_epoch



if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    img_size = config["IMG_SIZE"]
    def get_transforms(train=False):
        if train:
            transform = Albus.Compose([
                Albus.Resize(img_size, img_size),
                Albus.HorizontalFlip(p=0.3),
                Albus.VerticalFlip(p=0.3),
                ToTensorV2()], 
                bbox_params=Albus.BboxParams(format='coco'))
        else:
            transform = Albus.Compose([
                Albus.Resize(img_size, img_size),
                ToTensorV2()], 
                bbox_params=Albus.BboxParams(format='coco'))
        return transform
    
    train_dataset = KakaoDetection(root=config["ROOT_DIR"], split="train", transforms=get_transforms(True))
    test_dataset = KakaoDetection(root=config["ROOT_DIR"], split="test", transforms=get_transforms(False))

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_load = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_load = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    coco = COCO(os.path.join(config["ROOT_DIR"], "train", "_annotations.coco.json"))
    categories = coco.cats
    n_classes = len(categories.keys())
    classes = [i[1]['name'] for i in categories.items()]

    # DEFINE MODEL A
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    # model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
    # model = detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # model = detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
    
    # we need to change the head
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    # C = 3 # How many channels are in the input?
    # num_classes = 2 # How many classes are there?
    # n_filters = 32 # How many filters in our backbone?
    # backbone = nn.Sequential(
    #     cnnLayer(C, n_filters),
    #     cnnLayer(n_filters, n_filters),
    #     cnnLayer(n_filters, n_filters),
    #     nn.MaxPool2d((2,2)),
    #     cnnLayer(n_filters, 2*n_filters),
    #     cnnLayer(2*n_filters, 2*n_filters),
    #     cnnLayer(2*n_filters, 2*n_filters),
    #     nn.MaxPool2d((2,2)),
    #     cnnLayer(2*n_filters, 4*n_filters),
    #     cnnLayer(4*n_filters, 4*n_filters),
    # )
    # backbone.out_channels = n_filters*4
    # anchor_generator = AnchorGenerator(sizes=((32),), 
    #                           aspect_ratios=((1.0),))
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7,
    #                                                 sampling_ratio=2)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes,
    #                    image_mean=[0.5], image_std=[0.229],
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)
    
    # define device to use
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # Get all trainable parameters
    if config["TRAIN"]["OPTIMIZER"] == "sgd":
         optimizer = torch.optim.SGD(model.parameters(), lr=config["TRAIN"]["LR"], 
                                     momentum=config["TRAIN"]["LR"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["TRAIN"]["LR"])

    if config["TRAIN"]["LR_SCHEDULER"] == "step":    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["TRAIN"]["STEP_SIZE"], 
                                                    gamma=config["TRAIN"]["GAMMA"])
    else:
        scheduler = None

    # training for n epoch
    experimentNum = config["EXP_ID"]
    num_epochs=config["TRAIN"]["EPOCHS"]
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_load, device, epoch, experimentNum)
        if scheduler:
            scheduler.step()
        print()    
    
