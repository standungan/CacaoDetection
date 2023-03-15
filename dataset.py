import os
import yaml
import cv2
import copy
import torch
import albumentations as Albus
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms
from albumentations.pytorch import ToTensorV2

class KakaoDetection(VisionDataset):
    def __init__(self, root, split='train', transforms = None, transform = None, target_transform = None):
        super().__init__(root, transforms, transform, target_transform)

        self.split = split # train valid test
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json"))

        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self.get_target(id)) > 0)]
        self.len = len(self.ids)

    def get_image(self, id:int):
        img_name = self.coco.loadImgs(id)[0]['file_name']
        img_path = os.path.join(self.root, self.split, img_name)
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        return image

    def get_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):

        id = self.ids[index]
        image = self.get_image(id)
        target = self.get_target(id)
        target = copy.deepcopy(self.get_target(id))

        # get all boxes
        boxes = [t['bbox'] + [t['category_id']] for t in target]

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed["image"]
        boxes = transformed["bboxes"]

        new_boxes = [] # xywh to xyxy
        for box in boxes:
            xmin, ymin = box[0], box[1]
            xmax = xmin + box[2]
            ymax = ymin + box[3]
            
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        trgt = {} # transformed target

        trgt['boxes'] = boxes
        trgt['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        trgt['image_id'] = torch.tensor([t['image_id'] for t in target], dtype=torch.int64)
        trgt['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        trgt['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        
        return image.div(255), trgt
    
    def __len__(self) -> int:
        return self.len

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
                ToTensorV2()
            ], bbox_params=Albus.BboxParams(format='coco'))
        else:
            transform = Albus.Compose([
                Albus.Resize(img_size, img_size),
                ToTensorV2()
            ], bbox_params=Albus.BboxParams(format='coco'))

        return transform


    dataset = KakaoDetection(root=config["ROOT_DIR"], split="train", transforms=get_transforms(True))

    img, target = dataset[0]

    print(target)

    print(len(dataset))

