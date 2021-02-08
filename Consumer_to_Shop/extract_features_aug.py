import os
import os.path as osp
import re
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
import torch
from resnet import ResNet50
from transforms import build_transforms
import torch.nn as nn
import time
import numpy as np
import argparse
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import random
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
BORDER_VALUE = (255, 255, 255)


def read_image(img_path, mode="RGB"):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            if mode == "BGR":
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))
            got_img = True
        except IOError:
            print(
                f"IOError incurred when reading '{img_path}'. Will redo. Don't worry. Just chill."
            )
            pass
    return img


class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, transforms=None, mode="RGB"):
        self.mode = mode
        self.transforms = transforms
        self.root = osp.dirname(img_source)
        assert osp.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, "r") as f:
            for line in f:
                _path, _label = re.split(r",", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = osp.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label, index


class Rotate_Bound_White_Bg:
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, image):
        if not self.angle:
            return image
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        image = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        return image



class PAD:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, x):
        x = cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR)
        sh, sw, _ = x.shape
        if sw > sh:
            start = (sw - sh) // 2
            x = cv2.copyMakeBorder(x, start, start, 0, 0, borderType=cv2.BORDER_CONSTANT, value=BORDER_VALUE)
        else:
            start = (sh - sw) // 2
            x = cv2.copyMakeBorder(x, 0, 0, start, start, borderType=cv2.BORDER_CONSTANT, value=BORDER_VALUE)
        x = cv2.resize(x, self.size)
        x = Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        return x


def feat_extractor(model, data_loader, logger=None):
    model.eval()
    feats = list()
    if logger is not None:
        logger.info("Begin extract")
    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()
        if i % 10 == 0:
            print(i, '/', len(data_loader))
        with torch.no_grad():
            # out = model(imgs)[1].data.cpu().numpy()
            out = model(imgs).data.cpu().numpy()
            feats.append(out)

        if logger is not None and (i + 1) % 100 == 0:
            logger.debug(f"Extract Features: [{i + 1}/{len(data_loader)}]")
        del out
    feats = np.vstack(feats)
    return feats


def extract(img_source, model_path=None, save_path=None, head_dim=256, bs=256, backbone='res50'):
    split = 'gallery'
    if 'query' in img_source:
        split = 'query'

    if backbone == 'res50':
        model = ResNet50(head_dim=head_dim)
    model = nn.DataParallel(model.cuda(), device_ids=device_ids)
    if model_path is not None:
        state_dict = torch.load(model_path)['state_dict']
        for key in list(state_dict.keys()):
            if key.startswith('module.classifier'):
                del state_dict[key]
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("model successfully loaded")

    angle_list = [0]
    all_feats = []
    if split == 'gallery':
        angle_list = [0, 180]

    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    for angle in angle_list:
        transforms = T.Compose(
                [
                    Rotate_Bound_White_Bg(angle=angle),
                    PAD(),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
        if angle == 180:
            transforms = T.Compose(
                [
                    T.RandomHorizontalFlip(p=1.0),
                    PAD(),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )

        dataset = BaseDataSet(img_source, transforms=transforms)
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=bs,
            pin_memory=True,
        )

        labels = dataset.label_list
        feats = feat_extractor(model, data_loader)
        all_feats.append(feats)

    if save_path:
        if not osp.exists(save_path):
            os.makedirs(save_path)
        npz_path = f"{save_path}/{split}.npz"
    else:
        npz_path = f"output/{split}.npz"
    np.savez(npz_path, feat=all_feats, upc=labels)
    print(f"FEATS : \t {npz_path}")


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Extract feature")
    parser.add_argument(
        "--o", dest="save_dir", help="config file",
        default='/home/lcs/clothRetri/features/c2s',
        type=str
    )
    parser.add_argument(
        "--model", dest="model_path", help="model path file",
        default= "/home/lcs/codes/weights/best_consumer2shop.pth",
        type=str
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    head_dim = 128
    backbone = 'res50'
    gallery = '/home/lcs/codes/clothingRetri/Consumer_to_Shop/data/gallery_crop_c2s.txt'
    query = '/home/lcs/codes/clothingRetri/Consumer_to_Shop/data/query_crop_c2s.txt'
    extract(img_source=gallery, model_path=args.model_path, save_path=args.save_dir, bs=256, head_dim=head_dim, backbone=backbone)
    extract(img_source=query, model_path=args.model_path, save_path=args.save_dir, bs=256, head_dim=head_dim, backbone=backbone)
