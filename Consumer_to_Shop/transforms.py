import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import random
import math

BORDER_VALUE = (255, 255, 255)


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


class RandomErasing:
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1 / r1)

    def __call__(self, img):
        """
        perform random erasing
        Args:
            img: opencv numpy array in form of [w, h, c] range
                 from [0, 255]
        Returns:
            erased img
        """
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'
        if random.random() > self.p:
            while True:
                Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r)
                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))
                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])
                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))
                    break
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img


class RandomRotation:
    def __init__(self, p=0.5, max_angle=180):
        self.p = p
        self.max_angle = max_angle

    def __call__(self, im):
        """
        perform random rotation.
        Args:
            img: opencv numpy array in form of [w, h, c] range
                 from [0, 255]
        Returns:
            rotated img
        """
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        assert len(im.shape) == 3, 'image should be a 3 dimension numpy array'
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        img_h, img_w = im.shape[:2]
        angle = np.random.randint(0, self.max_angle + 1)
        if np.random.uniform() > 0.5:
            angle = -angle
        rotate_m = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)  # default anti-clock
        cos_m, sin_m = np.abs(rotate_m[0, 0]), np.abs(rotate_m[0, 1])
        r_w = int((img_h * sin_m) + (img_w * cos_m))
        r_h = int((img_h * cos_m) + (img_w * sin_m))
        # position bias
        rotate_m[0, 2] += r_w / 2 - img_w / 2
        rotate_m[1, 2] += r_h / 2 - img_h / 2
        # img_r = cv2.warpAffine(im, rotate_m, (r_w, r_h), borderValue=(104.0, 117.0, 123.0))
        img_r = cv2.warpAffine(im, rotate_m, (r_w, r_h), borderValue=BORDER_VALUE)
        img_r = Image.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
        return img_r


def build_transforms(origin_size=256, crop_size=224, is_train=True, crop_scale=[0.2, 1]):
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if is_train:
        transform = T.Compose(
            [
                # T.ToPILImage(mode=None),
                # T.Resize(size=(crop_size, crop_size)),
                T.RandomHorizontalFlip(p=0.5),
                RandomRotation(p=0.5, max_angle=60),
                #RandomErasing(sl=0.02, sh=0.4, r1=0.3),
                PAD(),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                # T.ToPILImage(mode=None),
                # T.Resize(size=(crop_size, crop_size)),
                PAD(),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
