import torchvision.transforms as T


def build_transforms(origin_size=256, crop_size=224, is_train=True, crop_scale=[0.2, 1]):
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if is_train:
        transform = T.Compose(
            [
                # T.ToPILImage(mode=None),
                T.Resize(size=origin_size),
                T.RandomResizedCrop(
                    scale=crop_scale, size=crop_size
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                # T.ToPILImage(mode=None),
                T.Resize(size=origin_size),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
