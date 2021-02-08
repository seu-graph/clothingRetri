import os
import cv2


save_root = '/media/lcs/ubtdata/lcsDataset/DeepFashion2/Consumer-to-shop/crop'
img_root = '/media/lcs/ubtdata/lcsDataset/DeepFashion2/Consumer-to-shop/'
anno_path = '/media/lcs/ubtdata/lcsDataset/DeepFashion2/Consumer-to-shop/list_bbox_consumer2shop.txt'

os.makedirs(save_root, exist_ok=True)

cnt = 0
for line in open(anno_path):
    if line.startswith('img/'):
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        image_name, clothes_type, pose_type, x_1, y_1, x_2, y_2 = line.strip().split()
        x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)
        img = cv2.imread(os.path.join(img_root, image_name))
        img = img[y_1: y_2, x_1: x_2, :]
        save_path = os.path.join(save_root, image_name)
        save_dir = os.path.split(save_path)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path, img)

