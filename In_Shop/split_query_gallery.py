import os
import shutil


partition_file = '/media/lcs/ubtdata/lcsDataset/DeepFashion2/In-Shop/list_eval_partition.txt'
inshop_root = '/media/lcs/ubtdata/lcsDataset/DeepFashion2/In-Shop'
gallery_txt = '/home/lcs/codes/clothingRetri/In_Shop/data/gallery_inshop.txt'
query_txt = '/home/lcs/codes/clothingRetri/In_Shop/data/query_inshop.txt'

fout_gallery = open(gallery_txt, 'w')
fout_query = open(query_txt, 'w')

for line in open(partition_file):
    if line.startswith('img'):
        img_path, id, split = line.strip().split()
        if split == 'train':
            continue
        img_path = os.path.join(inshop_root, img_path)
        if split == 'query':
            fout_query.write(img_path + ',' + id + '\n')
        elif split == 'gallery':
            fout_gallery.write(img_path + ',' + id + '\n')

fout_gallery.close()
fout_query.close()




