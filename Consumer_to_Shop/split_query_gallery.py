import os

query_txt = '/home/lcs/codes/clothingRetri/Consumer_to_Shop/data/query_crop_c2s.txt'
gallery_txt = '/home/lcs/codes/clothingRetri/Consumer_to_Shop/data/gallery_crop_c2s.txt'

split_txt = '/media/lcs/ubtdata/lcsDataset/DeepFashion2/Consumer-to-shop/list_eval_partition.txt'
crop_root = '/media/lcs/ubtdata/lcsDataset/DeepFashion2/Consumer-to-shop/crop'

query_out = open(query_txt, 'w')
gallery_out = open(gallery_txt, 'w')

gallery_set = set()
for line in open(split_txt):
    if line.startswith('img/'):
        c, s, l, p = line.strip().split()
        if p == 'test':
            query_out.write(os.path.join(crop_root, c) + ',' + l + '\n')
            ggg = os.path.join(crop_root, s) + ',' + l
            if ggg not in gallery_set:
                gallery_out.write(ggg + '\n')
                gallery_set.add(ggg)
            else:
                print(ggg)
query_out.close()
gallery_out.close()
