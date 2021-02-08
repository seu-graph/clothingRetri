import numpy as np
import os
import cv2
import shutil
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as SKPCA
import torch
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

use_pca = False
proj_dim = 128

feats_dir = '/home/lcs/clothRetri/features/c2s'

visual = False
visual_save_dir = '/home/lcs/clothRetri/vis/c2s'
query_txt = '/home/lcs/codes/clothingRetri/Consumer_to_Shop/data/query_crop_c2s.txt'
gallery_txt = '/home/lcs/codes/clothingRetri/Consumer_to_Shop/data/gallery_crop_c2s.txt'


def resize_with_as(img, size=(224, 224)):

    sh, sw, _ = img.shape
    if sw > sh:
        start = (sw - sh) // 2
        x = cv2.copyMakeBorder(img, start, start, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        start = (sh - sw) // 2
        x = cv2.copyMakeBorder(img, 0, 0, start, start, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    x1 = cv2.resize(x, size, interpolation=cv2.INTER_AREA)

    return x1


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        feats_dir = sys.argv[1]

    if len(sys.argv) == 3:
        feats_dir = sys.argv[1]
        visual_save_dir = sys.argv[2]

    # print('load query dict...')
    query_dict = np.load(os.path.join(feats_dir, 'query.npz'), allow_pickle=True)
    # query_dict = query_dict.item()
    query_names = query_dict['upc']
    query_feats = query_dict['feat']
    print(len(query_feats), query_feats[0].shape)

    # print('load gallery dict...')
    gallery_dict = np.load(os.path.join(feats_dir, 'gallery.npz'), allow_pickle=True)
    # gallery_dict = gallery_dict.item()
    gallery_names = gallery_dict['upc']
    gallery_feats = gallery_dict['feat']
    print(len(gallery_feats), gallery_feats[0].shape)

    if use_pca:
        print('pca ...')
        pca = SKPCA(n_components=proj_dim, whiten=True)
        gallery_feats = normalize(gallery_feats, norm="l2")
        query_feats = normalize(query_feats, norm="l2")
        # pca.fit(query_feats)
        pca.fit(gallery_feats)
        gallery_feats = pca.transform(gallery_feats)
        query_feats = pca.transform(query_feats)

    print('calculate  dists...')

    start_time = time.time()
    rank_1, rank_5, rank_10, rank_20, rank_30, rank_40, rank_50 = [], [], [], [], [], [], []

    if visual:
        gallery_img_paths = [line.split(',')[0].replace('/crop', '/') for line in open(gallery_txt)]
        query_img_paths = [line.split(',')[0].replace('/crop', '/') for line in open(query_txt)]
        visual_correct = os.path.join(visual_save_dir, 'correct')
        visual_false = os.path.join(visual_save_dir, 'false')
        os.makedirs(visual_correct, exist_ok=True)
        os.makedirs(visual_false, exist_ok=True)


    gallery_feats = [torch.from_numpy(feats).cuda() for feats in [gallery_feats[0], gallery_feats[-1]]]
    for qid, query in enumerate(query_feats[0]):
        dist_list = []
        r_num = 0
        query = torch.from_numpy(np.array([query])).cuda()
        for g_feats in gallery_feats:
            dist = torch.cosine_similarity(query, g_feats, dim=1)
            dist_list.append(dist.cpu().detach().numpy())
            r_num += 1
        q_g_dist = dist_list[0]
        for i in range(1, r_num):
            q_g_dist = np.maximum(q_g_dist, dist_list[i])

        q_name = query_names[qid]
        query_id = q_name

        k = 50
        rank = np.argsort(-q_g_dist)
        top_k_scores, top_k_id = [], []
        for i in range(k):
            top_k_id.append(rank[i])
            top_k_scores.append(q_g_dist[rank[i]])

        top_k = [gallery_names[gid] for gid in top_k_id]

        rank_1.append(1 if query_id == top_k[0] else 0)
        rank_5.append(1 if query_id in top_k[:5] else 0)
        rank_10.append(1 if query_id in top_k[:10] else 0)
        rank_20.append(1 if query_id in top_k[:20] else 0)
        rank_30.append(1 if query_id in top_k[:30] else 0)
        rank_40.append(1 if query_id in top_k[:40] else 0)
        rank_50.append(1 if query_id in top_k[:50] else 0)

        if visual:
            per_size = 224
            font_scale = 2
            rst = np.zeros((per_size * 3, per_size * 3, 3))
            for row in range(3):
                for col in range(3):
                    if row == 0 and col == 0:
                        query_path = query_img_paths[qid]
                        img = cv2.imread(query_path)
                        img = resize_with_as(img)
                    else:
                        k = row * 3 + col - 1
                        img = cv2.imread(gallery_img_paths[rank[k]])
                        img = resize_with_as(img)
                        if query_id == top_k[k]:
                            cv2.rectangle(img, (2, 2), (221, 221), (0, 255, 0), 2)
                    rst[row * per_size:(row + 1) * per_size, col * per_size:(col + 1) * per_size] = img
            if query_id != top_k[0]:
                save_path = os.path.join(visual_false, str(qid) + '.jpg')
            else:
                save_path = os.path.join(visual_correct, str(qid) + '.jpg')
            cv2.imwrite(save_path, rst)


    print(time.time() - start_time)
    print('============== Ranking results ==============')
    print('Ranking 1  accuracy: {:.4f}'.format(np.sum(rank_1) / len(rank_1)))
    print('Ranking 5  accuracy: {:.4f}'.format(np.sum(rank_5) / len(rank_5)))
    print('Ranking 10 accuracy: {:.4f}'.format(np.sum(rank_10) / len(rank_10)))
    print('Ranking 20 accuracy: {:.4f}'.format(np.sum(rank_20) / len(rank_20)))
    print('Ranking 30 accuracy: {:.4f}'.format(np.sum(rank_30) / len(rank_30)))
    print('Ranking 40 accuracy: {:.4f}'.format(np.sum(rank_40) / len(rank_40)))
    print('Ranking 50 accuracy: {:.4f}'.format(np.sum(rank_50) / len(rank_50)))




