# clothingRetri
Codes for paper "Clothing Retrieval Based on Deep Metric Learning"

### Results on Deepfashion2
Recall@K on [In-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html):

|  R@1   | R@10  | R@20  | R@50  |
| :----: | :----: |:----: | :----: |
|  91.8  |  98.7 |  99.2  |  99.6 |

Recall@K on [Consumer-to-Shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/Consumer2ShopRetrieval.html):

| Remarks |  R@1   | R@20  | R@50  |
| :----: | :----: |:----: | :----: |
| Ours |  27.6  |  64.5  |  74.9 |
| Ours+TTA|  28.2  |  65.3  |  75.5 |

### Pre-trained models & Extracted features
[Download(Baidu NetDisk)](https://pan.baidu.com/s/1hyWOEIh2Sifomzs8HPxV0g) (code: 6eih)

### Testing 
1. Clone this repository
    ```Shell
    git clone https://github.com/seu-graph/clothingRetri.git
    ```
   Then download Pre-trained models.
2. Download [Consumer-to-Shop](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/Consumer2ShopRetrieval.html) in Deepfashion2 Datasets
3. Edit ``` Consumer_to_Shop/crop_by_bbox_list.py ```. Modify paths to yours, Then run this code to crop clothes by bboxs.
4. Edit ``` Consumer_to_Shop/split_query_gallery.py ```. Modify paths to yours, Then run this code to generate splited files of query(Consumer) and gallery(Shop).
5. Edit ``` Consumer_to_Shop/extract_features_aug.py ```. Modify paths to yours, Then run this code to extract features of query(Consumer) and gallery(Shop).
   We also provide our extracted features, which can be downloaded from [here(Baidu NetDisk)](https://pan.baidu.com/s/1hyWOEIh2Sifomzs8HPxV0g) (code: 6eih)
6. Edit ``` Consumer_to_Shop/eval_c2s.py ```. Modify paths to yours, Then run this code to reproduce our results.
   What's more, you can set ``` visual = True ``` to save visual results.
