import cv2
import numpy as np
import pymeanshift as pms
from PIL import Image
from collections import deque
import time
from functools import wraps

import csv


def stop_watch(func):
    """
    関数の実行時間測定
    """
    @wraps(func)
    def wrapper(*args, **kargs) :
        start = time.time()
        result = func(*args, **kargs)
        process_time =  time.time() - start
        print(f"-- {func.__name__} : {int(process_time)}[sec] / {int(process_time/60)}[min]")
        return result
    return wrapper

# def meanshift(rgb_img, spatial_radius, range_radius, min_density):
#     lab_img, labels, num_seg = pms.segment(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2Lab), spatial_radius, range_radius,min_density)
#     new_rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
#     return (new_rgb_img, labels, num_seg)

def contrastize(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    new_v = clahe.apply(v)

    hsv_clahe = cv2.merge((h, s, new_v))
    new_rgb_img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    return new_rgb_img

def cluster(contrast_img):
    img = Image.fromarray(contrast_img)
    img_q = img.quantize(colors=128, method=0, dither=1)
    a = np.asarray(img_q)
    return np.stack([a]*3, axis=2)

def approximation(pix1, pix2):
    dif = abs(pix1 - pix2)
    dv = 10
    return (dif < dv).all()

def neighbors(idx, lim):
    w, h, _ = lim
    i, j = idx
    return sorted([
            (i + n, j + m)
            for n in range(-1, 2)
            for m in range(-1, 2)
            if not (n == 0 and m == 0) and i + n >= 0 and j + m >= 0 and i + n < w and j + m < h
            ])


def relabel(dummy, src, idx, labels, label):
    q = deque([idx])
    while len(q) > 0:
        idx = q.popleft()
        # labels[idx] = ((label * 5) % 255, (label * 10) % 255, (label * 30) % 255)
        labels[idx] = label
        dummy[idx] = label
        ns = neighbors(idx, src.shape)
        q.extendleft(n for n in ns if approximation(src[n], src[idx]) and dummy[n] == 0)


def extract_sediment(img, mask):
    """
    標高データから土砂領域を抽出

    img: 抽出対象の画像 
    mask: 土砂領域マスク
    """
    # 土砂マスクを用いて土砂領域以外を除去
    sed = np.zeros(img.shape).astype(np.uint8)
    idx = np.where(mask == 0)
    sed[idx] = img[idx]

    # # 土砂領域を抽出
    # idx = np.where(mask == 0).astype(np.float32)
    # # 土砂領域以外の標高データを消去
    # sed[idx] = np.nan
    # tif.save_tif(dsm, "./inputs/dsm_uav.tif", "./outputs/sediment.tif")
    # return sed.astype(np.uint8)

    return sed

@stop_watch
def label(filename):
    src = cv2.imread(filename)

    mask = cv2.imread("../inputs/manual_mask.png")

    if src is None:
        raise Exception
    print("start")
    # (img_shifted, labels, num_seg) = meanshift(src, 12, 3, 200)

    # 土砂マスクを用いて土砂領域以外を除去
    # TODO:::::::
    src = extract_sediment(src, mask)

    img_cont = contrastize(src)
    print("cont")
    
    img_cluster = cluster(img_cont)
    print("clust")

    h, w, c = src.shape
    dummy = np.zeros((h, w), dtype=int)
    # labels = np.zeros((h, w, c), dtype=int)
    labels = np.zeros((h, w), dtype=int)
    label = 1

    # with open('./color_inf.csv', 'w') as f:
        # writer = csv.writer(f)

    it = np.nditer(img_cluster, flags=['multi_index'], op_axes=[[0, 1]])
    for n in it:
        if dummy[it.multi_index] == 0:
            # relabel(dummy, img_cluster, it.multi_index,labels, label)
            relabel(dummy, img_cluster, it.multi_index,labels, label)
            label += 1

    # return label, labels.astype(np.uint8)


            # writer.writerow([label * 5, label * 10, label * 30])
    print("label_num", label)
    # NOTE: label_num 50288


    np.savetxt('dummy.txt', dummy.astype(np.uint8),fmt='%d')
    np.savetxt('label.txt', labels.astype(np.uint8),fmt='%d')

    with open('./labeee.csv', 'w') as f:
        writer = csv.writer(f)
        for ll in labels:
            writer.writerow(ll)

    

    cv2.imwrite('outputs/dummy.png', dummy.astype(np.uint8))
    # cv2.imwrite('result.png', labels.astype(np.uint8))
    cv2.imwrite('result.png', labels.astype(np.uint8))


import csv

@stop_watch
def test(max_label , img):
    label = 1

    img = cv2.imread("../outputs/___meanshift.png")
    # img = cv2.imread("result_.png")
    # img = cv2.imread("result.png")

    print("max_label:::::::", max_label)


    with open('./labeling_inf.csv', 'w') as f:
        writer = csv.writer(f)
        
        # ヘッダを追加
        columns_list = ['id', 'area', 'x_centroid', 'y_centroid']
        writer.writerow(columns_list)

        count = 0
        
        # while (True):
        for i in range(max_label):
            # color = [label * 5, label * 10, label * 30]
            # color = (label * 5, label * 10, label * 30)
            color = ((label * 5)%255, (label * 10)%255, (label * 30)%255)

            mask = np.where(img == color, 255, 0).astype(np.uint8)
            # mask = cv2.cvtColor(mask.astype(np.float32), cv2.COLOR_BGR2GRAY)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

            for j in range(retval):
                print(centroids)
                writer.writerow([
                    count, 
                    stats[j][4], 
                    int(centroids[j][0]),
                    int(centroids[j][1]),
                    stats[j][2],
                    stats[j][3],
                ])
                count += 1
                
            label += 1
            

    print("label:::", label)
    print("count:::", count)







if __name__ == '__main__':
    # label('Lenna_big.png')
    max_label ,labels = label("../outputs/___meanshift.png")
    # max_label ,labels = label("../inputs/hage.png")


    # test(max_label ,labels)