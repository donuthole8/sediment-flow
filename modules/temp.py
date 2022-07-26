import cv2
import numpy as np


def resize_img(in_path, out_path, size):
    """
    画像サイズ変更
    
    in_path: 入力画像パス
    out_path: 出力画像パス
    """
    img = cv2.imread(in_path)
    resize_img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path, resize_img)


def moment():
    """
    モーメントの計算
    """
    # ベース画像の読み込み
    file_name='./inputs/paleblue1.jpeg'
    img=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    h,w=img.shape[:2]
    # 二値化処理
    thresh, img_thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    cv2.imwrite('img_thresh.jpg',img_thresh)
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # 塗りつぶし画像の作成
    black_img=np.zeros((h,w),np.uint8)
    cv2.drawContours(black_img, contours, 0, 255, -1)
    cv2.imwrite('img_thresh2.jpg',black_img)
    # 重心計算
    M = cv2.moments(img_thresh, False)
    x,y= int(M["m10"]/M["m00"]) , int(M["m01"]/M["m00"])
    print('mom=('+str(x)+','+str(y)+')')
    cv2.circle(img, (x,y), 10, 255, -1)
    cv2.imwrite('img_mom.jpg',img)


def save_resize_image(path, image, size):
    """
    画像を縮小して保存

    path: 保存先のパス
    image: 画像データ
    size: 保存サイズ
    """
    # 画像をリサイズ
    resize_img = cv2.resize(
        image, 
        (size[1], size[0]), 
        interpolation=cv2.INTER_CUBIC
    )

    # 画像を保存
    cv2.imwrite(path, resize_img)

    return


def make_mask(path):
    """
    カラー画像（赤く着色された領域）からマスク画像を作成

    path: 画像パス
    """
    # 画像読み込み
    color_mask = cv2.imread(path, cv2.IMREAD_COLOR)

    # 白画像作成
    mask = np.full((color_mask.shape), [255,255,255])

    # 背景画素を透過
    idx = np.where(color_mask == [0,0,0])
    mask[idx] = 0
    # 画像よりマスク部分を抽出
    idx = np.where(color_mask == [0,0,255])
    mask[idx] = 0

    # 二値化
    mask = cv2.cvtColor(mask.astype(np.float32), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask.astype(np.uint8), 128, 255, cv2.THRESH_OTSU)

    # マスク画像を保存
    cv2.imwrite('./inputs_trim/mask.png', mask)


def tif2png(path1, path2):
    img = cv2.imread(path1)

    cv2.imwrite(path2, img)


if __name__ == "__main__":
    # 画像サイズの変更
    # resize_img("../inputs/manual_mask.png", "../inputs/_mask.png")

    # モーメントの計算
    # moment()

    # マスク画像を作成
    # make_mask('./inputs/koyaura_resize_answer.tif')

    # 画像拡張子の変更
    tif2png("../inputs/heli.tif", "../inputs/heli.png")
