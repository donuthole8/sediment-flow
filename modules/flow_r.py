import cv2
import csv
import numpy as np

from modules import tif
from modules import tool
from modules import driver
from modules import process
from modules import test_code


path1 = './inputs_re/dsm_uav.tif'
path2 = './inputs_re/degree.tif'
path3 = './inputs_re/manual_mask.png'
path4 = './inputs_re/uav_img.tif'


def detect_flow(deg):
	"""
	傾斜方向を画素インデックスに変換

	deg: 角度
	"""
	print("deg",deg)

	# 注目画素からの移動画素
	dx, dy = 0, 0

	if   (deg > 337.5) or  (deg <= 22.5):
		dx, dy = 0, 1
	elif (deg > 22.5)  and (deg <= 67.5):
		dx, dy = 1, 1
	elif (deg > 67.5)  and (deg <= 112.5):
		dx, dy = 1, 0
	elif (deg > 112.5) and (deg <= 157.5):
		dx, dy = 1, -1
	elif (deg > 157.5) and (deg <= 202.5):
		dx, dy = 0, -1
	elif (deg > 202.5) and (deg <= 247.5):
		dx, dy = -1, -1
	elif (deg > 247.5) and (deg <= 292.5):
		dx, dy = -1, 0
	elif (deg > 292.5) and (deg <= 337.5):
		dx, dy = -1, 1

	return dx, dy



def estimate_flow(dsm, deg):
	"""
	流出方向の予測
	
	deg: 傾斜方向データ
	"""
	# 領域データ読み込み
	with open("./area_data/l-centroid.csv", encoding='utf8', newline='') as f:
		area = csv.reader(f)
		area_list = [a for a in area]
		# ヘッダを削除
		area_list.pop(0)
		# 背景領域を削除
		area_list.pop(0)
	
	## 斜面崩壊領域を処理対象としてforを回す
	for area in area_list:
		## 注目画素から傾斜方向 + 隣接2方向の画素の標高値を調べる（傾斜方向はQgisの出力を近傍8画素に割り当て,　多分↑が0°になってるので上方向は 337.5° ~ 360° & 0° - 22.5°みたいな？）

		# 注目領域の重心標高
		cx, cy = int(area[2]), int(area[3])
		pix = dsm[cy, cx]


		## 標高値が注目画素よりも小さければそこも斜面崩壊領域としてもう一回ⅱをやるを繰り返してたと思う。（もとは傾斜とか条件つけてたけど全然領域が広がらなかったので標高の大小だけにした気がする）

		area_id_list = []

		# 傾斜方向の標高
		while True:
			# 注目領域の重心標高から傾斜方向を探査
			dx, dy = detect_flow(deg[cy, cx])

			if (dsm[cy + dy, cx + dx]) > pix:
				continue
			
			



		# downstread_pix = dsm[]


		sub_elevation = dsm_sub[cy, cx][0]



def main():
	"""
	Flow-R・中山さん手法による土砂流出予想

	ⅰ. すべての斜面崩壊領域を処理対象としてfor文を回す
	ⅱ. 注目画素から傾斜方向 + 隣接2方向の画素の標高値を調べる（傾斜方向はQgisの出力を近傍8画素に割り当て,　多分↑が0°になってるので上方向は 337.5° ~ 360° & 0° - 22.5°みたいな？）
	ⅲ. 標高値が注目画素よりも小さければそこも斜面崩壊領域としてもう一回ⅱをやる
	を繰り返してたと思う。（もとは傾斜とか条件つけてたけど全然領域が広がらなかったので標高の大小だけにした気がする）
	正直①のほうに時間かけすぎてかなり適当なので、一番直すべきだったのがここかも
	"""
	# 画像の読み込み
	dsm  = tif.load_tif(path1).astype(np.float32)
	deg  = tif.load_tif(path2).astype(np.float32)
	mask = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
	img  = cv2.imread(path4)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	dsm, _, _, deg, mask = driver.resampling_dsm(dsm, dsm, deg, deg, mask)

	# 土砂領域以外の除去
	print("# 植生領域の除去")
	# img = process.remove_vegitation(img)
	img = process.remove_black_pix(img, "./outputs/vegitation.png")

	# カラー画像の領域分割
	print("# 土砂領域の領域分割")
	img, regions_num = driver.divide_area(img, 3, 4.5, 100)
	# img = cv2.imread("./outputs/meanshift.png")

	# 斜面崩壊領域データをすべて抽出
	print("# 土砂マスク中の領域のみでラベリング")
	# driver.labeling_color_v1(mask, img)
	driver.labeling_bin(mask, img)
	# driver.extract_region(img, regions_num)

	# 流出推定
	estimate_flow(dsm, deg)
