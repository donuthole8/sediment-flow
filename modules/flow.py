import cv2
import csv
import math
import numpy as np

# from modules import tif
from modules import tool
from modules import operation
from modules import process
from modules import temp


# path1 = './inputs_re/dsm_uav.tif'
# path2 = './inputs_re/degree.tif'
# path3 = './inputs_re/manual_mask.png'
# path4 = './inputs_re/uav_img.tif'

# path1 = './inputs_trim/dsm_uav_re.tif'

path1 = './outputs/normed_uav_dsm2.tif'
path2 = './inputs_trim/degree.tif'
path3 = './inputs_trim/manual_mask.png'
path4 = './inputs_trim/uav_img.tif'


def detect_flow(deg: float) -> tuple(int, int):
	"""
	傾斜方向を画素インデックスに変換

	deg: 角度
	"""

	# 注目画素からの移動画素
	dx, dy = 0, 0
	if   (math.isnan(deg)):
		return np.nan, np.nan
	elif (deg > 337.5) or  (deg <= 22.5):
		dx, dy = 0, -1
	elif (deg > 22.5)  and (deg <= 67.5):
		dx, dy = 1, -1
	elif (deg > 67.5)  and (deg <= 112.5):
		dx, dy = 1, 0
	elif (deg > 112.5) and (deg <= 157.5):
		dx, dy = 1, 1
	elif (deg > 157.5) and (deg <= 202.5):
		dx, dy = 0, 1
	elif (deg > 202.5) and (deg <= 247.5):
		dx, dy = -1, 1
	elif (deg > 247.5) and (deg <= 292.5):
		dx, dy = -1, 0
	elif (deg > 292.5) and (deg <= 337.5):
		dx, dy = -1, -1

	return dx, dy


def estimate_flow(dsm: np.ndarray, deg:np.ndarray, img:np.ndarray) -> None:
	"""
	流出方向の予測

	dsm: 標高データ
	deg: 傾斜方向データ
	img: 画像データ
	"""
	# 領域データ読み込み
	with open("./area_data/region.csv", encoding='utf8', newline='') as f:
	# with open("./area_data/l-centroid.csv", encoding='utf8', newline='') as f:
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
		## 領域内での→を付けたい！！
		## 下端まで〜

		# 始点
		x, y = cx, cy

		# 傾斜方向の標高
		# NOTE: しきい値変えれる
		for i in range(0, 30):
		# for i in range(0, 10):
			# 注目領域の重心標高から傾斜方向を探査
			dx, dy = detect_flow(deg[cy, cx])
			# 終点
			x, y = x + dx, y + dy

			# nanだった場合
			if (math.isnan(dx)):
				break
			# 標高が上がった場合
			try:
				if ((dsm[y, x]) > pix):
					break
			except:
				print("err")
				if ((dsm[y-1, x-1]) > pix):
					break
		try:
			# 矢印の距離が短すぎる領域は除去
			# NOTE: しきい値変えれる
			if (i > 7):
				# 矢印の描画
				cv2.arrowedLine(
					img=img,            # 画像
					pt1=(cx, cy),       # 始点
					pt2=(x, y),         # 終点
					color=(20,20,180),  # 色
					thickness=3,        # 太さ
					tipLength=0.3       # 矢先の長さ
				)
		except:
			pass
	
	cv2.imwrite("estimate_map.png", img)

	return


@tool.stop_watch
def main() -> None:
	"""
	Flow-R・中山さん手法による土砂流出予想

	ⅰ. すべての斜面崩壊領域を処理対象としてfor文を回す
	ⅱ. 注目画素から傾斜方向 + 隣接2方向の画素の標高値を調べる（傾斜方向はQgisの出力を近傍8画素に割り当て,　多分↑が0°になってるので上方向は 337.5° ~ 360° & 0° - 22.5°みたいな？）
	ⅲ. 標高値が注目画素よりも小さければそこも斜面崩壊領域としてもう一回ⅱをやる
	を繰り返してたと思う。（もとは傾斜とか条件つけてたけど全然領域が広がらなかったので標高の大小だけにした気がする）
	正直①のほうに時間かけすぎてかなり適当なので、一番直すべきだったのがここかも
	"""
	# 画像の読み込み
	dsm  = cv2.imread(path1, cv2.IMREAD_ANYDEPTH).astype(np.float32)
	deg  = cv2.imread(path2, cv2.IMREAD_ANYDEPTH).astype(np.float32)
	# dsm  = tif.load_tif(path1).astype(np.float32)
	# deg  = tif.load_tif(path2).astype(np.float32)
	mask = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
	org_img = cv2.imread(path4)

	# クラス初期化
	image_op = operation.ImageOp(path_list)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	dsm, _, _, deg, mask = operation.resampling_dsm()

	# 次元を減らす
	dsm = cv2.split(dsm)[0]
	deg = cv2.split(deg)[0]

	# 傾斜方位の正規化（0-255 -> 0-360）
	print("# 傾斜方向の正規化")
	deg = operation.norm_degree_v2(deg)

	# 土砂領域以外の除去
	# print("# 植生領域の除去")
	# img = process.remove_vegitation(img)
	# img = process.remove_black_pix(org_img, "./outputs/vegitation.png")

	# 土砂マスク
	print("# 土砂マスクによる土砂領域抽出")
	img = operation.apply_mask(org_img, mask)

	# カラー画像の領域分割
	print("# 土砂領域の領域分割")
	img = operation.divide_area(img, 3, 4.5, 100)
	# img = cv2.imread("./outputs/meanshift.png")

	# 輪郭・重心データ抽出
	print("# 領域分割・土砂マスク済み画像から輪郭データ抽出")
	operation.calc_contours((img.shape[0], img.shape[1]))

	# # 斜面崩壊領域データをすべて抽出
	# print("# 土砂マスク中の領域のみでラベリング")
	# # driver.labeling_color_v1(mask, img)
	# driver.labeling_bin(mask, img)
	# # driver.extract_region(img, regions_num)




	# 流出推定
	estimate_flow(dsm, deg, org_img)
