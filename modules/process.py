import enum
from re import A
import cv2
import csv
import numpy as np
from math import dist
from collections import deque
import scipy.ndimage as ndimage

from modules import tif
from modules import tool


def binarize(src_img, thresh, mode):
	"""
	二値化
	"""
	gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
	bin_img = cv2.threshold(gray_img, thresh, 255, mode)[1]

	return bin_img


def morphology(mask, ksize, exe_num):
	"""
	モルフォロジー処理

	mask: 処理対象のマスク画像
	ksize: カーネルサイズ
	exe_num: 実行回数
	"""
	# モルフォロジー処理によるノイズ除去
	kernel = np.ones((ksize, ksize), np.uint8)
	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	for i in range(1,exe_num):
		opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	for i in range(1,exe_num):
		closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

	# クロージング処理による建物領域の穴埋め
	closing = cv2.dilate(closing, kernel, iterations = 1)
	closing = cv2.dilate(closing, kernel, iterations = 1)
	closing = cv2.erode (closing, kernel, iterations = 1)
	closing = cv2.erode (closing, kernel, iterations = 1)

	# 画像の保存
	height, width = closing.shape[:2]
	tool.save_resize_image("./outputs/closing.png", closing, (int(height/5), int(width/5)))
	tool.save_resize_image("./outputs/opening.png", opening, (int(height/5), int(width/5)))

	return closing


def get_norm_contours(mask, scale, ksize):
	"""
	輪郭をぼやけさせて抽出

	mask: マスク画像
	scale: 拡大倍率
	ksize: カーネルサイズ
	"""
	# 画像の高さと幅を取得
	h, w = mask.shape
	# 拡大することで輪郭をぼけさせ境界を識別しやすくする
	img_resize = cv2.resize(mask, (w * scale, h * scale))

	# ガウシアンによるぼかし処理
	img_blur = cv2.GaussianBlur(img_resize, (ksize,ksize), 0)

	# 二値化と大津処理
	r, dst = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# モルフォロジー膨張処理
	kernel = np.ones((ksize,ksize), np.uint8)
	dst = cv2.dilate(dst, kernel, iterations = 1)

	# 画像欠けがあった場合に塗りつぶし
	dst_fill = ndimage.binary_fill_holes(dst).astype(int) * 255

	# 輪郭抽出
	contours, _ = cv2.findContours(dst_fill.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

	return contours, dst_fill


def get_contours(mask, scale):
	"""
	輪郭を抽出

	mask: マスク画像
	scale: 拡大倍率
	ksize: カーネルサイズ
	"""
	# 画像の高さと幅を取得
	h, w = mask.shape
	# 拡大することで輪郭をぼけさせ境界を識別しやすくする
	img_resize = cv2.resize(mask, (w * scale, h * scale))

	# 二値化と大津処理
	r, dst = cv2.threshold(img_resize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# 輪郭抽出
	contours, _ = cv2.findContours(dst.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

	return contours, dst


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


@tool.stop_watch
def extract_neighbor():
	"""
	隣接している領域の組を全て抽出
	"""
	# 領域データ読み込み
	with open("./area_data/l-centroid.csv", encoding='utf8', newline='') as f:
		area = csv.reader(f)
		area_list = [a for a in area]
		# ヘッダを削除
		area_list.pop(0)
		# 背景領域を削除
		area_list.pop(0)

	# 領域の最短距離を算出
	neighbor_idx_list = []
	for area in area_list:
		# 注目領域の重心
		cx, cy = int(area[2]), int(area[3])

		# 各重心との距離を求める
		dist_list = [
			dist((cy, cx), (int(cent[3]), int(cent[2]))) 
			for cent in area_list
		]

		# dist_list = []
		# for _area in area_list:
		#   # 重心
		#   _cx, _cy = int(_area[2]), int(_area[3])
		#   # 2点間距離
		#   dis = dist((cx,cy), (_cx,_cy))
		#   # リストに追加
		#   dist_list.append(dis)

		# 距離がピクセル距離の閾値以下の領域を全て抽出
		idx_list = [
			idx for idx, d in enumerate(dist_list) 
			# if ((d > 5) and (d <= 10))
			# if ((d <= 10))
			if ((d > 7) and (d <= 20))
			# if ((d > 5) and (d <= 30))
			# if (d <= 30)
		]

		# リストに追加
		neighbor_idx_list.append(idx_list)

	return neighbor_idx_list


@tool.stop_watch
def extract_direction(deg, dem, dsm):
	"""
	傾斜方向が上流から下流の領域の組を全て抽出
	
	deg: 傾斜方向データ
	dem: 国土地理院DEM
	dsm: UAV画像のDSM
	"""
	# 傾斜方向データでやりたい
	# 領域データ読み込み
	with open("./area_data/l-centroid.csv", encoding='utf8', newline='') as f:
		area = csv.reader(f)
		area_list = [a for a in area]
		# ヘッダを削除
		area_list.pop(0)
		# 背景領域を削除
		area_list.pop(0)

	# 領域の重心標高を算出
	ave_elevation_list = []
	for i, area in enumerate(area_list):

		# 重心の標高値を算出
		cx, cy = int(area[2]), int(area[3])
		# FIXME: DEMかDSMかは要検討
		centroid_elevation = dem[cy, cx]

		# リストに追加
		ave_elevation_list.append(centroid_elevation[0])

	# 重心標高値より下流の領域を保持する
	downstream_idx_list = []
	for i, area in enumerate(area_list):
		# 平均標高が小さい領域のインデックスを全て抽出
		# downstream = [idx for idx, r in enumerate(ave_elevation_list) if (ave_elevation_list[i] > r)]

		# 注目画素の標高 
		target_elevation = ave_elevation_list[i]
		
		downstream = [
			idx for idx, ave in enumerate(ave_elevation_list) 
			if (target_elevation > ave)
		]

		# リストに追加
		downstream_idx_list.append(downstream)

	return downstream_idx_list


@tool.stop_watch
def extract_sub(dsm_sub):
	"""
	侵食と堆積の領域の組を全て抽出

	dsm_sub: 災害前後の標高差分
	"""
	# 領域データ読み込み
	with open("./area_data/l-centroid.csv", encoding='utf8', newline='') as f:
		area = csv.reader(f)
		area_list = [a for a in area]
		# ヘッダを削除
		area_list.pop(0)
		# 背景領域を削除
		area_list.pop(0)

	# 領域の最短距離を算出
	sub_idx_list = []
	for area in area_list:
		# 注目領域の重心
		cx, cy = int(area[2]), int(area[3])

		# 注目領域の重心の標高変化
		sub_elevation = dsm_sub[cy, cx][0]

		# 堆積・侵食の組み合わせを算出
		idx_list = [
			idx for idx, sub in enumerate(area_list)
			if ((sub_elevation > dsm_sub[int(sub[3]), int(sub[2])][0]))
		]

		# リストに追加
		sub_idx_list.append(idx_list)


	# # 領域の最短距離を算出
	# sub_idx_list = []
	# for area in area_list:
	#   # 注目領域の重心
	#   cx, cy = int(area[2]), int(area[3])

	#   # 注目領域の重心の標高変化
	#   sub_elevation = dsm_sub[cy, cx][0]

	#   # 堆積・侵食の組み合わせを算出
	#   idx_list = [
	#     idx for idx, sub in enumerate(area_list)
	#     if ((sub_elevation * dsm_sub[int(sub[3]), int(sub[2])][0]) < 0)
	#   ]

		# # リストに追加
		# sub_idx_list.append(idx_list)

	return sub_idx_list


# TODO: クラス化してcsv読み込みをselfにする
# ラベリングの改良
# ラベリングについて領域サイズを一定に
# 領域同士が隣接している領域を輪郭データ等で算出
# 傾斜方向が上から下である領域を平均標高値や傾斜方向で算出
# 建物領域にも矢印があるので除去など


# def calc_horizontal(move_list, area_list, img):
#   """
#   水平移動距離を計算

#   move_list: 土砂移動データ
#   area_list: 領域データ
#   img: 地図画像
#   """
#   # 解像度（m）
#   resolution = 0.075

#   for i, move in enumerate(move_list):
#     if (move != []):
#       # 注目領域の重心座標
#       cx, cy = int(area_list[i][2]), int(area_list[i][3])

#       # 土砂の流出方向へ→を描画
#       for m in move:
#         # 注目領域の重心座標





	


# def calc_vertical(move_list, area_list, img):
#   """
#   垂直移動距離を計算

#   move_list: 土砂移動データ
#   area_list: 領域データ
#   img: 地図画像
#   """
#   pass


@tool.stop_watch
def make_map(move_list, dsm, path):
	"""
	土砂移動図の作成

	list: 土砂移動推定箇所のリスト
	dsm: UAVのDSM 
	"""
	# 領域データ読み込み
	with open("./area_data/l-centroid.csv", encoding='utf8', newline='') as f:
		area = csv.reader(f)
		area_list = [a for a in area]
		# ヘッダを削除
		area_list.pop(0)
		# 背景領域を削除
		area_list.pop(0)

	# オルソ画像の読み込み
	ortho = cv2.imread(path)

	# 解像度（cm）
	resolution = 7.5

	# 水平方向の土砂移動図を作成
	# TODO: 関数にしたい
	for i, move in enumerate(move_list):
		if (move != []):
			# 注目領域の重心座標
			cx, cy = int(area_list[i][2]), int(area_list[i][3])

			# 土砂の流出方向へ矢印を描画
			for m in move:
				# 流出先の重心座標
				_cx, _cy = int(area_list[m][2]), int(area_list[m][3])
				cv2.arrowedLine(
					img=ortho,          # 画像
					pt1=(cx, cy),       # 始点
					pt2=(_cx, _cy),     # 終点
					color=(20,20,180),  # 色
					thickness=1,        # 太さ
					tipLength=1         # 矢先の長さ
				)

				# # 水平距離
				# dis = int(dist((cy, cx), (_cy, _cx)) * resolution)
				# # 水平方向の土砂移動を描画
				# cv2.putText(
				#   img=ortho,                        # 画像
				#   text="hor:"+str(dis)+"cm",        # テキスト
				#   org=(_cx+2, _cy+2),               # 位置
				#   fontFace=cv2.FONT_HERSHEY_PLAIN,  # フォント
				#   fontScale=1,                      # フォントサイズ
				#   color=(0, 255, 0),                # 色
				#   thickness=1,                      # 太さ
				#   lineType=cv2.LINE_AA              # タイプ
				# )

				# # 垂直距離
				# # TODO: DSMで良い？？今頭働いてない
				# dis = int(dsm[cy, cx][0] - dsm[_cy, _cx][0] * 100)
				# # 垂直方向の土砂変化標高
				# cv2.putText(
				#   img=ortho,                        # 画像
				#   text="ver:"+str(dis)+"cm",        # テキスト
				#   org=(_cx+2, _cy+14),              # 位置
				#   fontFace=cv2.FONT_HERSHEY_PLAIN,  # フォント
				#   fontScale=1,                      # フォントサイズ
				#   color=(255, 0, 0),                # 色
				#   thickness=1,                      # 太さ
				#   lineType=cv2.LINE_AA              # タイプ
				# )

	# # 標高差分の堆積・侵食を着色
	# idx = np.where(cv2.split(dsm_sub)[0] >= 0)
	# ortho = tool.draw_color(ortho, idx, (100, 70, 230))
	# idx = np.where(cv2.split(dsm_sub)[0] <  0)
	# # ortho = tool.draw_color(ortho, idx, (200, 135, 125))
	# ortho = tool.draw_color(ortho, idx, (0, 0, 0))

	# 土砂移動図の保存
	cv2.imwrite("./outputs/map.png", ortho)

	return


def standardization_dsm(dsm_uav,dsm_heli):
	# 平均・標準偏差算出
	ave_dsm_uav,sd_dsm_uav   = tool.calc_ave_sd(dsm_uav)
	print("- uav  ave,sd :", ave_dsm_uav, sd_dsm_uav)
	ave_dsm_heli,sd_dsm_heli = tool.calc_ave_sd(dsm_heli)
	print("- heli ave,sd :", ave_dsm_heli, sd_dsm_heli)

	# 標高最大値（山間部の頂上・植生頂上）の変化は無いと仮定する
	# （標高最小値（海・海岸）は海抜0mと仮定する）
	# UAVのDEMを0-180mに正規化
	# 最大標高180m，最小標高0mとする
	max_height_uav = 190
	max_height_heli = 185

	_dsm_uav  = (dsm_uav-ave_dsm_uav)   / sd_dsm_uav  * max_height_uav
	_dsm_heli = (dsm_heli-ave_dsm_heli) / sd_dsm_heli * max_height_heli

	return _dsm_uav, _dsm_heli


def normalize_height(dsm, normed_min, normed_max):
	"""
	標高値正規化

	dsm: 標高値
	normed_min: 正規化後の最大値
	normed_max: 正規化後の最小値
	"""
	# 標高データの最小・最大を算出
	dsm_min,dsm_max = np.nanmin(dsm),np.nanmax(dsm)

	# 正規化処理
	normed_dsm = (dsm-dsm_min) / (dsm_max-dsm_min) * (normed_max-normed_min) + normed_min
	tif.save_tif(normed_dsm, "./inputs/dsm_img.tif", "./outputs/normed_dsm.tif")

	return normed_dsm


def test():
	ortho = cv2.imread("./inputs/koyaura_resize.tif")
	# dsm_sub = cv2.imread("./outputs/dsm_sub.tif")

	tes = cv2.split(ortho)

	# idx = np.where(tes[0] >= 128)
	# ortho = tool.draw_color(ortho, idx, (100, 70,  230))
	# cv2.imwrite("test1.png", ortho)
	idx = np.where(tes[0] >= 128)
	ortho = tool.draw_color(ortho, idx, (200, 135, 125))
	cv2.imwrite("test2.png", ortho)


def remove_vegitation(img):
	"""
	植生領域の除去

	img: カラー画像
	"""	
	# アルファブレンド値
	al = 0.45

	# チャンネルの分割
	b, g, r = cv2.split(img)

	# 表色系変換
	lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	Lp, ap, bp = cv2.split(lab)
	hp, sp, vp = cv2.split(hsv)

	# 植生除去
	idx = np.where(((ap<120) | (bp<120)) & (hp>20))
	b[idx], g[idx], r[idx] = 0, 0, 0
	# b[idx] = (b[idx] * al + 40  * (1-al)),
	# g[idx] = (g[idx] * al + 220 * (1-al)),
	# r[idx] = (r[idx] * al + 140 * (1-al))
	img = np.dstack((np.dstack((b, g)), r))

	cv2.imwrite('outputs/vegitation.png', img)

	return img
