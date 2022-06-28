from os import stat
import cv2
import math
import random
import numpy as np
import pymeanshift as pms

from modules import tif
from modules import tool
from modules import process
from modules import labeling


def dem2gradient(dem, mesh_size):
	"""
	DEMを勾配データへ変換

	dem: 国土地理院DEM
	mesh_size: DEMのメッシュサイズ
	"""
	max = 0
	width, height = dem.shape[1], dem.shape[0]
	gradient = np.zeros((height - 2, width -2, 1))
	for y in range(1, height - 2):
		for  x in range(1, width - 2):
			for j in range(-1, 2):
				for i in range(1, 2):
					angle = math.degrees((float(abs(dem[y + j, x + i][0] - dem[y, x][0])) / float(mesh_size)))
					if angle > max:
						max = angle
			gradient[y,x] = angle
			max = 0
	gradient = gradient.astype(np.int16)

	# 3次元に戻す
	gradient = cv2.merge((gradient, gradient, gradient))

	# 画像を保存
	tif.save_tif(gradient, "./inputs/dem.tif", "./outputs/angle.tif")
	
	return gradient


def resampling_dsm(dsm_uav, dsm_heli, dem, deg, mask):
	"""
	UAVのNoData部分を航空画像DSM・DEMから切り取り・解像度のリサンプリング

	dsm_uav: UAV画像から作成したDSM
	dsm_heli: 航空画像から作成したDSM
	dem: 国土地理院DEM
	deg: 傾斜方位データ
	mask: マスク画像
	"""
	# 3次元にする
	mask = cv2.merge((mask, mask, mask))

	# バイキュービック補間で解像度の統一
	resize_dsm_heli = cv2.resize(dsm_heli, (dsm_uav.shape[1], dsm_uav.shape[0]), interpolation=cv2.INTER_CUBIC)
	resize_dem      = cv2.resize(dem,      (dsm_uav.shape[1], dsm_uav.shape[0]), interpolation=cv2.INTER_CUBIC)
	resize_deg      = cv2.resize(deg,      (dsm_uav.shape[1], dsm_uav.shape[0]), interpolation=cv2.INTER_CUBIC)
	resize_mask     = cv2.resize(mask,     (dsm_uav.shape[1], dsm_uav.shape[0]), interpolation=cv2.INTER_CUBIC)

	# UAV画像のDSMの最小値を算出（領域外の透過背景値）
	background_pix = np.min(dsm_uav)

	# 航空写真DSM・DEMの不要範囲除去（UAVでの透過背景消去）
	idx = np.where(dsm_uav == background_pix)
	dsm_uav[idx]         = np.nan
	resize_dsm_heli[idx] = np.nan
	resize_dem[idx]      = np.nan 
	resize_deg[idx]      = np.nan
	resize_mask[idx]     = 0

	# 画像の保存
	tool.save_resize_image("./outputs/heli_resampling.png", resize_dsm_heli, (500,500))
	tool.save_resize_image("./outputs/dem_resampling.png" , resize_dem     , (500,500))
	tool.save_resize_image("./outputs/deg_resampling.png" , resize_deg     , (500,500))
	tool.save_resize_image("./outputs/mask_resampling.png", resize_mask    , (500,500))

	# 1次元に戻す
	resize_mask = cv2.split(resize_mask)[0]

	return dsm_uav, resize_dsm_heli, resize_dem, resize_deg, resize_mask


def norm_mask(mask):
	"""
	マスク画像の前処理

	mask: マスク画像
	"""
	# 面積の閾値（スケール済みであるので(area_th/scale)が閾値となる）
	# 輪郭抽出の際の拡大倍率
	area_th, scale = 16666, 3

	# 画像反転
	mask = cv2.bitwise_not(mask)

	# モルフォロジー処理
	# TODO: カーネルサイズと実行回数の調整
	# morpho_mask = morphology(mask, 15, 15)
	morpho_mask = process.morphology(mask, 3, 3)

	# 輪郭抽出
	# contours, normed_mask = process.get_norm_contours(morpho_mask, scale, 15)
	contours, normed_mask = process.get_norm_contours(morpho_mask, scale, 3)

	# 面積が閾値未満の領域を削除
	contours = list(filter(lambda x: cv2.contourArea(x) >= 50000, contours))

	# 輪郭データをcsvに保存
	normed_mask = tool.contours2csv(contours, normed_mask, area_th, scale)

	return normed_mask


def use_org_mask(mask):
	"""
	マスク画像に前処理を行わずそのまま使用

	mask: マスク画像
	"""
	# 輪郭抽出の際の拡大倍率
	scale = 1

	# 画像の反転
	normed_mask = cv2.bitwise_not(mask)

	# 輪郭抽出
	contours, normed_mask = process.get_contours(mask, scale)

	# 輪郭データをcsvに保存
	normed_mask = tool.contours2csv(contours, normed_mask, 0, scale)
	
	return normed_mask


@tool.stop_watch
def divide_area(img, spatial_radius, range_radius, min_density):
	"""
	空撮画像に対し色相特徴を用いた領域分割を行う

	img: カラー画像
	spatial_radius: 
	range_radius: 
	min_density: 
	"""
	# 領域分割
	lab_img, label_image, number_regions = pms.segment(cv2.cvtColor(img, cv2.COLOR_BGR2Lab), spatial_radius, range_radius,min_density)
	img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
	cv2.imwrite('./outputs/meanshift.png',img)
	cv2.imwrite('./outputs/pms_label.png', label_image)
	print("- label-num:", number_regions)

	return img, number_regions


@tool.stop_watch
def labeling_color(mask, img):
	"""
	カラー画像をRGBデータでラベリング

	img: 領域分割済みのカラー画像
	"""
	# 土砂マスクを用いて土砂領域以外を除去
	sed = process.extract_sediment(img, mask)

	# 二値化
	bin = process.binarize(sed, 150, cv2.THRESH_BINARY_INV)

	# ラベリング
	labeling.labeling_color(sed, bin)


@tool.stop_watch
def labeling_color_v1(mask, img):
	"""
	土砂マスク中の領域のみを算出

	mask: マスク画像
	img: 領域分割画像
	"""
	# 土砂マスクを用いて土砂領域以外を除去
	sed = process.extract_sediment(img, mask)

	# ラベリング
	dummy, labels, label_num = labeling_color(sed)

	# 画像を保存
	np.savetxt('outputs/dummy.txt', dummy.astype(np.uint8),fmt='%d')
	with open('outputs/label.txt', 'w') as f:
		print(label_num, file=f)
	cv2.imwrite('outputs/labeling.png', labels.astype(np.uint8))
	cv2.imwrite('outputs/dummy.png', dummy.astype(np.uint8))

	# 領域データを保存
	tool.labeling2centroid()

	return


@tool.stop_watch
def labeling_bin(mask, img):
	"""
	土砂マスク中の領域のみを算出

	mask: マスク画像
	img: 領域分割画像
	"""
	# 土砂マスクを用いて土砂領域以外を除去
	sed = process.extract_sediment(img, mask)

	# 二値化
	# dst = process.binarize(sed, 150, cv2.THRESH_BINARY_INV)
	gray_img = cv2.cvtColor(sed, cv2.COLOR_BGR2GRAY)
	# blur_img = cv2.GaussianBlur(gray_img, (5, 5), 2)
	_, dst = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# ラベリング
	# 4-con : 36979 -> 5172
	# 8-con :  8253 -> 2436
	ret, markers, stats, centroids = cv2.connectedComponentsWithStats(
		image=dst, 
		connectivity=4
		# connectivity=8
	)

	# ラベル数を表示
	print("- label num :", ret)

	# 小領域除去しcsvに保存
	# area_th = 3
	area_th = 12
	# stats, centroids, markers = tool.labeling2csv(area_th, stats, centroids, markers)
	tool.labeling2csv(area_th, stats, centroids, markers)

	# 小領域除去後のラベル数を表示
	# ret = len(stats)
	# print("- label num(norm) :", ret)

	# # ラベリング結果書き出し準備
	# height, width = dst.shape[:2]
	# colors = []

	# うまく動いているペイント（FIXME: 除去領域分も回しているので遅い）
	# # 各オブジェクトをランダム色でペイント
	# for i in range(1, ret):
	#   colors.append(np.array([
	#     random.randint(0, 255), 
	#     random.randint(0, 255), 
	#     random.randint(0, 255)
	#   ]))
	# print("- color listed")
	# for i in range(1, ret):
	#   if stats[i][4] > area_th:
	#       img[markers == i, ] = colors[i - 1]
	# print("- color painted")



	# label_img = np.zeros_like(img)
	# for marker in range(markers.max() + 1):
	#   label_group_index = np.where(markers == marker)
	#   label_img[label_group_index] = random.sample(range(255), k=3)

	# for i in range(1, ret):
	#   colors.append(np.array([
	#     random.randint(0, 255), 
	#     random.randint(0, 255), 
	#     random.randint(0, 255)
	#   ]))
	# for i in range(1, ret):
	#   # if stats[i][4] >= area_th:
	#     img[markers[0] == i, ] = colors[i - 1]
	
	# # オブジェクトの総数を黄文字で表示
	# cv2.putText(img, str(ret - 1), (100, 100), cv2.FONT_HERSHEY_PLAIN, 100, (255, 255, 255))

	# # 画像を保存
	# tool.save_resize_image("./outputs/labeling_bin.png", img, (int(height/5), int(width/5)))

	return


@tool.stop_watch
def extract_contours(mask, img):
	"""
	土砂マスク中のカラー画像領域から輪郭を抽出

	mask: マスク画像
	img: 領域分割画像
	"""
	# 輪郭抽出の際の拡大倍率
	scale = 3

	# 土砂マスクを用いて土砂領域以外を除去
	sed = process.extract_sediment(mask, img)

	# 二値化
	# dst = process.binarize(sed, 150, cv2.THRESH_BINARY_INV)

	# グレースケール化
	gray = cv2.cvtColor(sed, cv2.COLOR_BGR2GRAY)

	# 二値化と大津処理
	r, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# 輪郭抽出
	contours, _ = cv2.findContours(dst.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# 輪郭データをcsvに保存
	con = tool.contours2csv(contours, dst, 0, scale)

	# 領域数を表示
	tool.show_area_num()

	# 画像を保存
	cv2.imwrite("./outputs/con.png", con)
	
	return


def norm_elevation(dsm_uav, dsm_heli, dem):
	"""
	DEMの標高をDSMにマッチングさせ標高値を対応付ける

	dsm_uav: UAV画像から作成したDSM
	dsm_heli: 航空画像から作成したDSM
	dem: 国土地理院DEM
	"""
	# 最小値・最大値算出
	min_uav,  max_uav  = tool.calc_min_max(dsm_uav)
	min_heli, max_heli = tool.calc_min_max(dsm_heli)
	min_dem,  max_dem  = tool.calc_min_max(dem)
	# print("- uav-range  :", min_uav , max_uav)    # 1.0 255.0
	# print("- heli-range :", min_heli, max_heli)   # 52.16754 180.19545
	# print("- dem-range  :", min_dem , max_dem)    # -0.54201436 146.51208

	# 植生を加味
	# max_dem += 15
	# max_dem += 10

	# 正規化処理
	# _dsm_uav  = (dsm_uav-min_uav)   / (max_uav-min_uav)   * (max_dem + 15)
	# _dsm_heli = (dsm_heli-min_heli) / (max_heli-min_heli) * (max_dem + 10)
	_dsm_uav  = (dsm_uav-min_uav)   / (max_uav-min_uav)
	_dsm_heli = (dsm_heli-min_heli) / (max_heli-min_heli)

	# 画像を保存
	tool.save_resize_image("./outputs/uav_norm.png" , _dsm_uav,  (500,500))
	tool.save_resize_image("./outputs/heli_norm.png", _dsm_heli, (500,500))

	return _dsm_uav, _dsm_heli


def calc_sedimentation(dsm_uav, dsm_heli, mask):
	"""
	災害前後での標高差分を算出
	DEMの標高をDSMにマッチングさせ標高値を対応付ける

	dsm_uav: UAV画像から作成したDSM
	dsm_heli: 航空画像から作成したDSM
	mask: 土砂マスク画像
	"""
	# 標高差分を算出
	dsm_sub = dsm_uav - dsm_heli

	# 土砂領域以外を除外
	idx_mask = np.where(mask == 255)
	dsm_sub[idx_mask] = np.nan

	# 堆積領域と侵食領域で二値化
	dsm_bin = dsm_sub.copy()
	idx = np.where(dsm_sub >  0)
	dsm_bin[idx] = 255
	idx = np.where(dsm_sub <= 0)
	dsm_bin[idx] = 0

	# tif画像への書き出し
	tif.save_tif(dsm_sub, "./inputs/dsm_uav.tif", "./outputs/dsm_sub.tif")
	tif.save_tif(dsm_bin, "./inputs/dsm_uav.tif", "./outputs/dsm_bin.tif")

	return dsm_sub


def norm_degree(deg):
	"""
	入力された傾斜方位（0-255）を実際の角度（0-360）に正規化

	deg: 傾斜方位データ
	"""
	# 0-360度に変換
	deg = deg / 255 * 360

	return deg


def norm_degree_v2(deg):
	"""
	入力された傾斜方位（負値含む）を実際の角度（0-360）に正規化

	deg: 傾斜方位データ
	"""
	# 最大・最小
	deg_max = np.nanmax(deg)
	deg_min = np.nanmin(deg)

	# normed_dsm = (dsm-dsm_min) / (dsm_max-dsm_min) * (normed_max-normed_min) + normed_min

	# 0-360度に変換
	deg = (deg - deg_min) / (deg_max - deg_min) * 360

	# deg = deg / 255 * 360

	return deg


def calc_movement(dsm_sub, dem, deg, grad, dsm, path):
	"""
	土砂移動の推定

	dsm_sub: 標高差分結果
	dem: 国土地理院DEM（切り抜き無し）
	deg: 傾斜方位データ
	grad: 傾斜データ
	dsm: UAVのDSM
	"""
	# 上流が侵食かつ下流が堆積の領域の組を全て抽出
	## 領域が隣接している領域の組を全て抽出
	area_list1 = process.extract_neighbor()

	## 傾斜方向が上から下の領域の組を全て抽出
	area_list2 = process.extract_direction(deg, dem, dsm)

	## 侵食と堆積の領域の組を全て抽出
	area_list3 = process.extract_sub(dsm_sub)

	# 上記3つの条件を全て満たす領域の組を抽出
	# area_list = tool.and_operation_2(area_list1, area_list2)
	area_list = tool.and_operation(area_list1, area_list2, area_list3)

	# 土砂移動図の作成
	process.make_map(area_list, dsm, path)
	print("- area-list :", area_list)
	print("- area-num  :", len(area_list))

	return


def standardization_dsm(dsm_uav,dsm_heli):
	# 平均・標準偏差算出
	ave_dsm_uav,sd_dsm_uav   = tool.calc_ave_sd(dsm_uav)
	print("- uav  ave, sd :", ave_dsm_uav, sd_dsm_uav)
	ave_dsm_heli,sd_dsm_heli = tool.calc_ave_sd(dsm_heli)
	print("- heli ave, sd :", ave_dsm_heli, sd_dsm_heli)

	# 標高最大値（山間部の頂上・植生頂上）の変化は無いと仮定する
	# （標高最小値（海・海岸）は海抜0mと仮定する）
	# UAVのDEMを0-180mに正規化
	# 最大標高180m，最小標高0mとする
	max_height_uav  = 190
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
	dsm_min, dsm_max = np.nanmin(dsm), np.nanmax(dsm)

	# 正規化処理
	normed_dsm = (dsm-dsm_min)/(dsm_max-dsm_min)*(normed_max-normed_min) + normed_min
	tif.save_tif(normed_dsm, "./inputs/dsm_img.tif", "./outputs/normed_dsm.tif")

	return normed_dsm


# def extract_region(img, num):
# 	"""
# 	領域データを抽出

# 	img: 領域分割画像
# 	num: 領域数
# 	"""
# 	h, w = img.shape
# 	# table = np.zeros((h+2, w+2),dtype=int)
# 	table = []
# 	label = 1
	
# 	# ラスタスキャンを行い同一色の画素値をテーブルに追加
# 	for i in range(img.shape[0]):
# 		for j in range(img.shape[1]):
# 			pix = img[i,j]
# 			print(pix)

# 			if (pix == [0, 0, 0]) and ():
# 				continue

# 			table.append([label, pix])
# 			label += 1

# 	print("- label num:", label)
# 	print("- regions num:", num)

# 	for i in range(num):
		
		
# 		# print(i)


# 		pass

