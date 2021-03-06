import cv2
import csv
import math
import random
import numpy as np
from math import dist
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


@tool.stop_watch
def calc_gradient(self, mesh_size):
	"""
	勾配データの算出

	mesh_size: DEMのメッシュサイズ
	"""
	# サイズ
	width, height = self.dem.shape[1], self.dem.shape[0]

	# 勾配データの算出
	max = 0
	grad = np.zeros((height - 2, width -2, 1))
	for y in range(1, height - 2):
		for  x in range(1, width - 2):
			for j in range(-1, 2):
				for i in range(1, 2):
					angle = math.degrees((
						float(abs(self.dem[y + j, x + i][0] - self.dem[y, x][0])) / float(mesh_size)
					))
					if angle > max:
						max = angle
			grad[y,x] = angle
			max = 0
	grad = grad.astype(np.int16)

	return grad


def morphology(self, ksize, exe_num):
	"""
	モルフォロジー処理

	ksize: カーネルサイズ
	exe_num: 実行回数
	"""
	# モルフォロジー処理によるノイズ除去
	kernel = np.ones((ksize, ksize), np.uint8)
	opening = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
	for i in range(1, exe_num):
		opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	for i in range(1, exe_num):
		closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

	# クロージング処理による建物領域の穴埋め
	closing = cv2.dilate(closing, kernel, iterations = 1)
	closing = cv2.dilate(closing, kernel, iterations = 1)
	closing = cv2.erode (closing, kernel, iterations = 1)
	closing = cv2.erode (closing, kernel, iterations = 1)

	# 画像を保存
	tool.save_resize_image("opening.png", opening, self.s_size_2d)
	tool.save_resize_image("closing.png", closing, self.s_size_2d)

	# 結果を保存
	self.mask = closing

	return


def get_norm_contours(self, scale, ksize):
	"""
	輪郭をぼやけさせて抽出

	scale: 拡大倍率
	ksize: カーネルサイズ
	"""
	# 画像の高さと幅を取得
	w, h = self.size_2d

	# 拡大することで輪郭をぼけさせ境界を識別しやすくする
	img_resize = cv2.resize(self.mask, (w * scale, h * scale))

	# ガウシアンによるぼかし処理
	img_blur = cv2.GaussianBlur(img_resize, (ksize,ksize), 0)

	# 二値化と大津処理
	_, dst = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# モルフォロジー膨張処理
	kernel = np.ones((ksize,ksize), np.uint8)
	dst = cv2.dilate(dst, kernel, iterations = 1)

	# 画像欠けがあった場合に塗りつぶし
	self.mask = ndimage.binary_fill_holes(dst).astype(int) * 255

	# 輪郭抽出
	contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	return contours


def remove_small_area(self, contours, area_th, scale):
	"""
	面積が閾値以下の領域を除去

	contours: 輪郭データ
	area_th: 面積の閾値
	scale: 拡大倍率
	"""
	# 画像の高さと幅を取得
	h, w = self.mask.shape

	# 輪郭データをフィルタリング
	contours = list(filter(lambda x: cv2.contourArea(x) >= area_th * scale, contours))

	# 黒画像
	campus = np.zeros((h, w))

	# TODO: スケール分は考慮して割る必要あり
	for i, contour in enumerate(contours):
		# 面積
		area = int(cv2.contourArea(contour) / scale)

		# 閾値以上の面積の場合画像に出力
		if (area >= area_th):
			normed_mask = cv2.drawContours(campus, contours, i, 255, -1)

	# スケールを戻す
	self.mask = cv2.resize(normed_mask, (int(w/scale), int(h/scale)))

	# 画像の保存
	cv2.imwrite("./outputs/normed_mask.png", normed_mask)

	return


def masking(self, img, mask):
	"""
	マスク処理にて不要領域を除去

	img: マスク対象の画像
	mask: マスク画像
	"""
	# 画像のコピー
	masked_img = img.copy()

	# マスク領域以外を除去
	idx = np.where(mask != 0)
	try:
		masked_img[idx] = np.nan
	except:
		masked_img[idx] = 0

	# # 土砂領域を抽出
	# idx = np.where(mask == 0).astype(np.float32)
	# # 土砂領域以外の標高データを消去
	# sed[idx] = np.nan
	# tif.save_tif(dsm, "dsm_uav.tif", "sediment.tif")
	# return sed.astype(np.uint8)

	# 画像を保存
	tool.save_resize_image("masked_img.png", masked_img, self.s_size_2d)

	return masked_img


def get_pms_contours(self, region_list):
	"""
	各領域をキャンパスに描画し1つずつ領域データを抽出
	領域分割結果からラベリング画像を作成

	region_list: 領域分割で得た領域データ
	"""
	with open('./area_data/region.csv', 'w') as f:
		writer = csv.writer(f)

		# ヘッダを追加
		writer.writerow(["id", "area", "x_centroid", "y_centroid", "circularity"])

		# キャンパス描画
		label_img = np.zeros((self.size_2d[1], self.size_2d[0], 3))

		for region in region_list:
			# 領域データを取得
			label, cords, area = tool.decode_area(region)

			# 注目領域のマスク画像を作成
			label_img, mask = draw_region(self, label_img, cords)

			# 輪郭抽出
			contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# FIXME: contoursが2要素以上出てくる場合がある
			# FIXME: pmsで算出した面積と異なる場合がある

			# 面積
			area_float = cv2.contourArea(contours[0])
			area = int(area_float)

			# 輪郭の重心
			M = cv2.moments(contours[0])
			try:
				cx = int((M["m10"] / M["m00"]))
				cy = int((M["m01"] / M["m00"]))
			except:
				cx, cy = 0, 0

			# 輪郭の周囲長
			arc_len = cv2.arcLength(contours[0], True)

			# 円形度
			circularity = 4.0 * math.pi * area_float / (arc_len * arc_len)

			# csvファイルに保存
			data_list = [label, area, cx, cy, circularity]
			writer.writerow(data_list)

		# 画像を保存
		cv2.imwrite("./outputs/label.png", label_img)
	
	return


def draw_region(self, label_img, cords):
	"""
	与えられた座標を領域とし特定画素で埋める

	label_img: ラベル画像
	cords: 領域の座標群
	"""
	# キャンパス描画
	campus = np.zeros((self.size_2d[1], self.size_2d[0]))

	# ランダム色を生成
	color = [
		random.randint(0, 255),
		random.randint(0, 255),
		random.randint(0, 255),
	]

	for cord in cords:
		# ランダム色のラベル画像を作成
		label_img[cord] = color
		# 領域座標を白画素で埋める
		campus[cord] = 255

	return label_img, campus


def extract_building(self, region_list, cords_list):
	"""
	建物領域を抽出

	region_list: 領域の詳細データ
	cords_list: 領域分割で得た領域座標データ
	"""
	# 建物領域検出用画像
	bld_img = self.ortho.copy()

	# キャンパス描画
	cir_img = np.zeros((self.size_2d[1], self.size_2d[0]))

	for region, cords in zip(region_list, cords_list):
		# 領域・座標データを取得
		circularity = region[4]
		_, cords, _  = tool.decode_area(cords)
		
		# 円形度を0-255に正規化
		circularity = int(float(circularity) * 255)

		# 円形度を大小で描画
		for cord in cords:
			cir_img[cord] = circularity

		# 建物領域の検出
		if is_building(self, circularity, cords[0]):
			# 塗りつぶし
			for cord in cords:
				bld_img[cord] = [0, 0, 220]

	# 画像を保存
	tool.save_resize_image("circularity.png", cir_img, self.s_size_2d)
	tool.save_resize_image("building.png", bld_img, self.s_size_2d)
	
	return


def is_building(self, cir, cord):
	"""
	建物領域かどうかを判別

	cir: 円形度
	cord: 該当領域の座標
	"""
	# 円形度
	if not (cir > 50):
		return False
	else:
		return True

	# TODO: 土砂・植生等も判別
	# # 土砂領域
	# if not (img[cord] ):


def bin_2area(self):
	"""
	堆積領域と侵食領域で二値化
	"""
	dsm_bin = self.dsm_sub.copy()
	idx = np.where(self.dsm_sub >  0)
	dsm_bin[idx] = 255
	idx = np.where(self.dsm_sub <= 0)
	dsm_bin[idx] = 0

	return dsm_bin


@tool.stop_watch
def extract_neighbor():
	"""
	隣接している領域の組を全て抽出
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
			if ((d > 5) and (d <= 15))
			# if ((d > 6) and (d <= 15))
			# if ((d > 8) and (d <= 15))
			# if ((d > 5) and (d <= 30))
			# if (d <= 30)
		]

		# リストに追加
		neighbor_idx_list.append(idx_list)

	return neighbor_idx_list


@tool.stop_watch
def extract_direction(self):
	"""
	傾斜方向が上流から下流の領域の組を全て抽出
	"""
	# 傾斜方向データでやりたい
	# 領域データ読み込み
	with open("./area_data/region.csv", encoding='utf8', newline='') as f:
	# with open("./area_data/l-centroid.csv", encoding='utf8', newline='') as f:
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
		centroid_elevation = self.dem[cy, cx]

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
def extract_sub(self):
	"""
	侵食と堆積の領域の組を全て抽出
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

	# 領域の最短距離を算出
	sub_idx_list = []
	for area in area_list:
		# 注目領域の重心
		cx, cy = int(area[2]), int(area[3])

		# 注目領域の重心の標高変化
		sub_elevation = self.dsm_sub[cy, cx][0]

		# 堆積・侵食の組み合わせを算出
		idx_list = [
			idx for idx, sub in enumerate(area_list)
			if ((sub_elevation > self.dsm_sub[int(sub[3]), int(sub[2])][0]))
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


@tool.stop_watch
def make_map(self, move_list):
	"""
	土砂移動図の作成

	list: 土砂移動推定箇所のリスト
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
					img=self.ortho,     # 画像
					pt1=(cx, cy),       # 始点
					pt2=(_cx, _cy),     # 終点
					color=(20,20,180),  # 色
					thickness=2,        # 太さ
					tipLength=0.4       # 矢先の長さ
				)
				# cv2.arrowedLine(
				# 	img=ortho,          # 画像
				# 	pt1=(cx, cy),       # 始点
				# 	pt2=(_cx, _cy),     # 終点
				# 	color=(20,20,180),  # 色
				# 	thickness=1,        # 太さ
				# 	tipLength=1         # 矢先の長さ
				# )

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
	tool.save_resize_image("map.png", self.ortho, self.size_2d)

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
	tif.save_tif(normed_dsm, "dsm_img.tif", "normed_dsm.tif")

	return normed_dsm


def test():
	ortho = cv2.imread("koyaura_resize.tif")
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


def remove_black_pix(img, path):
	"""
	読み込み画像の黒画素領域を除去

	img: 除去対象の画像
	path: 読み込み対象の画像
	"""
	# マスク画像を読み込み
	mask = cv2.imread(path)

	# マスク画像サイズをあわせる
	mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

	# マスク画像の次元サイズ変更
	mask = cv2.split(mask)[0]

	# 黒画素のインデックスを取得
	idx = np.where(mask == 0)

	# 黒画素領域を除去
	img[idx] = 0

	cv2.imwrite("outputs/test.png", img)

	return img

