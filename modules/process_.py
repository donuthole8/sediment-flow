import cv2
import math
import random
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import dist
from tqdm import trange

from modules import tif
from modules import tool



def binarize(
	src_img: np.ndarray, 
	thresh: int, 
	mode: str
) -> np.ndarray:
	"""
	二値化
	"""
	gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
	bin_img = cv2.threshold(gray_img, thresh, 255, mode)[1]

	return bin_img


@tool.stop_watch
def calc_gradient(self, mesh_size: int) -> np.ndarray:
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


def dem2gradient(self, size_mesh: int) -> None:
	"""
	傾斜度を算出する
	
	Parameters
	----------
	dem : np.float32
			DEMデータのグレースケール画像
	size_mesh : int
			DEMデータのメッシュサイズ(m)
	
	Returns
	-------
	gradient : np.int8
			傾斜度のグレースケール画像( 0 ~ 90度 )
	"""
	max = 0
	index = [-1,1]
	height, width = self.dem.shape[:2]
	gradient = np.zeros((height, width))
	dem = np.pad(self.dem, 1, mode = 'edge')
	
	for y in trange(height):
		for x in range(width):
			for j in range(-1,2):
				for i in range(-1,2):
					if i in index and j in index:
						angle = math.degrees(math.atan((float(abs(dem[y+j+1, x+i+1] - dem[y+1, x+1])) / float(size_mesh * pow(2, 0.5)))))
					else:
						angle = math.degrees(math.atan((float(abs(dem[y+j+1, x+i+1] - dem[y+1, x+1])) / float(size_mesh))))
					if angle > max:
						max = angle
			gradient[y,x] = angle
			max = 0
	self.gradient = gradient.astype(np.int8)
	
	return
	

def morphology(
	self, 
	ksize: int, 
	exe_num: int
) -> None:
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


def get_norm_contours(
	self, 
	scale: int, 
	ksize: int
) -> list[list]:
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
	contours, _ = cv2.findContours(
		self.mask.astype(np.uint8), 
		cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE
	)

	return contours


def remove_small_area(
	self, 
	contours: list[list], 
	area_th: int, 
	scale: int
) -> None:
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


def masking(
	self, 
	img: np.ndarray, 
	mask: np.ndarray
) -> np.ndarray:
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


def get_pms_contours(self) -> None:
	"""
	各領域をキャンパスに描画し1つずつ領域データを抽出
	領域分割結果からラベリング画像を作成
	"""
	# キャンパス描画
	label_img = np.zeros((self.size_2d[1], self.size_2d[0], 3))

	for region in self.pms_coords:
		# 領域データを取得
		label, coords, area = tool.decode_area(region)

		# 注目領域のマスク画像を作成
		label_img, mask = draw_region(self, label_img, coords)

		# 輪郭抽出
		contours, _ = cv2.findContours(
			mask.astype(np.uint8), 
			cv2.RETR_EXTERNAL, 
			cv2.CHAIN_APPROX_SIMPLE
		)

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

		# データの保存
		# NOTE: selfに保存するなら辞書型の方が良い？
		# data_list = [label, area, cx, cy, circularity]
		data_list = {
			"label":       label, 
			"area":        area, 
			"cx":          cx, 
			"cy":          cy, 
			"circularity": circularity
		}
		self.region.append(data_list)

	# 画像を保存
	cv2.imwrite("./outputs/label.png", label_img)
	
	return


def draw_region(
	self, 
	label_img: np.ndarray, 
	coords: list[tuple]
) -> tuple[np.ndarray, np.ndarray]:
	"""
	与えられた座標を領域とし特定画素で埋める

	label_img: ラベル画像
	coords: 領域の座標群
	"""
	# キャンパス描画
	campus = np.zeros((self.size_2d[1], self.size_2d[0]))

	# ランダム色を生成
	color = [
		random.randint(0, 255),
		random.randint(0, 255),
		random.randint(0, 255),
	]

	for coord in coords:
		# ランダム色のラベル画像を作成
		label_img[coord] = color
		# 領域座標を白画素で埋める
		campus[coord] = 255

	return label_img, campus


def texture_analysis(self) -> None:
	"""
	オルソ画像に対しテクスチャ解析
	"""
	mpl.rc('image', cmap='jet')
	kernel_size = 5
	levels = 8
	symmetric = False
	normed = True
	dst = cv2.cvtColor(self.ortho, cv2.COLOR_BGR2GRAY)

	# binarize
	dst_bin = dst // (256 // levels) # [0:255] -> [0:7]

	# calc_glcm
	h,w = dst.shape
	glcm = np.zeros((h,w,levels,levels), dtype=np.uint8)
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	dst_bin_r = np.append(dst_bin[:,1:], dst_bin[:,-1:], axis=1)
	for i in range(levels):
		for j in range(levels):
			mask = (dst_bin==i) & (dst_bin_r==j)
			mask = mask.astype(np.uint8)
			glcm[:,:,i,j] = cv2.filter2D(mask, -1, kernel)
	glcm = glcm.astype(np.float32)
	if symmetric:
		glcm += glcm[:,:,::-1, ::-1]
	if normed:
		glcm = glcm/glcm.sum(axis=(2,3), keepdims=True)
	# martrix axis
	axis = np.arange(levels, dtype=np.float32)+1
	w = axis.reshape(1,1,-1,1)
	x = np.repeat(axis.reshape(1,-1), levels, axis=0)
	y = np.repeat(axis.reshape(-1,1), levels, axis=1)

	# GLCM contrast
	glcm_contrast = np.sum(glcm*(x-y)**2, axis=(2,3))
	# GLCM dissimilarity（不均一性）
	glcm_dissimilarity = np.sum(glcm*np.abs(x-y), axis=(2,3))
	# GLCM homogeneity（均一性）
	glcm_homogeneity = np.sum(glcm/(1.0+(x-y)**2), axis=(2,3))
	# GLCM energy & ASM
	glcm_asm = np.sum(glcm**2, axis=(2,3))
	# GLCM entropy（情報量）
	ks = 5 # kernel_size
	pnorm = glcm / np.sum(glcm, axis=(2,3), keepdims=True) + 1./ks**2
	glcm_entropy = np.sum(-pnorm * np.log(pnorm), axis=(2,3))
	# GLCM mean
	glcm_mean = np.mean(glcm*w, axis=(2,3))
	# GLCM std
	glcm_std = np.std(glcm*w, axis=(2,3))
	# GLCM energy
	glcm_energy = np.sqrt(glcm_asm)
	# GLCM max
	glcm_max = np.max(glcm, axis=(2,3))
	
	# plot
	# plt.figure(figsize=(10,4.5))

	outs = [dst, glcm_mean, glcm_std,
		glcm_contrast, glcm_dissimilarity, glcm_homogeneity,
		glcm_asm, glcm_energy, glcm_max,
		glcm_entropy]
	titles = ['original','mean','std','contrast','dissimilarity','homogeneity','ASM','energy','max','entropy']
	for i in range(10):
		plt.imsave('./outputs/texture/' + titles[i] + '.png', outs[i])

	# GLCM dissimilarity（不均一性）
	# - [0.0 - 3.8399997]
	self.dissimilarity = outs[4]
	
	return


def edge_detection(
	self, 
	threshold1: int, 
	threshold2: int
) -> None:
	"""
	エッジ抽出
	"""
	# グレースケール化
	img_gray = cv2.cvtColor(self.ortho, cv2.COLOR_BGR2GRAY).astype(np.uint8)

	# エッジ抽出
	img_canny = cv2.Canny(img_gray, threshold1, threshold2)

	# 画像保存
	cv2.imwrite("./outputs/edge.png", img_canny)

	return


def extract_building(self) -> None:
	"""
	建物領域を抽出

	region_list: 領域の詳細データ
	coords_list: 領域分割で得た領域座標データ
	"""
	# 建物領域検出用画像
	bld_img = self.ortho.copy()

	# キャンパス描画
	cir_img  = np.zeros((self.size_2d[1], self.size_2d[0]))
	bld_mask = np.zeros((self.size_2d[1], self.size_2d[0]))

	for region, coords in zip(self.region, self.pms_coords):
		# 領域・重心・座標データを取得
		circularity  = int(float(region["circularity"]) * 255)
		centroids    = (region["cy"], region["cx"])
		_, coords, _  = tool.decode_area(coords)

		# 円形度を大小で描画
		for coord in coords:
			cir_img[coord] = circularity

		# 建物領域の検出
		if is_building(self, circularity, centroids):
			# 塗りつぶし
			for coord in coords:
				bld_img[coord]  = [0, 0, 220]
				bld_mask[coord] = 255

			# 建物領域の保存
			self.building.append({
				"label": region["label"], 
				"cx":    region["cx"], 
				"cy":    region["cy"], 
				"coords": tuple(coords)
			})

	# 画像を保存
	tool.save_resize_image("circularity.png", cir_img, self.s_size_2d)
	tool.save_resize_image("building.png", bld_img, self.s_size_2d)
	cv2.imwrite("./outputs/building_mask.png", bld_mask)

	self.bld_mask = bld_mask
	
	return


def is_building(
	self, 
	circularity: float, 
	centroids: tuple[int, int]
) -> bool:
	"""
	建物領域かどうかを判別

	circularity: 円形度
	centroids: 該当領域の重心座標
	"""
	# 注目領域の平均異質度を取得
	dissimilarity = np.mean(self.dissimilarity[centroids])

	if (not (circularity > 40)) or (not (dissimilarity < 1)):
		# 非建物領域
		return False
	elif is_sediment_or_vegetation(self, centroids):
		# 土砂領域・植生領域
		return False
	else:
		# 建物領域
		return True


def is_sediment_or_vegetation(self, centroids: tuple[int, int]) -> bool:
	"""
	土砂・植生領域かどうかを判別

	centroids: 該当領域の重心座標
	"""
	# Lab表色系に変換
	Lp, ap, bp = cv2.split(
		cv2.cvtColor(self.div_img.astype(np.uint8), cv2.COLOR_BGR2Lab)
	)
	Lp, ap, bp = Lp * 255, ap * 255, bp * 255

	# 土砂
	sediment   = (Lp[centroids] > 125) & (ap[centroids] > 130)
	# 植生
	vegetation = (ap[centroids] < 110) | (bp[centroids] <  70)

	if (sediment | vegetation):
		return True
	else:
		return False


def norm_building(self) -> None:
	"""
	建物領域の標高値を地表面と同等に補正する
	"""
	# 正規化後のDSM標高値
	# self.normed_dsm = self.dsm_uav.copy()

	# 比較用
	cv2.imwrite("./outputs/uav_dsm.tif",  self.dsm_uav)

	# 建物領域毎に処理
	for bld in self.building:
		# 建物領域の周囲領域の地表面標高値を取得
		neighbor_height = get_neighbor_region(self, bld["coords"])

		# TODO: できるか試す
		# self.dsm_uav[bld["coords"]] = neighbor_height

		# UAVのDSMの建物領域を周囲領域の地表面と同じ標高値にする
		# NOTE: DEMを使う場合はコメントアウトのままで良い
		# TODO: 航空画像を使う場合は航空画像からも建物領域を検出
		# neighbor_height_heli = get_neighbor_region(self, coords)

		for coords in bld["coords"]:
			# UAVのDSMの建物領域を周囲領域の地表面と同じ標高値にする
			# self.normed_dsm[coords] = neighbor_height
			self.dsm_uav[coords] = neighbor_height

			# TODO: subも検討
			# self.dsm_sub[coords] -= 10

	# 画像の保存
	# cv2.imwrite("./outputs/normed_uav_dsm.tif",  self.normed_dsm)
	cv2.imwrite("./outputs/normed_uav_dsm.tif",  self.dsm_uav)
	# cv2.imwrite("normed_uav_heli.tif", self.dsm_heli)

	# 建物領域データの開放
	self.building = None

	return


def get_neighbor_region(self, coords: list[tuple]) -> list[int]:
	"""
	建物領域でない隣接領域の標高値を取得
	TODO: 新しく作成した隣接領域検出を使えないか検討

	coords: 注目領域の座標群
	"""
	# キャンパス描画
	campus = np.zeros((self.size_2d[1], self.size_2d[0]))

	# 注目領域のマスク画像を作成
	for coord in coords:
		# 領域座標を白画素で埋める
		campus[coord] = 255

	# 輪郭抽出
	contours, _ = cv2.findContours(
		campus.astype(np.uint8), 
		cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE
	)

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

	# 疑似半径
	r1 = int(math.sqrt(area / math.pi))
	r2 = int(arc_len / (2 * math.pi))

	try:
		if (self.bld_mask[     cy - r2, cx - r2] == 0):
			return self.dsm_uav[(cy - r2, cx - r2)]
	except:
		pass
	try:
		if (self.bld_mask[     cy - r2, cx     ] == 0):
			return self.dsm_uav[(cy - r2, cx     )]
	except:
		pass
	try:
		if (self.bld_mask[     cy - r2, cx + r2] == 0):
			return self.dsm_uav[(cy - r2, cx + r2)]
	except:
		pass
	try:
		if (self.bld_mask[     cy     , cx + r2] == 0):
			return self.dsm_uav[(cy     , cx + r2)]
	except:
		pass
	try:
		if (self.bld_mask[     cy + r2, cx + r2] == 0):
			return self.dsm_uav[(cy + r2, cx + r2)]
	except:
		pass
	try:
		if (self.bld_mask[     cy + r2, cx     ] == 0):
			return self.dsm_uav[(cy + r2, cx     )]
	except:
		pass
	try:
		if (self.bld_mask[     cy + r2, cx - r2] == 0):
			return self.dsm_uav[(cy + r2, cx - r2)]
	except:
		pass
	try:
		if (self.bld_mask[     cy     , cx - r2] == 0):
			return self.dsm_uav[(cy     , cx - r2)]
	except:
		pass

	# 隣接領域の画素を取得
	try:
		if (self.bld_mask[     cy - r1, cx - r1] == 0):
			return self.dsm_uav[(cy - r1, cx - r1)]
	except:
		pass
	try:
		if (self.bld_mask[     cy - r1, cx     ] == 0):
			return self.dsm_uav[(cy - r1, cx     )]
	except:
		pass
	try:
		if (self.bld_mask[     cy - r1, cx + r1] == 0):
			return self.dsm_uav[(cy - r1, cx + r1)]
	except:
		pass
	try:
		if (self.bld_mask[     cy     , cx + r1] == 0):
			return self.dsm_uav[(cy     , cx + r1)]
	except:
		pass
	try:
		if (self.bld_mask[     cy + r1, cx + r1] == 0):
			return self.dsm_uav[(cy + r1, cx + r1)]
	except:
		pass
	try:
		if (self.bld_mask[     cy + r1, cx     ] == 0):
			return self.dsm_uav[(cy + r1, cx     )]
	except:
		pass
	try:
		if (self.bld_mask[     cy + r1, cx - r1] == 0):
			return self.dsm_uav[(cy + r1, cx - r1)]
	except:
		pass
	try:
		if (self.bld_mask[     cy     , cx - r1] == 0):
			return self.dsm_uav[(cy     , cx - r1)]
	except:
		pass

	return self.dsm_uav[(cy, cx)]


def binarize_2area(self) -> np.ndarray:
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
def extract_neighbor(self) -> list[int]:
	"""
	隣接している領域の組を全て抽出
	"""
	# 領域の最短距離を算出
	neighbor_idx_list = []
	for area in self.region:
		# 注目領域の重心
		cx, cy = int(area["cx"]), int(area["cy"])

		# 各重心との距離を求める
		dist_list = [
			dist((cy, cx), (int(cent["cy"]), int(cent["cx"]))) 
			for cent in self.region
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
			# if ((d > 5) and (d <= 15))
			if ((d > 20) and (d <= 25))

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
	# NOTE: 傾斜方向データでやりたい
	# 領域の重心標高を算出
	ave_elevation_list = []
	for i, area in enumerate(self.region):

		# 重心の標高値を算出
		cx, cy = int(area["cx"]), int(area["cy"])
		# FIXME: DEMかDSMかは要検討
		centroid_elevation = self.dem[cy, cx]

		# リストに追加
		# ave_elevation_list.append(centroid_elevation[0])
		ave_elevation_list.append(centroid_elevation)

	# 重心標高値より下流の領域を保持する
	downstream_idx_list = []
	for i, area in enumerate(self.region):
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
	# 領域の最短距離を算出
	sub_idx_list = []
	for area in self.region:
		# 注目領域の重心
		cx, cy = int(area["cx"]), int(area["cy"])

		# 注目領域の重心の標高変化
		# sub_elevation = self.dsm_sub[cy, cx][0]
		sub_elevation = self.dsm_sub[cy, cx]

		# 堆積・侵食の組み合わせを算出
		idx_list = [
			idx for idx, sub in enumerate(self.region)
			# if ((sub_elevation > self.dsm_sub[int(sub[3]), int(sub[2])][0]))
			if ((sub_elevation > self.dsm_sub[int(sub["cy"]), int(sub["cx"])]))
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


def estimate_flow(self):
	"""
	流出方向の予測
	"""
	## 斜面崩壊領域を処理対象としてforを回す
	flow_idx_list = []
	for area in self.region:
		## 注目画素から傾斜方向 + 隣接2方向の画素の標高値を調べる（傾斜方向はQgisの出力を近傍8画素に割り当て,　多分↑が0°になってるので上方向は 337.5° ~ 360° & 0° - 22.5°みたいな？）

		# 注目領域の重心標高
		cx, cy = int(area["cx"]), int(area["cy"])
		pix = self.dsm_uav[cy, cx]

		## 標高値が注目画素よりも小さければそこも斜面崩壊領域としてもう一回ⅱをやるを繰り返してたと思う。（もとは傾斜とか条件つけてたけど全然領域が広がらなかったので標高の大小だけにした気がする）
		## 領域内での→を付けたい！！
		## 下端まで〜

		# 始点
		x, y = cx, cy

		# 傾斜方向の標高
		# NOTE: しきい値変えれる
		idx_list = []

		# for i in range(0, 30):
		for i in range(0, 10):
			# 注目領域の重心標高から傾斜方向を探査
			dx, dy = detect_flow(self.degree[cy, cx])
			# 終点
			x, y = x + dx, y + dy

			# nanだった場合
			if (math.isnan(dx)):
				break
			# 標高が上がった場合
			try:
				if ((self.dsm_uav[y, x]) > pix):
					break
			except:
				print("err")
				if ((self.dsm_uav[y - 1, x - 1]) > pix):
					break

		# NOTE: ここにもう色々条件書いていっちゃう！！！
		# NOTE: 処理ごとに分けない

		try:
			# 矢印の距離が短すぎる領域は除去
			# NOTE: しきい値変えれる
			if (i > 7):
				idx_list.append([])
				# # 矢印の描画
				# cv2.arrowedLine(
				# 	img=self.ortho,     # 画像
				# 	pt1=(cx, cy),       # 始点
				# 	pt2=(x, y),         # 終点
				# 	color=(20,20,180),  # 色
				# 	thickness=3,        # 太さ
				# 	tipLength=0.3       # 矢先の長さ
				# )
		except:
			pass

		flow_idx_list.append(idx_list)
	
	
	return flow_idx_list
	
	# cv2.imwrite("test_map.png", img)


@staticmethod
def detect_flow(deg):
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


@tool.stop_watch
def make_map_v2(self):
	"""
	土砂移動図の作成
	"""
	# 各注目領域に対して処理を行う
	for area in self.region:
		# 注目領域の重心
		cx, cy = int(area["cx"]), int(area["cy"])

		## 隣接領域かどうかを判別
		if (is_neighbor()):


			## 傾斜方向が上から下の領域を抽出
			if (is_direction()):


				## 侵食と堆積の組み合わせを抽出
				if (is_sediment()):
				

					## 災害？前？後？地形より土砂移動推定
					# NOTE: 災害前後の地形のどちらを使用するか要検討

					# 矢印の描画
					tool.draw_vector(self, (cy, cx))

					# 注目領域の重心標高値
					elevation_value = self.dsm_uav[cy, cx]


	# NOTE:::
	# detect_flowのように10方向で精度評価＋できればベクトル量（流出距離も）
	# ラベル画像の領域単位で，そこからどこに流れてそうか正解画像（矢印図）を作成

	# NOTE:::
	# できればオプティカルフローやPIV解析，3D-GIV解析，特徴量追跡等も追加する


	# 土砂移動図の作成


@tool.stop_watch
def make_map(self, move_list):
	"""
	土砂移動図の作成

	list: 土砂移動推定箇所のリスト
	"""
	# 解像度（cm）
	resolution = 7.5

	# 水平方向の土砂移動図を作成
	# TODO: 関数にしたい
	for i, move in enumerate(move_list):
		if (move != []):
			# 注目領域の重心座標
			cx, cy = int(self.region[i]["cx"]), int(self.region[i]["cy"])

			# 土砂の流出方向へ矢印を描画
			for m in move:
				# 流出先の重心座標
				_cx, _cy = int(self.region[m]["cx"]), int(self.region[m]["cy"])
				cv2.arrowedLine(
					img=self.ortho,     # 画像
					pt1=(cx, cy),       # 始点
					pt2=(_cx, _cy),     # 終点
					color=(20,20,180),  # 色
					thickness=2,        # 太さ
					tipLength=0.4       # 矢先の長さ
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
				# # TODO: DSMで良いか検討
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
