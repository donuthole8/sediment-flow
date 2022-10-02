from tkinter.messagebox import NO
import cv2
import math
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import dist
from torch import ne
from tqdm import trange

from modules import tif
from modules import tool


# 方向への移動座標(Δy, Δx)
# FIXME: xとy逆の方が良いかも
DIRECTION = [
	(-1, 0),		# 北
	(-1, 1),		# 北東
	(0,  1),		# 東
	(1,  1),		# 南東
	(1,  0),		# 南
	(1,  -1),		# 南西
	(0,  -1),		# 西
	(-1, -1),		# 北西
]


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


def create_label_table(self) -> None:
	"""
	ラベリングテーブルを作成
	"""
	# 空画像を作成
	label_table = np.zeros((self.size_2d[1], self.size_2d[0])).astype(np.uint8)

	for region in self.pms_coords:
		# 注目領域のラベルID,座標を取得
		label, coords, _ = tool.decode_area(region)

		# ラベルを付与
		for coord in coords:
			label_table[coord] = label

	# ラベルテーブルの保存
	self.label_table = label_table

	return


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

		# FIXME: tool.coord2cont使う
		# 注目領域のマスク画像を作成
		label_img, mask = tool.draw_label(self, label_img, coords)

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


def get_neighbor_region(self, coords: tuple[int, int]) -> list[int]:
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


def extract_neighbor(self, region: tuple) -> list[int]:
	"""
	8方向で隣接している領域の組を全て抽出

	region: 注目領域の領域データ
	"""
	# 領域座標データを取得
	_, coordinates, _ = tool.decode_area(self.pms_coords[region["label"]])

	# 座標データから輪郭データを取得
	contour_coordinates = tool.coordinates2contours(self, coordinates)

	# 8方向それぞれの隣接領域を取得
	neighbor_region_labels = []
	for i in range(0, 8):
		# 輪郭の一番端座標からDIRECTION[i]の方向の座標を取得
		neighbor_coordinate = get_neighbor_coordinate(
			DIRECTION[i], 
			contour_coordinates, 
			(region["cy"], region["cx"])
		)

		# 取得した隣接座標が画像領域内に存在するか
		if (tool.is_index(self, neighbor_coordinate)):
			# ラベルIDを保存
			neighbor_region_labels.append(
				self.label_table[neighbor_coordinate]
			)

	# 重複を削除
	# FIXME: Noneがある場合があるので削除
	return list(set(neighbor_region_labels))


def get_neighbor_coordinate(
	direction: tuple[int, int], 
	contour_coordinates: list[tuple], 
	centroids: tuple[int, int]
) -> tuple[int, int]:
	"""
	注目座標の輪郭に隣接した領域の座標を1点取得

	direction: 注目方向
	contour_coordinates: 注目領域の輪郭座標群
	centroids: 注目座標の重心座標
	"""
	try:
		# 注目方向によって処理を変更
		if   (direction == DIRECTION[0]):		# 北
			return (
				np.min([c[0] for c in contour_coordinates]) + DIRECTION[0][0], 
				centroids[1] + DIRECTION[0][1]
			)

		elif (direction == DIRECTION[1]):		# 北東
			# y = -x + (Δy + Δx) の1次関数
			linear_function = [
				c for c in contour_coordinates 
				if (c[1] == (-1 * c[0]) + (centroids[1] + centroids[0]))
			]
			# 注目領域内でx座標の最も大きい座標を取得
			coord = max(linear_function, key=lambda x:x[1])

			return (coord[0] + DIRECTION[1][0], coord[1] + DIRECTION[1][1])

		elif (direction == DIRECTION[2]):		# 東
			return (
				centroids[0] + DIRECTION[2][0], 
				np.max([c[1] for c in contour_coordinates]) + DIRECTION[2][1]
			)

		elif (direction == DIRECTION[3]):		# 南東
			# y = x + (Δy - Δx) の1次関数
			linear_function = [
				c for c in contour_coordinates 
				if (c[1] == c[0] + (centroids[1] - centroids[0]))
			]
			# 注目領域内でx座標の最も大きい座標を取得
			coord = max(linear_function, key=lambda x:x[1])

			return (coord[0] + DIRECTION[3][0], coord[1] + DIRECTION[3][1])

		elif (direction == DIRECTION[4]):		# 南
			return (
				np.max([c[0] for c in contour_coordinates]) + DIRECTION[4][0], 
				centroids[1] + DIRECTION[4][1]
			)

		elif (direction == DIRECTION[5]):		# 南西
			# y = -x + (Δy + Δx) の1次関数
			linear_function = [
				c for c in contour_coordinates 
				if (c[1] == (-1 * c[0]) + (centroids[1] + centroids[0]))
			]
			# 注目領域内でy座標の最も大きい座標を取得
			coord = max(linear_function, key=lambda x:x[0])

			return (coord[0] + DIRECTION[5][0], coord[1] + DIRECTION[5][1])

		elif (direction == DIRECTION[6]):		# 西
			return (
				centroids[0] + DIRECTION[6][0], 
				np.min([c[1] for c in contour_coordinates]) + DIRECTION[6][1]
			)

		elif (direction == DIRECTION[7]):		# 北西
			# y = x + (Δy - Δx) の1次関数
			linear_function = [
				c for c in contour_coordinates 
				if (c[1] == c[0] + (centroids[1] - centroids[0]))
			]
			# 注目領域内でy座標の最も小さい座標を取得
			coord = min(linear_function, key=lambda x:x[0])

			return (coord[0] + DIRECTION[7][0], coord[1] + DIRECTION[7][1])
	except:
		# 領域内に１次関数が存在しない場合
		return (-1, -1)


@tool.stop_watch
def extract_direction(
	self, 
	region: tuple, 
	neighbor_labels: list[int]
) -> list[int]:
	"""
	隣接領域のうち傾斜方向が上流から下流の領域の組を全て抽出

	region: 注目領域の領域データ
	neighbor_labels: 隣接領域のラベルID
	"""
	# 重心標高値を取得
	# FIXME: 平均値にしたい
	centroid_elevation = self.dem[(region["cy"], region["cx"])]

	# 各隣接領域が下流かどうかを判別する
	downstream_region_labels = []
	for neighbor_label in neighbor_labels:

		# 隣接領域の重心座標を取得
		neighbor_centroids = (self.region[neighbor_label]["cy"], self.region[neighbor_label]["cx"])

		# 隣接領域の重心標高値を取得
		# FIXME: 平均値にしたい
		neighbor_centroid_elevation = self.dsm_uav[neighbor_centroids]

		# FIXME: ３チャンネルだったような気がする
		if (centroid_elevation > neighbor_centroid_elevation):
			downstream_region_labels.append(neighbor_label)
	
	# 重複ラベルは削除済み
	return downstream_region_labels








@tool.stop_watch
def extract_sub(self):
	"""
	侵食と堆積の領域の組を全て抽出
	"""
	pass


def estimate_flow(self):
	"""
	流出方向の予測
	"""
	pass



@staticmethod
def detect_flow(deg: float) -> tuple[int, int]:
	"""
	傾斜方向を画素インデックスに変換

	deg: 角度
	"""
	# 注目画素からの移動画素
	if   (math.isnan(deg)):
		return np.nan, np.nan
	elif (deg > 337.5) or  (deg <= 22.5):
		return DIRECTION[0]
	elif (deg > 22.5)  and (deg <= 67.5):
		return DIRECTION[1]
	elif (deg > 67.5)  and (deg <= 112.5):
		return DIRECTION[2]
	elif (deg > 112.5) and (deg <= 157.5):
		return DIRECTION[3]
	elif (deg > 157.5) and (deg <= 202.5):
		return DIRECTION[4]
	elif (deg > 202.5) and (deg <= 247.5):
		return DIRECTION[5]
	elif (deg > 247.5) and (deg <= 292.5):
		return DIRECTION[6]
	elif (deg > 292.5) and (deg <= 337.5):
		return DIRECTION[7]


