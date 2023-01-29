import cv2
import math
import numpy as np
import pymeanshift as pms

from modules.utils import common_util
from modules.utils import csv_util
from modules.utils import image_util
from modules.utils import tiff_util
from modules.utils import drawing_util
from modules.image_data import ImageData


class RegionProcessing():
	# 傾斜方位への座標(Δy, Δx)
	DIRECTION: list[int] = [
		(-1, 0),		# 北
		(-1, 1),		# 北東
		(0,  1),		# 東
		(1,  1),		# 南東
		(1,  0),		# 南
		(1,  -1),		# 南西
		(0,  -1),		# 西
		(-1, -1),		# 北西
	]


	@common_util.stop_watch
	def area_division(
			self, 
			image: ImageData, 
			spatial_radius: float, 
			range_radius: float, 
			min_density: float,
			clahe: tuple[float, tuple]
		) -> None:
		""" オルソ画像の領域分割を行う

		Args:
				image (ImageData): 画像データ
				spatial_radius (float): 空間半径
				range_radius (float): 範囲半径
				min_density (float): 最小密度
				clahe (tuple[float, tuple]): ヒストグラム平均化のパラメータ
		"""
		# ヒストグラム平均化
		img = self.equalization(image.ortho, clahe[0], clahe[1]).astype(np.uint8)
		cv2.imwrite('./outputs/' + image.experiment + '/equalization.png', img)

		# Lab表色系に変換
		# NOTE: RGB表色系のままでも良いかも

		# lab_img = cv2.cvtColor(self.ortho, cv2.COLOR_BGR2Lab)
		# lab_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2Lab)
		# lab_img = cv2.cvtColor(self.masked_ortho, cv2.COLOR_BGR2Lab)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

		# PyMeanShiftによる域分割
		# TODO: マスク済みオルソ画像を用いると良いかも
		img, _, number_regions = pms.segment(
			img,
			# image.ortho.astype(np.uint8), 
			# self.masked_ortho.astype(np.uint8), 
			spatial_radius, 
			range_radius, 
			min_density
		)

		# RGB表色系に変換
		# image.div_img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR).astype(np.float32)
		# image.div_img = img
		image.div_img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

		# 領域データを各実験フォルダに移動
		csv_util.move_area_data(image)

		# 画像の保存
		cv2.imwrite('./outputs/' + image.experiment + '/meanshift.png', image.div_img)

		print("- meanshift-num:", number_regions)

		return


	@staticmethod
	def equalization(img, clip_limit, tile_grid_size):
		# HSV表色系に変化
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv_img)

		# V値（明度）補正
		# clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(9, 9))
		clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
		new_v = (clahe.apply(v.astype(np.uint8))).astype(np.float32)

		# 画像を再生成
		hsv_clahe = cv2.merge((h, s, new_v))
		new_rgb_img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

		img = new_rgb_img

		return img


	@common_util.stop_watch
	def get_region_data(self, image: ImageData) -> None:
		"""	csvに保存された領域の座標データより領域データを算出

		Args:
				image (ImageData): 画像データ
		"""
		# 領域データの保存
		csv_util.csv2self(image)

		# ラベリングテーブルの作成
		self.__create_label_table(image)

		# 各領域をキャンパスに描画し1つずつ領域データを抽出
		self.__get_pms_contours(image)

		return


	@staticmethod
	def __create_label_table(image: ImageData) -> None:
		"""	ラベリングテーブルを作成

		Args:
				image (ImageData): 画像データ
		"""
		# 空画像を作成
		label_table = np.zeros(image.size_2d).astype(np.uint32)

		for region in image.pms_coords:
			# 注目領域のラベルID,座標を取得
			label, coords, _ = common_util.decode_area(region)

			# ラベルを付与
			for coord in coords:
				label_table[coord] = label

		# ラベルテーブルの保存
		image.label_table = label_table

		return


	@staticmethod
	def __get_pms_contours(image: ImageData) -> None:
		"""	各領域をキャンパスに描画し1つずつ領域データを抽出
				領域分割結果からラベリング画像を作成

		Args:
				image (ImageData): 画像データ
		"""
		# キャンパス描画
		label_img = np.zeros((image.size_2d[0], image.size_2d[1], 3))

		for region in image.pms_coords:
			# 領域データを取得
			label, coords, area = common_util.decode_area(region)

			# FIXME: util.coord2cont使う
			# 注目領域のマスク画像を作成
			label_img, mask = drawing_util.draw_label(image, label_img, coords)

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
			try:
				circularity = 4.0 * math.pi * area_float / (arc_len * arc_len)
			except:
				print("div by 0:", arc_len, area_float)
				circularity = 0.0

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
			image.region.append(data_list)

		# 画像を保存
		cv2.imwrite("./outputs/" + image.experiment + "/label.png", label_img)
		
		return


	def extract_building(
		self, 
		image: ImageData, 
		circularity_th: int,
		bld_polygon_th: int
		) -> np.ndarray:
		""" 建物領域を抽出する

		Args:
				image (ImageData): 画像データ
				circularity_th: 円形度閾値
				bld_polygon_th: 建物ポリゴン含有率閾値

		Returns:
				np.ndarray 建物領域データ
		"""
		# 閾値の設定
		self.circularity_th = circularity_th
		self.bld_polygon_th = bld_polygon_th

		# 建物領域を抽出
		image.bld_mask = self.__extract_building(image)

		return


	def __extract_building(self, image: ImageData) -> None:
		"""	建物領域を抽出

		Args:
				image (ImageData): 画像データ
		"""
		# 建物領域検出用画像
		bld_img = image.ortho.copy()

		# キャンパス描画
		cir_img  = np.zeros(image.size_2d)
		bld_mask = np.zeros(image.size_2d)

		for region, coords in zip(image.region, image.pms_coords):
			# 領域・重心・座標データを取得
			circularity  = int(float(region["circularity"]) * 255)
			centroid     = (region["cy"], region["cx"])
			_, coords, _  = common_util.decode_area(coords)

			# 円形度を大小で描画
			for coord in coords:
				cir_img[coord] = circularity

			# 建物領域の検出
			if self.__is_building(image, circularity, centroid, coords):
				# 塗りつぶし
				for coord in coords:
					bld_img[coord]  = [0, 0, 220]
					bld_mask[coord] = 255

				# 建物領域の保存
				image.building.append({
					"label": region["label"], 
					"cx":    region["cx"], 
					"cy":    region["cy"], 
					"coords": tuple(coords)
				})

		# 画像を保存
		cv2.imwrite("./outputs/" + image.experiment + "/circularity.png", cir_img)
		cv2.imwrite("./outputs/" + image.experiment + "/building.png", bld_img)
		cv2.imwrite("./outputs/" + image.experiment + "/building_mask.png", bld_mask)

		return bld_mask


	def __is_building(
			self, 
			image: ImageData, 
			circularity: float, 
			centroid: tuple[int, int], 
			coords: list[tuple]
		) -> bool:
		""" 建物領域かどうかを判別

		Args:
				image (ImageData): 画像データ
				circularity (float): 円形度
				centroid (tuple[int, int]): 該当領域の重心座標
				coords (list[tuple]): 領域座標群

		Returns:
				bool: 建物領域フラグ
		"""
		# # 注目領域の平均異質度を取得
		# dissimilarity = np.mean(image.dissimilarity[centroid])

		# 土砂マスクの範囲内か
		# if (self.__is_masking(image, centroid)):
		# 円形度
		if (circularity > self.circularity_th):
			# 建物ポリゴンと一致度が閾値以上あるか
			if (self.__is_building_polygon(image, coords)):
				return True

		return False


	@staticmethod
	def __is_masking(image: ImageData, centroid: tuple[int, int]):
		""" 土砂マスクの土砂領域かどうかを判別

		Args:
				image (ImageData): 画像データ
				centroid (tuple[int, int]): 該当領域の重心座標

		Returns:
				bool: 土砂領域フラグ
		"""
		if (image.mask[centroid] == 255):
			return True
		return False


	def __is_building_polygon(self, image: ImageData, coords: list[tuple]) -> bool:
		""" 建物ポリゴンと領域が閾値以上一致しているか判別

		Args:
				image (ImageData): 画像データ
				coords (list[tuple]): 領域座標群

		Returns:
				bool: 建物フラグ
		"""
		building_counter = 0
		region_counter = len(coords)

		for coord in coords:
			if (image.bld_gsi[coord] == 0):
				building_counter += 1

		# 建物ポリゴンとの一致率
		agreement = (building_counter / region_counter)

		if (agreement > self.bld_polygon_th):
			return True
		else:
			return False


	def norm_building(self, image: ImageData) -> None:
		"""	建物領域の標高値を地表面と同等に補正する

		Args:
				image (ImageData): 画像データ
		"""
		# 正規化後のDSM標高値
		image.normed_dsm = image.dsm_after.copy()

		# 建物領域毎に処理
		for bld in image.building:
			# 注目建物領域の隣接領域（建物領域・土砂マスク外は除く）の標高値を取得
			h = self.__get_neighbor_regions(image, bld["coords"])

			if (h != -1):
				# UAVのDSMの建物領域を周囲領域の地表面と同じ標高値にする
				for coords in bld["coords"]:
					image.normed_dsm[coords] = h

					# TODO: subも検討
					# image.dsm_sub[coords] -= 10

				tiff_util._save_tif(
					image.normed_dsm,
					image.path_list[0],
					"./outputs/" + image.experiment + "/normed_dsm3.tif"
				)

			else:
				pass

		# 建物領域データの開放
		image.building = None

		tiff_util._save_tif(
			image.normed_dsm,
			image.path_list[0],
			"./outputs/" + image.experiment + "/normed_dsm2.tif"
		)

		return


	def __get_neighbor_regions(self, image: ImageData, coords: tuple[int, int]) -> list[int]:
		"""	建物領域でない隣接領域ラベルを取得

		Args:
				image (ImageData): 画像データ
				coords (tuple[int, int]): 注目領域の座標群

		Returns:
				list[int]: 隣接領域のラベル
		"""
		# キャンパス描画
		campus = np.zeros(image.size_2d)

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

		# 輪郭の重心
		M = cv2.moments(contours[0])
		try:
			cx = int((M["m10"] / M["m00"]))
			cy = int((M["m01"] / M["m00"]))
		except:
			cx, cy = 0, 0

		# 面積
		area_float = cv2.contourArea(contours[0])
		area = int(area_float)

		# 輪郭の周囲長
		arc_len = cv2.arcLength(contours[0], True)

		# 疑似半径
		r1 = int(math.sqrt(area / math.pi))
		r2 = int(arc_len / (2 * math.pi))

		height = 0.0
		c = 0

		for i in range(0, 8):
			_cy, _cx = self.__get_neighbor_coordinate(
				self.DIRECTION[i],
				(r1+r2)/2 * 2,
				(cy, cx)
			)

			neighbor_coordinate = int(_cy), int(_cx)

			# 取得した隣接座標が画像領域内に存在するか
			if (common_util.is_index(image.dsm_after.shape, neighbor_coordinate)):
				# 隣接座標は建物領域でないか・土砂マスクの領域内か
					if (common_util.is_ground(image, neighbor_coordinate)):
						
						height += image.dsm_after[neighbor_coordinate]
						c += 1

		try:
			return height / c
		except:
			return -1
		
		# # 重複を削除
		# # FIXME: Noneがある場合があるので削除
		# return list(set(neighbor_region_labels))


	def __get_neighbor_coordinate(self, direction, r, centroids):
		""" 注目座標から指定距離移動した座標を取得

		direction: 注目方向
		r: 半径
		centroids(cy, cx): 注目座標の重心座標
		"""
		try:
			r_sqrt = r / math.sqrt(2)

			# 注目方向によって処理を変更
			if   (direction == self.DIRECTION[0]):		# 北
				return (
					centroids[0] - r, 
					centroids[1]
				)

			elif (direction == self.DIRECTION[1]):		# 北東
				return (centroids[0] - r_sqrt, centroids[1] + r_sqrt)

			elif (direction == self.DIRECTION[2]):		# 東
				return (
					centroids[0],
					centroids[1] + r
				)

			elif (direction == self.DIRECTION[3]):		# 南東
				return (centroids[0] + r_sqrt, centroids[1] + r_sqrt)

			elif (direction == self.DIRECTION[4]):		# 南
							return (
								centroids[0] + r, 
								centroids[1]
				)

			elif (direction == self.DIRECTION[5]):		# 南西
				return (centroids[0] + r_sqrt, centroids[1] - r_sqrt)

			elif (direction == self.DIRECTION[6]):		# 西
				return (
					centroids[0],
					centroids[1] - r
				)

			elif (direction == self.DIRECTION[7]):		# 北西
				return (centroids[0] - r_sqrt, centroids[1] - r_sqrt)

		except:
			# 領域内に１次関数が存在しない場合
			return (-1, -1)


	def __get_neighbor_height(self, image: ImageData, label_list: list[int]) -> float:
		"""	地表面領域の平均標高値を取得

		Args:
				image (ImageData): 画像データ
				label_list (list[int]): 地表面領域ラベルのリスト

		Returns:
				float: 標高値
		"""
		# return image.dsm_after[list(zip(*np.where(image.label_table == label_list[0])))[0]]

		# 平均標高値を取得
		mean_height = 0.0
		for label in label_list:
			idx = list(zip(*np.where(image.label_table == label)))
			mean_height += np.nanmean(image.dsm_after[idx])
			return mean_height

		return mean_height / len(label_list)

