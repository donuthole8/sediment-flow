import cv2
import numpy as np
import pymeanshift as pms

from modules import tif
from modules import tool
from modules import process


class ImageOp():

	def __init__(self, path_list):
		"""
		初期化メソッド

		path_list: 入力パス
		"""
		# 入力パス
		self.path_list = path_list

		# 画像
		self.dsm_uav     = tif.load_tif(path_list[0]).astype(np.float32)
		self.dsm_heli    = tif.load_tif(path_list[1]).astype(np.float32)
		self.dem         = tif.load_tif(path_list[2]).astype(np.float32)
		self.degree      = tif.load_tif(path_list[3]).astype(np.float32)
		self.mask        = cv2.imread(path_list[4], cv2.IMREAD_GRAYSCALE)
		self.ortho       = cv2.imread(path_list[5])
		self.maked_ortho = cv2.imread(path_list[5])
		self.gradient    = None
		self.div_img     = None
		self.dsm_sub     = None

		# 画像サイズ
		self.size_3d   = self.dsm_uav.shape
		self.size_2d   = (self.size_3d[1], self.size_3d[0])
		self.s_size_2d = (int(self.size_3d[1] / 2), int(self.size_3d[0] / 2))


	def dem2gradient(self, mesh_size):
		"""
		DEMを勾配データへ変換

		mesh_size: DEMのメッシュサイズ
		"""
		# 勾配データの算出
		grad = process.calc_gradient(self, mesh_size)

		# 3次元に変換
		self.gradient = cv2.merge((grad, grad, grad))

		# 画像を保存
		tif.save_tif(self.gradient, "dem.tif", "angle.tif")

		return


	def norm_degree(self):
		"""
		入力された傾斜方位（0-255）を実際の角度（0-360）に正規化
		"""
		# 0-360度に変換
		self.degree = self.degree / 255 * 360

		return


	def norm_degree_v2(self):
		"""
		入力された傾斜方位（負値含む）を実際の角度（0-360）に正規化
		"""
		# 最大・最小
		deg_max = np.nanmax(self.degree)
		deg_min = np.nanmin(self.degree)

		# 0-360度に変換
		self.degree = (self.degree - deg_min) / (deg_max - deg_min) * 360

		return


	def resampling_dsm(self):
		"""
		UAVのNoData部分を航空画像DSM・DEMから切り取り・解像度のリサンプリング
		"""
		# 3次元に変更
		self.mask = cv2.merge((self.mask, self.mask, self.mask))

		# バイキュービック補間で解像度の統一
		self.dsm_heli = cv2.resize(self.dsm_heli, self.size_2d, interpolation=cv2.INTER_CUBIC)
		self.dem      = cv2.resize(self.dem,      self.size_2d, interpolation=cv2.INTER_CUBIC)
		self.degree   = cv2.resize(self.degree,   self.size_2d, interpolation=cv2.INTER_CUBIC)
		self.mask     = cv2.resize(self.mask,     self.size_2d, interpolation=cv2.INTER_CUBIC)
		self.ortho    = cv2.resize(self.ortho,    self.size_2d, interpolation=cv2.INTER_CUBIC)

		# UAV画像のDSMの最小値を算出（領域外の透過背景値）
		background_pix = np.min(self.dsm_uav)

		# 航空写真DSM・DEMの不要範囲除去（UAVでの透過背景消去）
		idx = np.where(self.dsm_uav == background_pix)
		self.dsm_uav[idx]  = np.nan
		self.dsm_heli[idx] = np.nan
		self.dem[idx]      = np.nan 
		self.degree[idx]   = np.nan
		self.mask[idx]     = 0
		self.ortho[idx]    = 0

		# 画像の保存
		tool.save_resize_image("resamp_heli.png",  self.dsm_heli, self.s_size_2d)
		tool.save_resize_image("resamp_dem.png",   self.dem,      self.s_size_2d)
		tool.save_resize_image("resamp_deg.png",   self.degree,   self.s_size_2d)
		tool.save_resize_image("resamp_mask.png",  self.mask,     self.s_size_2d)
		tool.save_resize_image("resamp_ortho.png", self.ortho,    self.s_size_2d)

		# 1次元に戻す
		self.mask = cv2.split(self.mask)[0]

		return


	@tool.stop_watch
	def norm_mask(self, area_th, scale):
		"""
		マスク画像の前処理

		area_th: 面積の閾値（スケール済みであるので(area_th/scale)が閾値となる）
		scale: 輪郭抽出の際の拡大倍率
		"""
		# 画像反転
		self.mask = cv2.bitwise_not(self.mask)

		# モルフォロジー処理
		# TODO: カーネルサイズと実行回数の調整
		# morpho_mask = morphology(mask, 15, 15)
		process.morphology(self, 3, 3)

		# 輪郭抽出
		# contours, self.mask = process.get_norm_contours(morpho_mask, scale, 15)
		contours = process.get_norm_contours(self, scale, 3)

		# 面積が閾値未満の領域を除去
		process.remove_small_area(self, contours, area_th, scale)

		return


	def extract_sediment(self):
		"""
		標高データから土砂領域を抽出
		"""
		# 土砂マスクを用いて土砂領域以外を除去
		process.extract_sediment(self)

		return


	@tool.stop_watch
	def divide_area(self, spatial_radius, range_radius, min_density):
		"""
		オルソ画像の領域分割を行う

		spatial_radius: 空間半径
		range_radius: 範囲半径
		min_density: 最小密度
		"""
		# 領域分割
		# TODO: マスク済みオルソ画像を用いると良いかも
		lab_img, _, number_regions = pms.segment(cv2.cvtColor(self.ortho, cv2.COLOR_BGR2Lab), spatial_radius, range_radius,min_density)
		# lab_img, _, number_regions = pms.segment(cv2.cvtColor(self.masked_ortho, cv2.COLOR_BGR2Lab), spatial_radius, range_radius,min_density)
		self.div_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

		# 画像の保存
		cv2.imwrite('./outputs/meanshift.png', self.div_img)

		print("- meanshift-num:", number_regions)

		return


	@tool.stop_watch
	def calc_contours(self):
		"""
		csvに保存された領域の座標データより領域データを算出
		"""
		# 領域データ読み込み
		region_list = tool.load_csv("./area_data/pms_cords.csv")

		# 各領域をキャンパスに描画し1つずつ領域データを抽出
		process.get_pms_contours(self, region_list)

		return


	def extract_building(self):
		"""
		円形度より建物領域を抽出する
		"""
		# 領域・座標データ読み込み
		region_list = tool.load_csv("./area_data/region.csv")
		cords_list  = tool.load_csv("./area_data/pms_cords.csv")

		# 建物領域を抽出
		# process.extract_building(self, self.div_img, self.ortho, region_list, cords_list, self.size_2d)
		process.extract_building(self, region_list, cords_list)

		return


	def norm_elevation_0to1(self):
		"""
		DSM標高値を0から1に正規化する
		"""
		# 最小値・最大値算出
		min_uav,  max_uav  = tool.calc_min_max(self.dsm_uav)
		min_heli, max_heli = tool.calc_min_max(self.dsm_heli)

		# 正規化処理
		self.dsm_uav  = (self.dsm_uav - min_uav) / (max_uav - min_uav)
		self.dsm_heli = (self.dsm_heli - min_heli) / (max_heli - min_heli)

		# 画像を保存
		tool.save_resize_image("normed_uav.png",  self.dsm_uav,  self.s_size_2d)
		tool.save_resize_image("normed_heli.png", self.dsm_heli, self.s_size_2d)

		return


	def norm_elevation_sd(self):
		"""
		DSM標高値を標準偏差によってに正規化する
		"""
		# 平均・標準偏差算出
		ave_dsm_uav, sd_dsm_uav   = tool.calc_ave_sd(self.dsm_uav)
		ave_dsm_heli, sd_dsm_heli = tool.calc_ave_sd(self.dsm_heli)

		print("- uav  ave, sd :", ave_dsm_uav,  sd_dsm_uav)
		print("- heli ave, sd :", ave_dsm_heli, sd_dsm_heli)

		# 標高最大値（山間部の頂上・植生頂上）の変化は無いと仮定する
		# （標高最小値（海・海岸）は海抜0mと仮定する）
		# UAVのDEMを0-180mに正規化
		# 最大標高180m，最小標高0mとする
		max_height_uav  = 190
		max_height_heli = 185

		self.dsm_uav  = (self.dsm_uav - ave_dsm_uav) / sd_dsm_uav  * max_height_uav
		self.dsm_heli = (self.dsm_heli - ave_dsm_heli) / sd_dsm_heli * max_height_heli

		return


	def norm_elevation_meter(self):
		"""
		DEMの標高をDSMにマッチングさせ標高値をm単位で対応付ける
		"""
		# 最小値・最大値算出
		min_uav,  max_uav  = tool.calc_min_max(self.dsm_uav)
		min_heli, max_heli = tool.calc_min_max(self.dsm_heli)
		min_dem,  max_dem  = tool.calc_min_max(self.dem)
		# print("- uav-range  :", min_uav , max_uav)    # 1.0 255.0
		# print("- heli-range :", min_heli, max_heli)   # 52.16754 180.19545
		# print("- dem-range  :", min_dem , max_dem)    # -0.54201436 146.51208

		# 植生を加味
		veg_height = 15
		# veg_height = 10

		# 正規化処理
		self.dsm_uav  = (self.dsm_uav-min_uav) / (max_uav-min_uav) * (max_dem + veg_height)
		self.dsm_heli = (self.dsm_heli-min_heli) / (max_heli-min_heli) * (max_dem + veg_height)

		# 画像を保存
		tool.save_resize_image("normed_uav.png",  self.dsm_uav,  self.s_size_2d)
		tool.save_resize_image("normed_heli.png", self.dsm_heli, self.s_size_2d)

		return


	def calc_sedimentation(self):
		"""
		災害前後での標高差分を算出
		DEMの標高をDSMにマッチングさせ標高値を対応付ける
		"""
		# 標高差分を算出
		self.dsm_sub = self.dsm_uav - self.dsm_heli

		# 土砂領域以外を除外
		idx_mask = np.where(self.mask == 255)
		self.dsm_sub[idx_mask] = np.nan

		# 堆積領域と侵食領域で二値化
		dsm_bin = process.bin_2area(self)

		# 画像の保存
		tif.save_tif(self.dsm_sub, "dsm_uav.tif", "dsm_sub.tif")
		tool.save_resize_image("dsm_bin.png", dsm_bin, self.s_size_2d)

		return


	def calc_movement(self):
		"""
		土砂移動の推定
		"""
		# 上流が侵食かつ下流が堆積の領域の組を全て抽出
		## 領域が隣接している領域の組を全て抽出
		area_list1 = process.extract_neighbor()

		## 傾斜方向が上から下の領域の組を全て抽出
		area_list2 = process.extract_direction(self)

		## 侵食と堆積の領域の組を全て抽出
		area_list3 = process.extract_sub(self)

		# 上記3つの条件を全て満たす領域の組を抽出
		# area_list = tool.and_operation_2(area_list1, area_list2)
		area_list = tool.and_operation(area_list1, area_list2, area_list3)

		# 土砂移動図の作成
		process.make_map(self, area_list)
		print("- area-list :", area_list)
		print("- area-num  :", len(area_list))

		return

