import cv2
import numpy as np
import pymeanshift as pms

from modules import tif
from modules import tool
from modules import process


class ImageOp():
	def __init__(self, path_list: list[str]) -> None:
		"""
		初期化メソッド

		path_list: 入力パス
		"""
		# 入力パス
		self.path_list = path_list

		# 画像
		# self.dsm_uav     = tif.load_tif(path_list[0]).astype(np.float32)
		# self.dsm_heli    = tif.load_tif(path_list[1]).astype(np.float32)
		# self.dem         = tif.load_tif(path_list[2]).astype(np.float32)
		# self.degree      = tif.load_tif(path_list[3]).astype(np.float32)

		# tif画像
		self.dsm_uav     = cv2.imread(path_list[0], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.dsm_heli    = cv2.imread(path_list[1], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.dem         = cv2.imread(path_list[2], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.degree      = cv2.imread(path_list[3], cv2.IMREAD_ANYDEPTH).astype(np.float32)

		# 画像
		self.mask          = cv2.imread(path_list[4], cv2.IMREAD_GRAYSCALE)
		self.ortho         = cv2.imread(path_list[5]).astype(np.float32)
		self.masked_ortho  = None
		self.gradient      = None
		self.div_img       = None
		self.label_table   = None
		self.dissimilarity = None
		self.edge          = None
		self.bld_mask      = None
		self.normed_dsm    = None
		self.dsm_sub       = None

		# 画像サイズ
		self.size_3d   = self.dsm_uav.shape
		self.size_2d   = (self.size_3d[1], self.size_3d[0])
		self.s_size_2d = (int(self.size_3d[1] / 2), int(self.size_3d[0] / 2))

		# 領域データ
		self.pms_coords = []
		self.pms_pix    = []
		self.region     = []
		self.building   = []


	def dem2gradient(self, mesh_size: int) -> None:
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


	def norm_degree(self) -> None:
		"""
		入力された傾斜方位（0-255）を実際の角度（0-360）に正規化
		"""
		# 0-360度に変換
		self.degree = self.degree / 255 * 360

		return


	def norm_degree_v2(self) -> None:
		"""
		入力された傾斜方位（負値含む）を実際の角度（0-360）に正規化
		"""
		# 最大・最小
		deg_max = np.nanmax(self.degree)
		deg_min = np.nanmin(self.degree)

		# 0-360度に変換
		self.degree = (self.degree - deg_min) / (deg_max - deg_min) * 360

		return


	def resampling_dsm(self) -> None:
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
		cv2.imwrite("dem_powerpo.tif", self.dem)
		cv2.imwrite("dem_powerpo.png", self.dem)

		# tool.save_resize_image("resamp_heli.png",  self.dsm_heli, self.s_size_2d)
		# tool.save_resize_image("resamp_dem.png",   self.dem,      self.s_size_2d)
		# tool.save_resize_image("resamp_deg.png",   self.degree,   self.s_size_2d)
		# tool.save_resize_image("resamp_mask.png",  self.mask,     self.s_size_2d)
		# tool.save_resize_image("resamp_ortho.png", self.ortho,    self.s_size_2d)

		# 1次元に戻す
		self.mask = cv2.split(self.mask)[0]

		return


	@tool.stop_watch
	def norm_mask(self, area_th: int, scale: int) -> None:
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


	def apply_mask(self) -> None:
		"""
		マスク画像を適用し土砂領域のみを抽出
		"""
		# 土砂マスクを用いて土砂領域以外を除去
		self.masked_ortho = process.masking(self, self.ortho, self.mask)

		return


	@tool.stop_watch
	def divide_area(
		self, 
		spatial_radius: float, 
		range_radius: float, 
		min_density: float
	) -> None:
		"""
		オルソ画像の領域分割を行う

		spatial_radius: 空間半径
		range_radius: 範囲半径
		min_density: 最小密度
		"""
		# Lab表色系に変換
		# NOTE: RGB表色系のままでも良いかも

		# lab_img = cv2.cvtColor(self.ortho, cv2.COLOR_BGR2Lab)
		# lab_img = cv2.cvtColor(self.masked_ortho, cv2.COLOR_BGR2Lab)

		# PyMeanShiftによる域分割
		# TODO: マスク済みオルソ画像を用いると良いかも
		img, _, number_regions = pms.segment(
			self.ortho.astype(np.uint8), 
			# self.masked_ortho.astype(np.uint8), 
			spatial_radius, 
			range_radius, 
			min_density
		)

		# RGB表色系に変換
		# self.div_img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR).astype(np.float32)
		self.div_img = img

		# 画像の保存
		cv2.imwrite('./outputs/meanshift.png', self.div_img)

		print("- meanshift-num:", number_regions)

		return


	@tool.stop_watch
	def calc_contours(self) -> None:
		"""
		csvに保存された領域の座標データより領域データを算出
		"""
		# 領域データの保存
		tool.csv2self(self)

		# ラベリングテーブルの作成
		process.create_label_table(self)

		# 各領域をキャンパスに描画し1つずつ領域データを抽出
		process.get_pms_contours(self)

		return


	@tool.stop_watch
	def texture_analysis(self) -> None:
		"""
		テクスチャ解析
		"""
		# テクスチャ解析
		# TODO: オルソ画像でなく領域分割済み画像等でやっても良いかも
		process.texture_analysis(self)

		return


	def edge_detection(self) -> None:
		"""
		エッジ抽出
		"""
		# エッジ抽出
		process.edge_detection(self, 100, 200)

		return


	@tool.stop_watch
	def extract_building(self) -> None:
		"""
		円形度より建物領域を抽出する
		"""
		# 建物領域を抽出
		process.extract_building(self)

		return


	def norm_building(self) -> None:
		"""
		建物領域の標高値を地表面と同等に補正する
		"""
		# 標高データを補正
		process.norm_building(self)

		return


	def norm_elevation_0to1(self) -> None:
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


	def norm_elevation_sd(self) -> None:
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


	def norm_elevation_meter(self) -> None:
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
		# veg_height = 0
		veg_height = 15
		# veg_height = 10

		# 正規化処理
		self.dsm_uav  = (self.dsm_uav - min_uav) / (max_uav - min_uav) * (max_dem + veg_height)
		# self.dsm_heli = (self.dsm_heli - min_heli) / (max_heli - min_heli) * (max_dem + veg_height)

		# 画像を保存
		tool.save_resize_image("normed_uav.png",  self.dsm_uav,  self.s_size_2d)
		tool.save_resize_image("normed_heli.png", self.dsm_heli, self.s_size_2d)

		# 画像の保存
		cv2.imwrite("meterd_uav_dsm.tif", self.dsm_uav)
		cv2.imwrite("meterd_uav_heli.tif", self.dsm_heli)

		return


	def calc_sedimentation(self) -> None:
		"""
		災害前後での標高差分を算出
		DEMの標高をDSMにマッチングさせ標高値を対応付ける
		"""
		# 標高差分を算出
		self.dsm_sub = self.dsm_uav - self.dsm_heli

		print("self.mask", self.mask.shape)
		print("self.mask", self.dsm_sub.shape)

		# 土砂領域以外を除外
		idx_mask = np.where(self.mask[0] == 255)
		self.dsm_sub[idx_mask] = np.nan

		# 堆積領域と侵食領域で二値化
		dsm_bin = process.binarize_2area(self)

		# 画像の保存
		tif.save_tif(self.dsm_sub, "dsm_uav.tif", "dsm_sub.tif")
		tool.save_resize_image("dsm_bin.png", dsm_bin, self.s_size_2d)

		return


	@tool.stop_watch
	def calc_movement(self) -> None:
		"""
		土砂移動の推定
		"""
		# 各注目領域に対して処理を行う
		for region in self.region:
			# # 注目領域の重心
			# cx, cy = int(region["cx"]), int(region["cy"])

			# # 注目領域のラベル番号
			# label = region["label"]
			# print("label:", label)

			# TODO: 順番を考えることによって処理を減らせそう
			# TODO: 最初に4つの処理で共通に必要なデータを取得することでメモリ使用等を減らせそう


			## 8方向に対しての隣接領域を取得
			neighbor_labels = process.extract_neighbor(self, region)

			## 傾斜方向が上から下の領域を抽出
			downstream_labels = process.extract_downstream(self, region, neighbor_labels)

			## 侵食と堆積の組み合わせの領域を抽出
			sediment_labels = process.extract_sediment(self, region, downstream_labels)

			# print("neighbor  :", neighbor_labels)
			# print("downstream:", downstream_labels)
			print("sediment  :", sediment_labels)
			# print("")

			# return


			# return

				# 	## 侵食と堆積の組み合わせを抽出
				# 	if (process.is_sediment()):
					
				# 		## 災害？前？後？地形より土砂移動推定
				# 		# NOTE: 災害前後の地形のどちらを使用するか要検討
				# 		if (process.estimate_flow()):

				# 			# 矢印の描画
				# 			tool.draw_vector(self, (cy, cx))

				# 			# 注目領域の重心標高値
				# 			elevation_value = self.dsm_uav[cy, cx]


		# NOTE:::
		# detect_flowのように10方向で精度評価＋できればベクトル量（流出距離も）
		# ラベル画像の領域単位で，そこからどこに流れてそうか正解画像（矢印図）を作成

		# NOTE:::
		# できればオプティカルフローやPIV解析,3D-GIV解析，特徴量追跡等も追加する


		# 土砂移動図の作成

		return
		

