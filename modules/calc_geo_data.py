import cv2
import math
import numpy as np
from tqdm import trange

from modules.utils import tiff_util
from modules.utils import calculation_util
from modules.utils import image_util
from modules.image_data import ImageData


class CalcGeoData():
	def dem2gradient(self, image: ImageData, mesh_size: int) -> None:
		""" DEMを勾配データへ変換

		Args:
				image (ImageData): 画像データ
				mesh_size (int): DEMのメッシュサイズ
		"""
		# TODO: tiffutil.save_tifでQGISの位置情報に対応できていない
		# 勾配データの算出
		gradient = self.__dem2gradient(image, mesh_size)

		# データ保存
		image.gradient = gradient

		# 画像を保存
		tiff_util.save_tif(image.gradient, "gradient.tif")

		return


	@staticmethod
	def norm_degree(image: ImageData) -> None:
		"""
		入力された傾斜方位（0-255）を実際の角度（0-360）に正規化
		"""
		# 0-360度に変換
		image.degree = image.degree / 255 * 360

		return


	@staticmethod
	def norm_degree_v2(image: ImageData) -> None:
		"""
		入力された傾斜方位（負値含む）を実際の角度（0-360）に正規化
		"""
		# 最大・最小
		deg_max = np.nanmax(image.degree)
		deg_min = np.nanmin(image.degree)

		# 0-360度に変換
		image.degree = (image.degree - deg_min) / (deg_max - deg_min) * 360

		return


	@staticmethod
	def __dem2gradient(image: ImageData, size_mesh: int) -> None:
		""" 勾配データを算出する

		Args:
				image (ImageData): 画像データ
				size_mesh (int): DEMのメッシュサイズ
		"""
		max = 0
		index = [-1,1]
		height, width = image.dem.shape[:2]
		gradient = np.zeros((height, width))
		dem = np.pad(image.dem, 1, mode = 'edge')

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
		
		return gradient.astype(np.int8)


	@staticmethod
	def norm_elevation_0to1(image: ImageData) -> None:
		"""
		DSM標高値を0から1に正規化する

		image: 画像データ
		"""
		# 最小値・最大値算出
		min_uav,  max_uav  = calculation_util.calc_min_max(image.dsm_after)
		min_heli, max_heli = calculation_util.calc_min_max(image.dem_before)

		# 正規化処理
		image.dsm_after  = (image.dsm_after - min_uav) / (max_uav - min_uav)
		image.dem_before = (image.dem_before - min_heli) / (max_heli - min_heli)

		# 画像を保存
		image_util.save_resize_image("normed_uav.png",  image.dsm_after,  image.s_size_2d)
		image_util.save_resize_image("normed_heli.png", image.dem_before, image.s_size_2d)

		return


	@staticmethod
	def norm_elevation_sd(image: ImageData) -> None:
		""" DSM標高値を標準偏差によってに正規化する

		Args:
				image (ImageData): 画像データ
		"""
		# 平均・標準偏差算出
		ave_dsm_after, sd_dsm_after   = calculation_util.calc_mean_sd(image.dsm_after)
		ave_dem_before, sd_dem_before = calculation_util.calc_mean_sd(image.dem_before)

		print("- uav  ave, sd :", ave_dsm_after,  sd_dsm_after)
		print("- heli ave, sd :", ave_dem_before, sd_dem_before)

		# 標高最大値（山間部の頂上・植生頂上）の変化は無いと仮定する
		# （標高最小値（海・海岸）は海抜0mと仮定する）
		# UAVのDEMを0-180mに正規化
		# 最大標高180m，最小標高0mとする
		max_height_uav  = 190
		max_height_heli = 185

		image.dsm_after  = (image.dsm_after - ave_dsm_after) / sd_dsm_after  * max_height_uav
		image.dem_before = (image.dem_before - ave_dem_before) / sd_dem_before * max_height_heli

		return


	@staticmethod
	def norm_elevation_meter(image: ImageData) -> None:
		""" DEMの標高をDSMにマッチングさせ標高値をm単位で対応付ける

		Args:
				image (ImageData): 画像データ
		"""
		# 最小値・最大値算出
		min_after,  max_after  = calculation_util.calc_min_max(image.dsm_after)
		min_before, max_before = calculation_util.calc_min_max(image.dem_before)
		min_dem,    max_dem    = calculation_util.calc_min_max(image.dem)

		print("->>>>>>> 正規化前")
		image_util.show_max_min(image)

		# 正規化処理
		# TODO: 修正・改善
		# image.dsm_after  = (image.dsm_after - min_before) / (max_before - min_before) * max_dem
		# image.dsm_after  = (image.dsm_after - min_after) / (max_after - min_after) * max_before


		# DEMの場合
		image.dsm_after  = min_before + (max_before - min_before) * ((image.dsm_after - min_after) / (max_after - min_after)) 



		# # ヘリとUAVの場合
		# print("dem", min_dem ,max_dem)
		# image.dsm_after   = min_dem + (max_dem - min_dem) * ((image.dsm_after - min_after) / (max_after - min_after)) 
		# image.dem_before  = min_dem + (max_dem - min_dem) * ((image.dem_before - min_before) / (max_before - min_before)) 

		# print("after", np.nanmin(image.dsm_after), np.nanmax(image.dsm_after))
		# print("before", np.nanmin(image.dem_before), np.nanmax(image.dem_before))



		print("->>>>>>> 正規化後")
		image_util.show_max_min(image)

		# 画像を保存
		image_util.save_resize_image("normed_uav.png",  image.dsm_after,  image.s_size_2d)
		image_util.save_resize_image("normed_heli.png", image.dem_before, image.s_size_2d)

		# 画像の保存
		cv2.imwrite("./output/meterd_uav_dsm.tif", image.dsm_after)
		cv2.imwrite("./output/meterd_uav_heli.tif", image.dem_before)

		return


	def norm_coord(self, image: ImageData):
		""" 標高座標の最適化

		Args:
				image (ImageData): 画像データ
		"""
		return 
