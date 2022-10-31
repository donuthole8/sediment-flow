import cv2
import math
import numpy as np
from tqdm import trange

from modules import tif
from modules import tool
from modules.image_data import ImageData


class CalcGeoData():
	def dem2gradient(self, image: ImageData, mesh_size: int) -> None:
		"""
		DEMを勾配データへ変換

		image: 画像データ
		mesh_size: DEMのメッシュサイズ
		"""
		# 勾配データの算出
		grad = self.__dem2gradient(image, mesh_size)

		# 3次元に変換
		self.gradient = cv2.merge((grad, grad, grad))

		# 画像を保存
		tif.save_tif(image.gradient, "dem.tif", "angle.tif")

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
		"""
		傾斜データを算出する
		
		image: 画像データ
		size_mesh: DEMのメッシュサイズ
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
		min_uav,  max_uav  = tool.calc_min_max(image.dsm_uav)
		min_heli, max_heli = tool.calc_min_max(image.dsm_heli)

		# 正規化処理
		image.dsm_uav  = (image.dsm_uav - min_uav) / (max_uav - min_uav)
		image.dsm_heli = (image.dsm_heli - min_heli) / (max_heli - min_heli)

		# 画像を保存
		tool.save_resize_image("normed_uav.png",  image.dsm_uav,  image.s_size_2d)
		tool.save_resize_image("normed_heli.png", image.dsm_heli, image.s_size_2d)

		return


	@staticmethod
	def norm_elevation_sd(image: ImageData) -> None:
		"""
		DSM標高値を標準偏差によってに正規化する

		image: 画像データ
		"""
		# 平均・標準偏差算出
		ave_dsm_uav, sd_dsm_uav   = tool.calc_ave_sd(image.dsm_uav)
		ave_dsm_heli, sd_dsm_heli = tool.calc_ave_sd(image.dsm_heli)

		print("- uav  ave, sd :", ave_dsm_uav,  sd_dsm_uav)
		print("- heli ave, sd :", ave_dsm_heli, sd_dsm_heli)

		# 標高最大値（山間部の頂上・植生頂上）の変化は無いと仮定する
		# （標高最小値（海・海岸）は海抜0mと仮定する）
		# UAVのDEMを0-180mに正規化
		# 最大標高180m，最小標高0mとする
		max_height_uav  = 190
		max_height_heli = 185

		image.dsm_uav  = (image.dsm_uav - ave_dsm_uav) / sd_dsm_uav  * max_height_uav
		image.dsm_heli = (image.dsm_heli - ave_dsm_heli) / sd_dsm_heli * max_height_heli

		return


	@staticmethod
	def norm_elevation_meter(image: ImageData) -> None:
		"""
		DEMの標高をDSMにマッチングさせ標高値をm単位で対応付ける

		image: 画像データ
		"""
		# 最小値・最大値算出
		min_uav,  max_uav  = tool.calc_min_max(image.dsm_uav)
		min_heli, max_heli = tool.calc_min_max(image.dsm_heli)
		min_dem,  max_dem  = tool.calc_min_max(image.dem)
		# print("- uav-range  :", min_uav , max_uav)    # 1.0 255.0
		# print("- heli-range :", min_heli, max_heli)   # 52.16754 180.19545
		# print("- dem-range  :", min_dem , max_dem)    # -0.54201436 146.51208

		# 植生を加味
		# TODO: 植生領域を除去し，植生による誤差を除去
		# veg_height = 0
		veg_height = 15
		# veg_height = 10

		# 正規化処理
		# TODO: 修正・改善
		image.dsm_uav  = (image.dsm_uav - min_uav) / (max_uav - min_uav) * (max_dem + veg_height)
		# self.dsm_heli = (self.dsm_heli - min_heli) / (max_heli - min_heli) * (max_dem + veg_height)

		# 画像を保存
		tool.save_resize_image("normed_uav.png",  image.dsm_uav,  image.s_size_2d)
		tool.save_resize_image("normed_heli.png", image.dsm_heli, image.s_size_2d)

		# 画像の保存
		cv2.imwrite("./output/meterd_uav_dsm.tif", image.dsm_uav)
		cv2.imwrite("./output/meterd_uav_heli.tif", image.dsm_heli)

		return
