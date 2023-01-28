import cv2
import numpy as np

from modules.image_data import ImageData
from modules.utils import tiff_util
from modules.utils import image_util


class Resampling():
	@staticmethod
	def __init__(image: ImageData):
		""" UAVのNoData部分を航空画像DSM・DEMから切り取り・解像度のリサンプリング

		Args:
				image (ImageData): 画像データ
		"""
		# 3次元に変更
		image.mask = cv2.merge((image.mask, image.mask, image.mask))

		# バイキュービック補間で解像度の統一
		image.dem_before = cv2.resize(image.dem_before, image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.dem      = cv2.resize(image.dem,      image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.degree   = cv2.resize(image.degree,   image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.mask     = cv2.resize(image.mask,     image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.ortho    = cv2.resize(image.ortho,    image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.bld_gsi  = cv2.resize(image.bld_gsi,  image.size_2d_xy, interpolation=cv2.INTER_CUBIC)

		# UAV画像のDSMの最小値を算出（領域外の透過背景値）
		background_pix = np.min(image.dsm_after)

		# # 航空写真DSM・DEMの不要範囲除去（UAVでの透過背景消去）
		# idx = np.where(image.dsm_after == background_pix)
		# image.dsm_after[idx]  = np.nan
		# image.dem_before[idx] = np.nan
		# image.dem[idx]      = np.nan 
		# image.degree[idx]   = np.nan
		# image.mask[idx]     = 0
		# image.ortho[idx]    = 0
		# image.bld_gsi[idx]  = 255

		# 1次元に戻す
		image.mask = cv2.split(image.mask)[0]

		# # 画像の保存
		# tiff_util._save_tif(
		# 	cv2.merge((image.dem, image.dem, image.dem)), 
		# 	image.path_list[0],
		# 	"./outputs/" + image.experiment + "/dem_resampling.tif" 
		# )

		# # 画像サイズの確認
		# image_util.show_image_size(image)

		# # 入力画像の最大最小値確認
		# image_util.show_max_min(image)

		return
