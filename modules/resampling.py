import cv2
import numpy as np
from modules.image_data import ImageData


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
		image.dsm_heli = cv2.resize(image.dsm_heli, image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.dem      = cv2.resize(image.dem,      image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.degree   = cv2.resize(image.degree,   image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.mask     = cv2.resize(image.mask,     image.size_2d_xy, interpolation=cv2.INTER_CUBIC)
		image.ortho    = cv2.resize(image.ortho,    image.size_2d_xy, interpolation=cv2.INTER_CUBIC)

		# UAV画像のDSMの最小値を算出（領域外の透過背景値）
		background_pix = np.min(image.dsm_uav)

		# 航空写真DSM・DEMの不要範囲除去（UAVでの透過背景消去）
		idx = np.where(image.dsm_uav == background_pix)
		image.dsm_uav[idx]  = np.nan
		image.dsm_heli[idx] = np.nan
		image.dem[idx]      = np.nan 
		image.degree[idx]   = np.nan
		image.mask[idx]     = 0
		image.ortho[idx]    = 0

		# tool.save_resize_image("resamp_heli.png",  image.dsm_heli, image.s_size_2d)
		# tool.save_resize_image("resamp_dem.png",   image.dem,      image.s_size_2d)
		# tool.save_resize_image("resamp_deg.png",   image.degree,   image.s_size_2d)
		# tool.save_resize_image("resamp_mask.png",  image.mask,     image.s_size_2d)
		# tool.save_resize_image("resamp_ortho.png", image.ortho,    image.s_size_2d)

		# 1次元に戻す
		image.mask = cv2.split(image.mask)[0]

		return
