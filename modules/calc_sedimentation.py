import cv2
import numpy as np

from modules.utils import tiff_util
from modules.utils import image_util
from modules.image_data import ImageData


class CalcSedimentation():
	def __init__(self, image: ImageData) -> None:
		""" 災害前後での標高差分を算出
				DEMの標高をDSMにマッチングさせ標高値を対応付ける

		Args:
				image (ImageData): 画像データ
		"""
		# 標高差分を算出
		image.dsm_sub = image.dsm_after - image.dem_before

		# 土砂領域以外を除外
		idx_mask = np.where(image.mask[0] == 255)
		image.dsm_sub[idx_mask] = np.nan

		# 堆積領域と侵食領域で二値化
		dsm_bin = self.__binarize_2area(image)

		# 画像の保存
		tiff_util._save_tif(
			image.dsm_sub, 
			image.path_list[0], 
			"./outputs/dsm_sub.tif"
		)

		return


	@staticmethod
	def __binarize_2area(image: ImageData) -> np.ndarray:
		""" 堆積領域と侵食領域で二値化

		Args:
				image (ImageData): 画像データ

		Returns:
				np.ndarray: 2値化画像
		"""
		dsm_bin = image.dsm_sub.copy()
		idx = np.where(image.dsm_sub >  0)
		dsm_bin[idx] = 255
		idx = np.where(image.dsm_sub <= 0)
		dsm_bin[idx] = 0

		return dsm_bin


