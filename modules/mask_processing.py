import cv2
import numpy as np
import scipy.ndimage as ndimage

from modules.utils import common_util
from modules.utils import image_util
from modules.image_data import ImageData



class MaskProcessing():
	def apply_mask(self, image: ImageData) -> None:
		""" マスク画像を適用し土砂領域のみを抽出

		Args:
				image (ImageData): 画像データ
		"""
		# 土砂マスクを用いて土砂領域以外を除去
		image.dsm_after  = self.__masking(image, image.dsm_after,  image.mask)
		image.dem_before = self.__masking(image, image.dem_before, image.mask)
		image.dem        = self.__masking(image, image.dem,        image.mask)
		image.aspect     = self.__masking(image, image.aspect,     image.mask)

		return


	@staticmethod
	def __masking(image: ImageData, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
		""" マスク処理にて不要領域を除去

		Args:
				image (ImageData): 画像データ
				img (np.ndarray): マスク対象の画像
				mask (np.ndarray): マスク画像

		Returns:
				np.ndarray: マスク適用後の画像
		"""
		# 画像のコピー
		masked_img = img.copy()

		# マスク画像の次元を対象画像と同じにする
		if (len(masked_img.shape) == 3):
			mask = cv2.merge((mask, mask, mask))

		# マスク領域以外を除去
		idx = np.where(mask == 0)
		try:
			masked_img[idx] = np.nan
		except:
			masked_img[idx] = 0

		# # 土砂領域を抽出
		# idx = np.where(mask == 0).astype(np.float32)
		# # 土砂領域以外の標高データを消去
		# sed[idx] = np.nan
		# tif.save_tif(dsm, "dsm_after.tif", "sediment.tif")
		# return sed.astype(np.uint8)

		return masked_img


