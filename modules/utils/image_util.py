import cv2
import numpy as np

# from modules.image_data import ImageData
from modules.utils import calculation_util


def show_image_size(image) -> None:
	"""	画像サイズの確認

	Args:
			image (imageData): 画像データ
	"""
	print("# 入力画像のサイズ確認")
	
	print("- uav-size  :", image.dsm_after.shape)
	print("- heli-size :", image.dem_before.shape)
	print("- dem-size  :", image.dem.shape)
	print("- deg-size  :", image.degree.shape)
	print("- mask-size :", image.mask.shape)
	print("- img-size  :", image.ortho.shape)
	print("- bld-size  :", image.bld_gsi.shape)

	return


def show_max_min(image) -> None:
	""" 画像の最大最小値の表示

	Args:
			image (ImageData): 画像データ
	"""
	print("# 入力画像の最大最小値確認")

	print("- after (min, max):", calculation_util.calc_min_max(image.dsm_after))
	print("- before(min, max):", calculation_util.calc_min_max(image.dem_before))
	print("- dem   (min, max):", calculation_util.calc_min_max(image.dem))
	print("- deg   (min, max):", calculation_util.calc_min_max(image.degree))

	return
