import cv2
import numpy as np

from modules.image_data import ImageData



def save_resize_image(path: str, image: np.ndarray, size: tuple) -> None:
	"""	画像を縮小して保存

	Args:
			path (str): 保存先のパス
			image (np.ndarray): 画像データ
			size (tuple):  保存サイズ
	"""
	# 画像をリサイズ
	resize_img = cv2.resize(
		image, 
		size, 
		interpolation=cv2.INTER_CUBIC
	)

	# 画像を保存
	cv2.imwrite("./outputs/" + path, resize_img)

	return


def show_image_size(image: ImageData) -> None:
	"""	画像サイズの確認

	Args:
			image (imageData): 画像データ
	"""
	print("- uav-size  :", image.dsm_uav.shape)
	print("- heli-size :", image.dsm_heli.shape)
	print("- dem-size  :", image.dem.shape)
	print("- deg-size  :", image.degree.shape)
	print("- mask-size :", image.mask.shape)
	print("- img-size  :", image.ortho.shape)
	print("- bld-size  :", image.bld_gsi.shape)

	return

