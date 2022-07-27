import cv2
import numpy as np

from modules import tif


class ImageData():

	def __init__(self, path_list):
		"""
		初期化メソッド

		path_list: 入力パス
		"""
		# 入力パス
		self.path_list = path_list

		# 画像
		self.dsm_uav  = tif.load_tif(path_list[0]).astype(np.float32)
		self.dsm_heli = tif.load_tif(path_list[1]).astype(np.float32)
		self.dem      = tif.load_tif(path_list[2]).astype(np.float32)
		self.degree   = tif.load_tif(path_list[3]).astype(np.float32)
		self.mask     = cv2.imread(path_list[4], cv2.IMREAD_GRAYSCALE)
		self.ortho    = cv2.imread(path_list[5])
		self.gradient = None
		self.div_img  = None
		self.dsm_sub  = None

		# 画像サイズ
		self.size_3d   = self.dsm_uav.shape
		self.size_2d   = (self.size_3d[1], self.size_3d[0])
		self.s_size_2d = (int(self.size_3d[1] / 2), int(self.size_3d[0] / 2))

