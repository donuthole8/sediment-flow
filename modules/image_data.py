import cv2
import numpy as np

from modules.utils import image_util


"""
各種画像データや画像サイズを扱う
"""
class ImageData():
	def __init__(self, experiment: str, path_list: list[str]) -> None:
		""" コンストラクタ

		Args:
				experiment: 実験地域
				path_list (list[str]): 入力パス
		"""
		# 実験対象
		self.experiment = experiment

		# 入力パス
		self.path_list = path_list

		# tif画像
		self.dsm_after  = cv2.imread(path_list[0], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.dem_before = cv2.imread(path_list[1], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.dem        = cv2.imread(path_list[2], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.degree     = cv2.imread(path_list[3], cv2.IMREAD_ANYDEPTH).astype(np.float32)

		# 画像
		self.mask          = cv2.split(
			cv2.imread(path_list[4], cv2.IMREAD_GRAYSCALE)
		)[0]
		self.ortho         = cv2.imread(path_list[5]).astype(np.float32)
		self.masked_ortho  = None
		self.slope         = None
		self.div_img       = None
		self.label_table   = None
		self.dissimilarity = None
		self.edge          = None
		self.bld_mask      = None
		self.normed_dsm    = None
		self.dsm_sub       = None
		self.bld_gsi       = cv2.imread(path_list[6], cv2.IMREAD_GRAYSCALE)

		# 2値化
		# NOTE: いらないかも？
		self.bld_gsi[np.where(self.bld_gsi < 128)] = 0

		# 画像サイズ（shape=(y,x,z)）
		self.size_3d    = self.dsm_after.shape
		self.size_2d    = (self.size_3d[0], self.size_3d[1])
		self.size_2d_xy = (self.size_3d[1], self.size_3d[0])
		self.s_size_2d  = (int(self.size_3d[1] / 2), int(self.size_3d[0] / 2))

		# 領域データ
		self.pms_coords = []
		self.pms_pix    = []
		self.region     = []
		self.building   = []

		cv2.imwrite("./outputs/" + self.experiment + "/ortho.png", self.ortho)
		
		return
