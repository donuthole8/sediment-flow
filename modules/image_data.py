import cv2
import numpy as np

"""
各種画像データや画像サイズを扱う
"""
class ImageData():
	def __init__(self, path_list: list[str]) -> None:
		"""
		初期化メソッド

		path_list: 入力パス
		"""
		# 入力パス
		self.path_list = path_list

		# tif画像
		self.dsm_uav     = cv2.imread(path_list[0], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.dsm_heli    = cv2.imread(path_list[1], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.dem         = cv2.imread(path_list[2], cv2.IMREAD_ANYDEPTH).astype(np.float32)
		self.degree      = cv2.imread(path_list[3], cv2.IMREAD_ANYDEPTH).astype(np.float32)

		# 画像
		self.mask          = cv2.imread(path_list[4], cv2.IMREAD_GRAYSCALE)
		self.ortho         = cv2.imread(path_list[5]).astype(np.float32)
		self.masked_ortho  = None
		self.gradient      = None
		self.div_img       = None
		self.label_table   = None
		self.dissimilarity = None
		self.edge          = None
		self.bld_mask      = None
		self.normed_dsm    = None
		self.dsm_sub       = None

		# 画像サイズ
		self.size_3d   = self.dsm_uav.shape
		# FIXME: ２ｄのみ(x,y)なので(y,x)に修正
		self.size_2d   = (self.size_3d[1], self.size_3d[0])
		self.s_size_2d = (int(self.size_3d[1] / 2), int(self.size_3d[0] / 2))

		# 領域データ
		self.pms_coords = []
		self.pms_pix    = []
		self.region     = []
		self.building   = []
