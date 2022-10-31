import cv2
import numpy as np
import scipy.ndimage as ndimage

from modules import tool
from modules.image_data import ImageData



class MaskProcessing():
	@tool.stop_watch
	def norm_mask(
		self, 
		image: ImageData, 
		area_th: int, 
		scale: int
	) -> None:
		"""
		マスク画像の前処理

		area_th: 面積の閾値（スケール済みであるので(area_th/scale)が閾値となる）
		scale: 輪郭抽出の際の拡大倍率
		"""
		# 画像反転
		image.mask = cv2.bitwise_not(image.mask)

		# モルフォロジー処理
		# TODO: カーネルサイズと実行回数の調整
		# morpho_mask = morphology(mask, 15, 15)
		self.__morphology(image, 3, 3)

		# 輪郭抽出
		# contours, self.mask = self.get_norm_contours(morpho_mask, scale, 15)
		contours = self.__get_norm_contours(image, scale, 3)

		# 面積が閾値未満の領域を除去
		self.__remove_small_area(image, contours, area_th, scale)

		return


	@staticmethod
	def __morphology(
		image: ImageData, 
		ksize: int, 
		exe_num: int
	) -> None:
		"""
		モルフォロジー処理

		image: 画像データ
		ksize: カーネルサイズ
		exe_num: 実行回数
		"""
		# モルフォロジー処理によるノイズ除去
		kernel = np.ones((ksize, ksize), np.uint8)
		opening = cv2.morphologyEx(image.mask, cv2.MORPH_OPEN, kernel)
		for i in range(1, exe_num):
			opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		for i in range(1, exe_num):
			closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

		# クロージング処理による建物領域の穴埋め
		closing = cv2.dilate(closing, kernel, iterations = 1)
		closing = cv2.dilate(closing, kernel, iterations = 1)
		closing = cv2.erode (closing, kernel, iterations = 1)
		closing = cv2.erode (closing, kernel, iterations = 1)

		# 画像を保存
		tool.save_resize_image("opening.png", opening, image.s_size_2d)
		tool.save_resize_image("closing.png", closing, image.s_size_2d)

		# 結果を保存
		image.mask = closing

		return


	@staticmethod
	def __get_norm_contours(
		image: ImageData, 
		scale: int, 
		ksize: int
	) -> list[list]:
		"""
		輪郭をぼやけさせて抽出

		image: 画像データ
		scale: 拡大倍率
		ksize: カーネルサイズ
		"""
		# 画像の高さと幅を取得
		w, h = image.size_2d

		# 拡大することで輪郭をぼけさせ境界を識別しやすくする
		img_resize = cv2.resize(image.mask, (w * scale, h * scale))

		# ガウシアンによるぼかし処理
		img_blur = cv2.GaussianBlur(img_resize, (ksize,ksize), 0)

		# 二値化と大津処理
		_, dst = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		# モルフォロジー膨張処理
		kernel = np.ones((ksize,ksize), np.uint8)
		dst = cv2.dilate(dst, kernel, iterations = 1)

		# 画像欠けがあった場合に塗りつぶし
		image.mask = ndimage.binary_fill_holes(dst).astype(int) * 255

		# 輪郭抽出
		contours, _ = cv2.findContours(
			image.mask.astype(np.uint8), 
			cv2.RETR_EXTERNAL, 
			cv2.CHAIN_APPROX_SIMPLE
		)

		return contours


	@staticmethod
	def __remove_small_area(
		image: ImageData, 
		contours: list[list], 
		area_th: int, 
		scale: int
	) -> None:
		"""
		面積が閾値以下の領域を除去

		image: 画像データ
		contours: 輪郭データ
		area_th: 面積の閾値
		scale: 拡大倍率
		"""
		# 画像の高さと幅を取得
		h, w = image.mask.shape

		# 輪郭データをフィルタリング
		contours = list(filter(lambda x: cv2.contourArea(x) >= area_th * scale, contours))

		# 黒画像
		campus = np.zeros((h, w))

		# TODO: スケール分は考慮して割る必要あり
		for i, contour in enumerate(contours):
			# 面積
			area = int(cv2.contourArea(contour) / scale)

			# 閾値以上の面積の場合画像に出力
			if (area >= area_th):
				normed_mask = cv2.drawContours(campus, contours, i, 255, -1)

		# スケールを戻す
		image.mask = cv2.resize(normed_mask, (int(w/scale), int(h/scale)))

		# 画像の保存
		cv2.imwrite("./outputs/normed_mask.png", image.mask)

		return

	
	def apply_mask(self, image: ImageData) -> None:
		"""
		マスク画像を適用し土砂領域のみを抽出

		image: 画像データ
		"""
		# 土砂マスクを用いて土砂領域以外を除去
		image.dsm_uav  = self.__masking(image, image.dsm_uav,  image.mask)
		image.dsm_heli = self.__masking(image, image.dsm_heli, image.mask)
		image.degree   = self.__masking(image, image.degree,   image.mask)

		return


	@staticmethod
	def __masking(
		image: ImageData, 
		img: np.ndarray, 
		mask: np.ndarray
	) -> np.ndarray:
		"""
		マスク処理にて不要領域を除去

		img: マスク対象の画像
		mask: マスク画像
		"""
		# 画像のコピー
		masked_img = img.copy()

		# マスク画像の次元を対象画像と同じにする
		if (len(masked_img.shape) != 3):
			mask = cv2.split(mask)[0]

		# マスク領域以外を除去
		idx = np.where(mask != 0)
		try:
			masked_img[idx] = np.nan
		except:
			masked_img[idx] = 0

		# # 土砂領域を抽出
		# idx = np.where(mask == 0).astype(np.float32)
		# # 土砂領域以外の標高データを消去
		# sed[idx] = np.nan
		# tif.save_tif(dsm, "dsm_uav.tif", "sediment.tif")
		# return sed.astype(np.uint8)

		# 画像を保存
		tool.save_resize_image("masked_img.png", masked_img, image.s_size_2d)

		return masked_img


