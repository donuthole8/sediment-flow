import cv2
import numpy as np
import matplotlib as mpl

from modules import tool
from modules.image_data import ImageData


class AnalyzeImage():
	@tool.stop_watch
	def texture_analysis(self, image: ImageData) -> None:
		""" テクスチャ解析

		Args:
				image (ImageData): 画像データ
		"""
		# テクスチャ解析
		# TODO: オルソ画像でなく領域分割済み画像等でやっても良いかも
		image.dissimilarity = self.__texture_analysis(image)

		return


	@staticmethod
	def __texture_analysis(image: ImageData) -> np.ndarray:
		""" テクスチャ解析

		Args:
				image (ImageData): 画像データ

		Returns:
				np.ndarray: テクスチャ画像
		"""
		mpl.rc('image', cmap='jet')
		kernel_size = 5
		levels = 8
		symmetric = False
		normed = True
		dst = cv2.cvtColor(image.ortho, cv2.COLOR_BGR2GRAY)

		# binarize
		dst_bin = dst // (256 // levels) # [0:255] -> [0:7]

		# calc_glcm
		h,w = dst.shape
		glcm = np.zeros((h,w,levels,levels), dtype=np.uint8)
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		dst_bin_r = np.append(dst_bin[:,1:], dst_bin[:,-1:], axis=1)
		for i in range(levels):
			for j in range(levels):
				mask = (dst_bin==i) & (dst_bin_r==j)
				mask = mask.astype(np.uint8)
				glcm[:,:,i,j] = cv2.filter2D(mask, -1, kernel)
		glcm = glcm.astype(np.float32)
		if symmetric:
			glcm += glcm[:,:,::-1, ::-1]
		if normed:
			glcm = glcm/glcm.sum(axis=(2,3), keepdims=True)
		# martrix axis
		axis = np.arange(levels, dtype=np.float32)+1
		w = axis.reshape(1,1,-1,1)
		x = np.repeat(axis.reshape(1,-1), levels, axis=0)
		y = np.repeat(axis.reshape(-1,1), levels, axis=1)

		# GLCM contrast
		glcm_contrast = np.sum(glcm*(x-y)**2, axis=(2,3))
		# GLCM dissimilarity（不均一性）
		glcm_dissimilarity = np.sum(glcm*np.abs(x-y), axis=(2,3))
		# GLCM homogeneity（均一性）
		glcm_homogeneity = np.sum(glcm/(1.0+(x-y)**2), axis=(2,3))
		# GLCM energy & ASM
		glcm_asm = np.sum(glcm**2, axis=(2,3))
		# GLCM entropy（情報量）
		ks = 5 # kernel_size
		pnorm = glcm / np.sum(glcm, axis=(2,3), keepdims=True) + 1./ks**2
		glcm_entropy = np.sum(-pnorm * np.log(pnorm), axis=(2,3))
		# GLCM mean
		glcm_mean = np.mean(glcm*w, axis=(2,3))
		# GLCM std
		glcm_std = np.std(glcm*w, axis=(2,3))
		# GLCM energy
		glcm_energy = np.sqrt(glcm_asm)
		# GLCM max
		glcm_max = np.max(glcm, axis=(2,3))
		
		# plot
		# plt.figure(figsize=(10,4.5))

		outs = [dst, glcm_mean, glcm_std,
			glcm_contrast, glcm_dissimilarity, glcm_homogeneity,
			glcm_asm, glcm_energy, glcm_max,
			glcm_entropy]
		titles = ['original','mean','std','contrast','dissimilarity','homogeneity','ASM','energy','max','entropy']
		for i in range(10):
			cv2.imwrite("./outputs/texture/" + titles[i] + '.tif', outs[i])

		# GLCM dissimilarity（不均一性）
		# - [0.0 - 3.8399997]
		return outs[4]


	def edge_analysis(self, image: ImageData) -> None:
		""" エッジ抽出

		Args:
				image (ImageData): 画像データ
		"""
		# エッジ抽出
		self.edge = self.__edge_analysis(image, 100, 200)

		return


	@staticmethod
	def __edge_analysis(
			image: ImageData, 
			threshold1: int, 
			threshold2: int
		) -> np.ndarray:

		""" エッジ抽出

		Args:
				image (ImageData): 画像データ
				threshold1 (int): キャニー法の閾値1
				threshold2 (int): キャニー法の閾値2

		Returns:
				np.ndarray: エッジ画像
		"""
		# グレースケール化
		img_gray = cv2.cvtColor(image.ortho, cv2.COLOR_BGR2GRAY).astype(np.uint8)

		# エッジ抽出
		img_canny = cv2.Canny(img_gray, threshold1, threshold2)

		# 画像保存
		cv2.imwrite("./outputs/edge.png", img_canny)

		return img_canny