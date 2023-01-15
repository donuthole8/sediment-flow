import numpy as np


def calc_min_max(dsm: np.ndarray) -> tuple[float, float]:
	""" 標高値モデルから最小値と最大値を算出

	Args:
			dsm (np.ndarray): DSM（或いはDEM）

	Returns:
			tuple[float, float]: 最小値・最大値
	"""
	# 最大値と最小値を算出
	_min, _max = np.nanmin(dsm), np.nanmax(dsm)

	return _min, _max


def calc_mean_sd(dsm: np.ndarray) -> tuple[float, float]:
	""" 標高値モデルから平均と標準偏差を算出

	Args:
			dsm (np.ndarray): DSM（或いはDEM）

	Returns:
			tuple[float, float]: 平均・標準偏差
	"""
	# 平均と標準偏差を算出
	mean, sd = np.nanmean(dsm), np.nanstd(dsm)

	return mean, sd

