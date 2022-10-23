import cv2
import math
import numpy as np

from modules import operation
from modules import constant


class AccuracyValuation():
	def __init__(self, calc_movement_result: list[float, tuple]) -> None:
		# 精度評価用の土砂移動推定結果データ
		self.calc_movement_result = calc_movement_result

		# 正解データ
		self.answer = constant.CORRECT_DATA_TRIM_100

		return


	def main(self) -> None:
		"""
		メッシュベースでの精度評価
		"""
		accuracies = []

		for i, answer in enumerate(self.answer):
			# print("answer:::::::", i, answer)
			# print("result:::::::", self.calc_movement_result[i])
			# print()

			# 結果データの角度
			result_direction = self.calc_movement_result[i]["direction"]

			# FIXME: 100サイズメッシュの場合限定
			for j in range(12):
				# 正解データの角度
				answer_direction = answer[j]["direction"]

				# 正解データと結果データの絶対値を360°で除算（誤差の割合）
				error = abs(result_direction - answer_direction) / 360
				print("res, ans, err", i, j, result_direction, answer_direction, error)

				# 精度を算出
				accuracies.append(1 - error)

				print("精度 メッシュ中心座標(y,x):", 1 - error, self.calc_movement_result[i]["center"])
				print()

		# 各メッシュの精度平均
		print("各メッシュの精度平均:", np.nanmean(accuracies))
