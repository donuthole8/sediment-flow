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
		mesh_num = 0
		accuracies = []
		for i, answer in enumerate(self.answer):
			# FIXME: 100サイズメッシュの場合限定
			for j in range(len(answer)):
				# 結果データの角度
				result_direction = self.calc_movement_result[i * len(answer) + j]["direction"]

				# 正解データの角度
				answer_direction = answer[j]["direction"]

				if (answer_direction is not np.nan) and (result_direction is not np.nan):
					# 正解データと結果データの絶対値を360°で除算（誤差の割合）
					error = abs(result_direction - answer_direction) / 180

					# 精度を算出
					accuracies.append(1 - error)

					# メッシュ番号
					mesh_num += 1

					print(" -", mesh_num, self.calc_movement_result[i * len(answer) + j]["center"], ":", 1 - error)

		# 各メッシュの精度平均
		print("各メッシュの精度平均:", np.nanmean(accuracies))

