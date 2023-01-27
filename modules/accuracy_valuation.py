import numpy as np


class AccuracyValuation():
	# (w, h) = 12 x 13
	# direction: 方向 0<= dir <360
	# distance: 水平移動距離 0<= dis <=70.7 (50 ** 2) ,,,
	# 　※7月11日と被災前の比較（被災前写真：平成21年4月撮影）で作成

	## 違うかも
	# 小屋浦切り抜き画像正解データ
	# 約 400m x 400m
	# 5mメッシュ -> 80 x 80
	CORRECT_DATA_TRIM_100: list[dict[int, float]] = [
		# 1 - 3 rows
		[
		{"direction": 170, "distance": 5}, {"direction": 175, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction":220, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
		], [
		{"direction": 160, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": 220, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], [
		{"direction": 160, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": 200, "distance": 5},
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], 

		# 4 - 6 rows
		[
		{"direction": 135, "distance": 5}, {"direction": 175, "distance": 5}, {"direction": 190, "distance": 5},
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": 190, "distance": 5},
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], [
		{"direction": np.nan, "distance": np.nan}, {"direction": 170, "distance": 5}, {"direction": 200, "distance": 5},
		{"direction": 225, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": 190, "distance": 5}, {"direction": 200, "distance": 5}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan},
		], [
		{"direction": np.nan, "distance": np.nan}, {"direction": 225, "distance": 5}, {"direction": 225, "distance": 5},
		{"direction": 230, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": 225, "distance": 5}, {"direction": 220, "distance": 5}, {"direction": 250, "distance": 5}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan},
		],

		# 7 - 9 rows
		[
		{"direction": 250, "distance": 5}, {"direction": 250, "distance": 5}, {"direction": 250, "distance": 5},
		{"direction": 245, "distance": 5}, {"direction": 245, "distance": 5}, {"direction": 225, "distance": 5},
		{"direction": 225, "distance": 5}, {"direction": 225, "distance": 5}, {"direction": 225, "distance": 5},
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan},
		], [
		{"direction": 250, "distance": 5}, {"direction": 250, "distance": 5}, {"direction": 260, "distance": 5},
		{"direction": 260, "distance": 5}, {"direction": 260, "distance": 5}, {"direction": 260, "distance": 5},
		{"direction": 260, "distance": 5}, {"direction": 250, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], [
		{"direction": 265, "distance": 5}, {"direction": 275, "distance": 5}, {"direction": 270, "distance": 5},
		{"direction": 275, "distance": 5}, {"direction": 270, "distance": 5}, {"direction": 270, "distance": 5},
		{"direction": 250, "distance": 5}, {"direction": 225, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], 

		# 10 - 13 row
		[
		{"direction": 290, "distance": 5}, {"direction": 290, "distance": 5}, {"direction": 275, "distance": 5},
		{"direction": 280, "distance": 5}, {"direction": 275, "distance": 5}, {"direction": 290, "distance": 5},
		{"direction": 275, "distance": 5}, {"direction": 275, "distance": 5}, {"direction": 275, "distance": 5},
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], [
		{"direction": 290, "distance": 5}, {"direction": 290, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": 325, "distance": 5}, {"direction": 320, "distance": 5}, {"direction": 325, "distance": 5},
		{"direction": 295, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], [
		{"direction": np.nan, "distance": np.nan}, {"direction": 325, "distance": 5}, {"direction": 325, "distance": 5},
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], [
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		{"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
		], 
	]


	def __init__(self, calc_movement_result: list[float, tuple]) -> None:
		# 精度評価用の土砂移動推定結果データ
		self.calc_movement_result = calc_movement_result

		# 正解データ
		self.answer = self.CORRECT_DATA_TRIM_100

		return


	def main(self) -> None:
		""" メッシュベースでの精度評価
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

					# # FIXME: 負値がある
					# print("- {} ({}, {}) {}: {}".format(
					# 	mesh_num, 
					# 	i, j,
					# 	self.calc_movement_result[i * len(answer) + j]["center"], 1 - error
					# ))

					# Latex用
					print("{} ({},{}) & {} & {} & {:} \\\\".format(
						mesh_num, 
						i, j, 
						int(result_direction),
						answer_direction,
						round(1 - error, 3))
					)

					# TODO: 正解画像の描画関数を作成


		# 各メッシュの精度平均
		print("- 各メッシュの精度平均:", np.nanmean(accuracies))

		return
