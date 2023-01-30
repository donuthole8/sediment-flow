import numpy as np

from modules.utils import drawing_util


class AccuracyValuation():
	# (w, h) = 12 x 13
	# direction: 方向 0<= dir <360
	# distance: 水平移動距離 0<= dis <=70.7 (50 ** 2) ,,,
	# 　※7月11日と被災前の比較（被災前写真：平成21年4月撮影）で作成

	## 違うかも
	# 小屋浦切り抜き画像正解データ
	# 約 400m x 400m
	# 5mメッシュ -> 80 x 80
	CORRECT_DATA_KOYAURA: list[dict[int, float]] = [
		# 1 - 3 rows
		[
		{"direction": 170}, {"direction": 175}, {"direction": 185}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": 210}, {"direction":230}, {"direction": np.nan}, 
		], [
		{"direction": 160}, {"direction": 170}, {"direction": 230}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": 200}, {"direction": np.nan}, {"direction": np.nan}, 
		], [
		{"direction": 160}, {"direction": 170}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": 200},
		{"direction": 195}, {"direction": np.nan}, {"direction": np.nan}, 
		], 

		# 4 - 6 rows
		[
		{"direction": 135}, {"direction": 175}, {"direction": 190},
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": 190},
		{"direction": 200}, {"direction": np.nan}, {"direction": np.nan}, 
		], [
		{"direction": np.nan}, {"direction": 170}, {"direction": 200},
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": 190}, {"direction": 200}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan},
		], [
		{"direction": np.nan}, {"direction": 220}, {"direction": 225},
		{"direction": 230}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": 215}, {"direction": 200}, {"direction": 230}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan},
		],

		# 7 - 9 rows
		[
		{"direction": 200}, {"direction": 220}, {"direction": 240},
		{"direction": 245}, {"direction": 245}, {"direction": 220},
		{"direction": 220}, {"direction": 185}, {"direction": 240},
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan},
		], [
		{"direction": 195}, {"direction": 215}, {"direction": 235},
		{"direction": 250}, {"direction": 250}, {"direction": 250},
		{"direction": 210}, {"direction": 205}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		], [
		{"direction": 245}, {"direction": 255}, {"direction": 250},
		{"direction": 250}, {"direction": 255}, {"direction": 230},
		{"direction": 235}, {"direction": 230}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		], 

		# 10 - 13 row
		[
		{"direction": 290}, {"direction": 290}, {"direction": 275},
		{"direction": 280}, {"direction": 275}, {"direction": 290},
		{"direction": 280}, {"direction": 290}, {"direction": 275},
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		], [
		{"direction": 290}, {"direction": 285}, {"direction": 280}, 
		{"direction": 285}, {"direction": 280}, {"direction":280},
		{"direction": 295}, {"direction": 310}, {"direction": 305}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		], [
		{"direction": np.nan}, {"direction": 325}, {"direction": 320},
		{"direction": 320}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": 310}, {"direction": np.nan}, {"direction": np.nan}, 
		], [
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		{"direction": np.nan}, {"direction": np.nan}, {"direction": np.nan}, 
		], 
	]


	def __init__(self, calc_movement_mesh: list[float, tuple], experiment: str) -> None:
		# 精度評価用の土砂移動推定結果データ
		self.calc_movement_mesh = calc_movement_mesh
		self.calc_movement_result = calc_movement_mesh.calc_movement_result

		# 正解データ
		if   (experiment == "koyaura"):
			self.answer = self.CORRECT_DATA_KOYAURA
		elif (experiment == "mihara"):
			self.answer = self.CORRECT_DATA_KOYAURA
		elif (experiment == "atami"):
			self.answer = self.CORRECT_DATA_KOYAURA

		return


	def main(self, image) -> None:
		""" メッシュベースでの精度評価
		"""
		# 正解画像を作成
		drawing_util.write_answer(image, self.calc_movement_mesh, self.answer)

		# 精度評価を実施
		self.__accuracy_valuation()

		return


	def __accuracy_valuation(self):
		""" 精度評価
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

					try:
						# Latex用
						print("{} ({},{}) & {} & {} & {:} \\\\".format(
							mesh_num, 
							i, j, 
							int(result_direction),
							answer_direction,
							round(1 - error, 3))
						)
					except:
						# Latex用
						print("{} ({},{}) & {} & {} & {:} \\\\".format(
							mesh_num, 
							i, j, 
							result_direction,
							answer_direction,
							round(1 - error, 3))
						)


					# TODO: 正解画像の描画関数を作成


		# 各メッシュの精度平均
		print("- 各メッシュの精度平均:", np.nanmean(accuracies))

		return
