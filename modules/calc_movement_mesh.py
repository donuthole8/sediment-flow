import cv2
import math
import numpy as np
from tqdm import trange

from modules import tool
from modules.image_data import ImageData


class CalcMovementMesh():
	# 傾斜方位への座標(Δy, Δx)
	DIRECTION: list[int] = [
		(-1, 0),		# 北
		(-1, 1),		# 北東
		(0,  1),		# 東
		(1,  1),		# 南東
		(1,  0),		# 南
		(1,  -1),		# 南西
		(0,  -1),		# 西
		(-1, -1),		# 北西
	]


	def __init__(self, mesh_size: int, size: tuple[int, int]) -> None:
		""" コンストラクタ

		Args:
				mesh_size (int): メッシュサイズ
				size (tuple[int, int]): 画像サイズ
		"""
		# メッシュサイズ
		self.mesh_size = mesh_size

		# メッシュサイズよりメッシュの高さと幅を取得
		self.mesh_height = (size[0] // self.mesh_size) + 1
		self.mesh_width  = (size[1] // self.mesh_size) + 1

		# 精度評価用の土砂移動推定結果データ
		self.calc_movement_result = []


		self.c = 1	


		return


	def main(self, image: ImageData) -> None:
		"""	メッシュベースでの土砂移動の推定

		Args:
				image (ImageData): 画像データ
		"""
		print("deg")
		print(image.degree)


		# メッシュの格子線を描画
		tool.draw_mesh(self, image)

		# メッシュ中心が土砂の場合各注目領域に対して処理を行う
		# for y in trange(self.mesh_height):
		for y in range(self.mesh_height):
			for x in range(self.mesh_width):
				# 注目メッシュの成分を保存
				self.y, self.x = y, x

				# 注目メッシュの中心座標を取得
				self.center_coord = self.get_center_coord()

				# 注目メッシュの座標群を取得
				self.mesh_coords = self.get_mesh_coords(image.size_3d, y, x)

				# 注目メッシュの中心座標が土砂領域か判別
				# TODO: 中心座標で判別するのではなくメッシュ内の土砂領域画素数（中心座標から数十ピクセル）で判別
				if (self.is_sedimentation_mask(image)):
				# if (self.is_sedimentation(image, mesh_coords)):	# 精度悪い
					# 点を描画
					cv2.circle(
						img=image.ortho,	# 画像
						center=(self.center_coord[1], self.center_coord[0]),	# 中心
						radius=3,	# 半径
						color=(0, 0, 255),	# 色
						thickness=self.mesh_size // 20,	# 太さ
					)

					# 傾斜方位のと隣接2方向の3方向に対しての隣接領域を取得
					labels = self.extract_neighbor(image)
					print("yx:", (y, x), "labels:", labels)
					print("-----------------------------")

					# # 傾斜方位が上から下の領域を抽出
					# labels = self.extract_downstream(image, labels)

					# # 侵食と堆積の組み合わせの領域を抽出
					# coords = self.extract_sediment(self, region, labels)
				else:
					self.calc_movement_result.append({"direction": np.nan, "center": np.nan})

		# メッシュ画像を保存
		cv2.imwrite("./outputs/mesh.png", image.ortho)
		# 隣接領域抽出での土砂流れ方向検知結果
		cv2.imwrite("mesh_line.png", image.ortho)

		return self.calc_movement_result


	def get_center_coord(self) -> tuple[int, int]:
		""" 注目メッシュの中心座標を取得

		Returns:
				tuple[int, int]: 注目メッシュの中心座標
		"""
		return (
			(self.mesh_size // 2) + self.y * self.mesh_size, 
			(self.mesh_size // 2) + self.x * self.mesh_size
		)


	def get_mesh_coords(
			self, 
			size: tuple[int, int, int], 
			y: int, 
			x: int
		) -> list[tuple]:
		""" 注目メッシュの座標群を取得

		Args:
				size (tuple[int, int, int]): 画像サイズ
				y (int): 注目メッシュのy成分
				x (int): 注目メッシュのx成分

		Returns:
				list[tuple]: 注目メッシュの座標群
		"""
		coords = []
		for _y in range(self.mesh_size):
			for _x in range(self.mesh_size):
				# 座標を取得
				coord = (
					(y * self.mesh_size) + _y, 
					(x * self.mesh_size) + _x 
				)
				# 座標が画像内に収まっているかを判定
				if (tool.is_index(size, coord)):
					coords.append(coord)

		return coords


	def is_sedimentation(self, image: ImageData, coords: list[tuple]) -> bool:
		""" 土砂領域かどうかを判別

		Args:
				image (ImageData): 画像データ
				coords (list[tuple]): 該当領域の座標群

		Returns:
				bool: 土砂領域フラグ
		"""
		# カウンタ
		is_sedimentation = 0
		is_not_sedimentation = 0

		# 全画素について調べる
		for coord in coords:
			if (self.is_sedimentation_mask(image, coord)):
				is_sedimentation     += 1
			else:
				is_not_sedimentation += 1

		# 画素数の多い方を返却
		return True if (is_sedimentation > is_not_sedimentation) else False


	def is_sedimentation_mask(self, image: ImageData) -> bool:
		""" 土砂マスク画像を用いて土砂領域かどうかを判別

		Args:
				image (ImageData): 画像データ

		Returns:
				bool: 土砂領域フラグ
		"""
		# マスク画像より土砂の判別
		try:
			if (image.mask[self.center_coord][0] == 0):
				return True
			else:
				return False
		except:
			return False


	def extract_neighbor_tmp(self, image: ImageData) -> list[int]:
		""" 8方向で隣接している領域の組かつ傾斜方位に沿った3領域を全て抽出

		Args:
				image (ImageData): 画像データ

		Returns:
				list[int]: 隣接領域のラベル
		"""
		# 注目メッシュの土砂領域マスク内の平均傾斜方位を取得
		average_direction = self.get_average_direction()
		# # 中心座標の傾斜方位
		# average_direction = image.degree[center]

		# # 精度評価用のデータを保存
		# self.calc_movement_result.append({"direction": average_direction, "center": center})

		# 傾斜方位の角度データを三角関数用の表記に変更
		if (average_direction >= 90):
			average_direction = average_direction - 90
		else:
			average_direction = 270 + average_direction

		# FIXME: 違うかも
		# https://qiita.com/FumioNonaka/items/c146420c3aeab27fc736
		try:
			# 傾斜方位の座標取得
			y_coord = int((self.mesh_size // 2) * math.sin(math.radians(average_direction))) + self.center_coord[0]
			x_coord = int((self.mesh_size // 2) * math.cos(math.radians(average_direction))) + self.center_coord[1]

			# 矢印を描画
			cv2.arrowedLine(
				img=image.ortho,	# 画像
				pt1=(self.center_coord[1], self.center_coord[0]), # 始点
				pt2=(x_coord, y_coord),	# 終点
				color=(0, 0, 255),	# 色			
				thickness=2,	# 太さ
			)
		except Exception as e:
			print(e, ", average_direction: ", average_direction)


	def extract_neighbor(
			self, 
			image: ImageData
		) -> list[int]:
		""" 8方向で隣接している領域の組かつ傾斜方位に沿った3領域を全て抽出

		Args:
				image (ImageData): 画像データ

		Returns:
				list[tuple]: 隣接領域データ
		"""
		# 注目メッシュの土砂領域マスク内の平均傾斜方位を取得
		average_direction = self.get_average_direction(image)
		print("ave", average_direction)

		try:
			# 傾斜方位データを取得
			directions = self.get_directions(average_direction)

			# 傾斜方位と隣接2領域のメッシュラベルを取得
			neighbor_mesh_labels = [
				(self.y + directions[0][0], self.x + directions[0][1]), 
				(self.y + directions[1][0], self.x + directions[1][1]), 
				(self.y + directions[2][0], self.x + directions[2][1]), 
			]

			# 平均傾斜方位を描画
			tool.draw_direction(self, image, average_direction)

		except:
			# average_directionがnp.nanの場合
			neighbor_mesh_labels = []

		# 角度データを保存
		# TODO: 他の土砂移動を実装したら消去
		self.calc_movement_result.append({
			"direction": average_direction, 
			"center": self.center_coord
		})

		return neighbor_mesh_labels


	def get_average_direction(self, image: ImageData) -> float:
		""" 注目メッシュの土砂領域マスク内の平均傾斜方位を取得

		Args:
				image (ImageData): 画像データ

		Returns:
				float: 平均傾斜方位データ
		"""
		# TODO: ここを中山さんの手法に修正(flow.py)
		# NOTE: 画素単位で3方向に土砂追跡するか,領域単位でとりあえず3領域取得するか
		# NOTE: 加重平均等にした方が良いのか

		sediment_pix_num = 0
		mask = cv2.split(image.mask)[0]
		average_direction = 0.0
		for coord in self.mesh_coords:
			# 土砂マスクの領域のみ
			if (mask[coord] == 0):
				average_direction += image.degree[coord]
				sediment_pix_num  += 1

		return average_direction / sediment_pix_num


	def get_directions(self, degree: float) -> tuple[tuple]:
		""" 傾斜方位を画素インデックスに変換し傾斜方位と隣接2方向を取得

		Args:
				degree (float): 角度

		Returns:
				tuple[tuple]: 傾斜方位と隣接2方向
		"""
		# FIXME: 傾斜方位が限定的になっている
		# 注目画素からの移動画素
		if   (math.isnan(degree)):
			# NOTE: 返り値違うかも
			return np.nan, np.nan
		elif (degree > 337.5) or  (degree <= 22.5):
			return self.DIRECTION[7], self.DIRECTION[0], self.DIRECTION[1]
		elif (degree > 22.5)  and (degree <= 67.5):
			return self.DIRECTION[0], self.DIRECTION[1], self.DIRECTION[2]
		elif (degree > 67.5)  and (degree <= 112.5):
			return self.DIRECTION[1], self.DIRECTION[2], self.DIRECTION[3]
		elif (degree > 112.5) and (degree <= 157.5):
			return self.DIRECTION[2], self.DIRECTION[3], self.DIRECTION[4]
		elif (degree > 157.5) and (degree <= 202.5):
			return self.DIRECTION[3], self.DIRECTION[4], self.DIRECTION[5]
		elif (degree > 202.5) and (degree <= 247.5):
			return self.DIRECTION[4], self.DIRECTION[5], self.DIRECTION[6]
		elif (degree > 247.5) and (degree <= 292.5):
			return self.DIRECTION[5], self.DIRECTION[6], self.DIRECTION[7]
		elif (degree > 292.5) and (degree <= 337.5):
			return self.DIRECTION[6], self.DIRECTION[7], self.DIRECTION[0]


	@staticmethod
	def get_heights(
			image: ImageData, 
			directions: tuple[tuple],
			coord: tuple[int, int]
		) -> tuple[int, int, int]:
		""" 注目座標からの標高値を取得

		Args:
				image (ImageData): 画像データ
				directions (tuple[tuple]): 注目座標からの傾斜方位
				coord (tuple[int, int]): 注目座標

		Returns:
				tuple[int, int, int]: 注目座標からの標高値
		"""
		heights = []

		for direction in directions:
			# 注目画素からの傾斜方位の標高値を取得
			heights.append(image.dsm_uav[
				coord[0] + direction[0], 
				coord[1] + direction[1]
			])

		return heights


	