import cv2
import math
import numpy as np

from modules import tool
from modules.image_data import ImageData


class CalcMovementMesh():
	# 傾斜方向への座標(Δy, Δx)
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

		return


	def main(self, image: ImageData, ) -> None:
		"""	メッシュベースでの土砂移動の推定

		Args:
				image (ImageData): 画像データ
		"""
		# メッシュの格子線を描画
		tool.draw_mesh(self, image)
		
		# メッシュ中心が土砂の場合各注目領域に対して処理を行う
		for y in range(self.mesh_height):
			for x in range(self.mesh_width):
				# print()
				# print("(y, x) ", y, x)

				# 注目メッシュの中心座標を取得
				center_coord = self.get_center_coord(y, x)

				# 注目メッシュの座標群を取得
				mesh_coords  = self.get_mesh_coords(image.size_3d, y, x)

				# 注目メッシュの中心座標が土砂領域か判別
				# TODO: 土砂マスクを利用しない場合座標群(mesh_coords)を利用
				# if not (self.is_sedimentation(image, center_coord)):

				# print(center_coord)


				if (self.is_sedimentation_mask(image, center_coord)):
					# 点を描画
					cv2.circle(
						img=image.ortho,                       # 画像
						center=(center_coord[1], center_coord[0]),  # 中心
						radius=3,                                   # 半径
						color=(0, 0, 255),                          # 色
						thickness=self.mesh_size // 20,                                # 太さ
					)

					# 傾斜方向のと隣接2方向の3方向に対しての隣接領域を取得
					coords = self.extract_neighbor(image, center_coord, mesh_coords)

					# # 傾斜方向が上から下の領域を抽出
					# coords = self.extract_downstream(self, region, labels)

					# # 侵食と堆積の組み合わせの領域を抽出
					# coords = self.extract_sediment(self, region, labels)
				else: 
					self.calc_movement_result.append({"direction": np.nan, "center": np.nan})

		# メッシュ画像を保存
		cv2.imwrite("./outputs/mesh.png", image.ortho)
		# 隣接領域抽出での土砂流れ方向検知結果
		cv2.imwrite("mesh_line.png", image.ortho)

		return self.calc_movement_result


	def is_sedimentation(self, image: ImageData, center: tuple[int, int]) -> bool:
		""" 土砂領域かどうかを判別

		Args:
				image (ImageData): 画像データ
				center (tuple[int, int]): 該当領域の中心座標

		Returns:
				bool: 土砂領域フラグ
		"""
		# Lab表色系に変換
		Lp, ap, bp = cv2.split(
			cv2.cvtColor(
				image.div_img.astype(np.uint8), 
				cv2.COLOR_BGR2Lab)
		)
		Lp, ap, bp = Lp * 255, ap * 255, bp * 255

		# 土砂の判別
		if (Lp[center] > 125) & (ap[center] > 130):
			return True
		else:
			return False


	def is_sedimentation_mask(self, image: ImageData, center: tuple[int, int]) -> bool:
		""" 土砂マスク画像を用いて土砂領域かどうかを判別

		Args:
				image (ImageData): 画像データ
				center (tuple[int, int]): 該当領域の中心座標

		Returns:
				bool: 土砂領域フラグ
		"""

		# マスク画像より土砂の判別
		try:
			if (image.mask[center][0] == 0):
				return True
			else:
				return False
		except:
			return False


	def get_center_coord(self, y: int, x: int) -> tuple[int, int]:
		""" 注目メッシュの中心座標を取得

		Args:
				y (int): 注目メッシュのy成分
				x (int): 注目メッシュのx成分

		Returns:
				tuple[int, int]: 注目メッシュの中心座標
		"""
		return (
			(self.mesh_size // 2) + y * self.mesh_size, 
			(self.mesh_size // 2) + x * self.mesh_size
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


	def extract_neighbor(
			self, 
			image: ImageData, 
			center: tuple[int, int], 
			coords: list[tuple]
		) -> list[int]:
		""" 8方向で隣接している領域の組かつ傾斜方向に沿った3領域を全て抽出

		Args:
				image (ImageData): 画像データ
				center (tuple[int, int]): 注目メッシュの中心座標
				coords (list[tuple]): 注目メッシュの座標群

		Returns:
				list[int]: 隣接領域のラベル
		"""
		# 中心の傾斜方向データを取得
		# TODO: ここを中山さんの手法に修正(flow.py)
		# NOTE: 画素単位で3方向に土砂追跡するか,領域単位でとりあえず3領域取得するか
		# NOTE: 3画素分岐するとしても最終的には1点からは１本ベクトルにしたい
		pass

		# 注目メッシュの平均傾斜方向を取得
		sediment_pix_num = 0
		mask = cv2.split(image.mask)[0]
		average_direction = 0.0
		for coord in coords:
			if (mask[coord] == 0):
				average_direction += image.degree[coord]
				sediment_pix_num += 1

		average_direction = average_direction / sediment_pix_num

		# # 中心座標の傾斜方向
		# average_direction = image.degree[center]

		# 精度評価用のデータを保存
		self.calc_movement_result.append({"direction": average_direction, "center": center})

		# 傾斜方向の角度データを三角関数用の表記に変更
		average_direction = 360 - average_direction

		# 傾斜方向の座標取得
		# FIXME: 違うかも
		# https://qiita.com/FumioNonaka/items/c146420c3aeab27fc736
		try:
			y_coord = int((self.mesh_size // 2) * math.sin(math.radians(average_direction))) + center[0]
			x_coord = int((self.mesh_size // 2) * math.cos(math.radians(average_direction))) + center[1]

			# 矢印を描画
			cv2.arrowedLine(
				img=image.ortho,     	# 画像
				pt1=(center[1], center[0]), # 始点
				pt2=(x_coord, y_coord),			# 終点
				color=(0, 0, 255),  				# 色			
				thickness=2,        				# 太さ
			)
		except Exception as e:
			print(e, "average_direction: ", average_direction)



		# # 土砂候補画素の座標
		# y_coord, x_coord = center
		# sediment_coords = []

		# # 作業用配列の座標
		# temp_coords = [(y_coord, x_coord)]

		# while (len(temp_coords)) > 0:
		# 	# 注目画素の座標を取得
		# 	# NOTE: ここの座標が違うかも
		# 	y_coord, x_coord = temp_coords[0]
		# 	print("target coords", (y_coord, x_coord))

		# 	# 一時配列から注目座標を削除
		# 	temp_coords.pop(0)

		# 	# 座標群に入っていなかったら処理を行わない
		# 	if ((y_coord, x_coord) in coords) and (not ((y_coord, x_coord) in sediment_coords)):

		# 		# 土砂座標を追加
		# 		sediment_coords.append((
		# 			y_coord, 
		# 			x_coord
		# 		))

		# 		# 注目画素の標高値を取得
		# 		target_height = image.dsm_uav[(y_coord, x_coord)]

		# 		# 注目画素の傾斜方向の画素と隣接2画素の傾斜方向を取得
		# 		directions = self.get_directions(image.degree[center])

		# 		# 注目画素の傾斜方向の画素と隣接2画素の標高値を取得
		# 		heights = self.get_heights(image, directions, center)

		# 		# 注目画素の標高値と傾斜方向の画素値を比較
		# 		for direction, height in zip(directions, heights):					
		# 			# 注目標高値より低ければ処理を継続・土砂座標にすでに追加済みでなければ
		# 			# if ((height <= target_height) and (not (y_coord, x_coord) in sediment_coords)):
		# 			if ((height <= target_height)):
						
		# 				# 座標を保存
		# 				temp_coords.append((
		# 					y_coord + direction[0], 
		# 					x_coord + direction[1]
		# 				))

		# 				# image.ortho[(
		# 				# 	y_coord + direction[0], 
		# 				# 	x_coord + direction[1]
		# 				# )] = [0, 255, 0]

		# image.ortho[sediment_coords] = [0, 0, 255]


	def get_directions(self, deg: float) -> tuple[tuple]:
		""" 傾斜方向を画素インデックスに変換し傾斜方向と隣接2方向を取得

		Args:
				deg (float): 角度

		Returns:
				tuple[tuple]: 傾斜方向と隣接2方向
		"""

		# FIXME: 傾斜方向が限定的になっている
		# 注目画素からの移動画素
		if   (math.isnan(deg)):
			# NOTE: 返り値違うかも
			return np.nan, np.nan
		elif (deg > 337.5) or  (deg <= 22.5):
			return self.DIRECTION[7], self.DIRECTION[0], self.DIRECTION[1]
		elif (deg > 22.5)  and (deg <= 67.5):
			return self.DIRECTION[0], self.DIRECTION[1], self.DIRECTION[2]
		elif (deg > 67.5)  and (deg <= 112.5):
			return self.DIRECTION[1], self.DIRECTION[2], self.DIRECTION[3]
		elif (deg > 112.5) and (deg <= 157.5):
			return self.DIRECTION[2], self.DIRECTION[3], self.DIRECTION[4]
		elif (deg > 157.5) and (deg <= 202.5):
			return self.DIRECTION[3], self.DIRECTION[4], self.DIRECTION[5]
		elif (deg > 202.5) and (deg <= 247.5):
			return self.DIRECTION[4], self.DIRECTION[5], self.DIRECTION[6]
		elif (deg > 247.5) and (deg <= 292.5):
			return self.DIRECTION[5], self.DIRECTION[6], self.DIRECTION[7]
		elif (deg > 292.5) and (deg <= 337.5):
			return self.DIRECTION[6], self.DIRECTION[7], self.DIRECTION[0]


	def get_heights(
			self, 
			image: ImageData, 
			directions: tuple[tuple],
			coord: tuple[int, int]
		) -> tuple[int, int, int]:
		""" 注目座標からの標高値を取得

		Args:
				image (ImageData): 画像データ
				directions (tuple[tuple]): 注目座標からの傾斜方向
				coord (tuple[int, int]): 注目座標

		Returns:
				tuple[int, int, int]: 注目座標からの標高値
		"""

		heights = []

		for direction in directions:
			# 注目画素からの傾斜方向の標高値を取得
			heights.append(image.dsm_uav[
				coord[0] + direction[0], 
				coord[1] + direction[1]
			])

		return heights
