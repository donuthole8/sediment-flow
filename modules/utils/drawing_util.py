import cv2
import math
import random
import numpy as np

from modules.image_data import ImageData
from modules.calc_movement_mesh import CalcMovementMesh


def draw_color(img: np.ndarray, idx: np.ndarray, color: list[int]) -> np.ndarray:
	"""	指定インデックスの領域を着色

	Args:
			img (np.ndarray): 着色する画像
			idx (np.ndarray): 着色する領域の座標
			color (list[int]):  着色する色のRGBデータ

	Returns:
			np.ndarray: 着色後の画像
	"""
	# 透過率
	al = 0.45

	# チャンネルを分離
	b, g, r = cv2.split(img)

	# 画像を着色
	b[idx] = b[idx] * al + color[0] * (1 - al)
	g[idx] = g[idx] * al + color[1] * (1 - al)
	r[idx] = r[idx] * al + color[2] * (1 - al)

	# チャンネルを結合
	res = np.dstack((np.dstack((b, g)), r))

	return res


def draw_region(image: ImageData, coords: list[tuple]) -> list[tuple]:
	"""	与えられた座標を領域とし特定画素で埋める

	Args:
			image (ImageData): 画像データ
			coords (list[tuple]): 領域の座標群

	Returns:
			list[tuple]: 1領域のマスク画像
	"""
	# キャンパス描画
	campus = np.zeros(image.size_2d)

	for coord in coords:
		# 領域座標を白画素で埋める
		campus[coord] = 255

	return campus



def draw_label(
		image: ImageData, 
		label_img: np.ndarray, 
		coords: list[tuple]
	) -> tuple[np.ndarray, np.ndarray]:
	""" 与えられた座標を領域とし特定画素で埋める

	Args:
			image (ImageData): 画像データ
			label_img (np.ndarray): ラベル画像
			coords (list[tuple]): 領域の座標群

	Returns:
			tuple[np.ndarray, np.ndarray]: 1領域のラベル画像・マスク画像
	"""
	# キャンパス描画
	campus = np.zeros(image.size_2d)

	# ランダム色を生成
	color = [
		random.randint(0, 255),
		random.randint(0, 255),
		random.randint(0, 255),
	]

	for coord in coords:
		# ランダム色のラベル画像を作成
		label_img[coord] = color
		# 領域座標を白画素で埋める
		campus[coord] = 255

	return label_img, campus



def draw_vector(image: ImageData,  region: tuple, labels: list[int]) -> None:
	"""	土砂移動の矢印を描画

	Args:
			image (ImageData): 画像データ
			region (tuple): 注目領域の領域データ
			labels (list[int]): 流出先の領域ラベルID
	"""
	# 精度評価データ
	answer_direction = []
	answer_distance  = []

	# 解像度
	resolution = 7.5


	# 各ラベルに対して
	for i, label in enumerate(labels):
		# 流出元の重心座標
		cy, cx   = region["cy"], region["cx"]

		# 流出先の重心座標
		_cy, _cx = image.region[label]["cy"], image.region[label]["cx"]

		# # 始点
		# cv2.circle(self.ortho, (cx, cy), 3, (0, 0, 255), thickness=5, lineType=cv2.LINE_8, shift=0)

		# # 終点
		# cv2.circle(self.ortho, (_cx, _cy), 3, (255, 0, 0), thickness=5, lineType=cv2.LINE_8, shift=0)

		# # 精度評価データ保存
		# answer_direction.append()
		# answer_distance.append(int(dist((cy, cx), (_cy, _cx)) * resolution))


		if (i == 0):
			color = (255,0,0)
		elif (i == 1):
			color = (0,255,0)
		elif (i == 2):
			color = (0,0,255)

		print(i, label, labels, color)

		# 矢印を描画
		cv2.arrowedLine(
			img=image.ortho,     	# 画像
			pt1=(cx, cy),       	# 始点
			pt2=(_cx, _cy),     	# 終点
			# color=(20, 20, 180),  # 色
			color=color,  # 色
			thickness=2,        	# 太さ
		)


		# # 水平距離
		# dis = int(dist((cy, cx), (_cy, _cx)) * resolution)
		# # 水平方向の土砂移動を描画
		# cv2.putText(
		#   img=ortho,                        # 画像
		#   text="hor:"+str(dis)+"cm",        # テキスト
		#   org=(_cx+2, _cy+2),               # 位置
		#   fontFace=cv2.FONT_HERSHEY_PLAIN,  # フォント
		#   fontScale=1,                      # フォントサイズ
		#   color=(0, 255, 0),                # 色
		#   thickness=1,                      # 太さ
		#   lineType=cv2.LINE_AA              # タイプ
		# )

		# # 垂直距離
		# # TODO: DSMで良いか検討
		# dis = int(dsm[cy, cx][0] - dsm[_cy, _cx][0] * 100)
		# # 垂直方向の土砂変化標高
		# cv2.putText(
		#   img=ortho,                        # 画像
		#   text="ver:"+str(dis)+"cm",        # テキスト
		#   org=(_cx+2, _cy+14),              # 位置
		#   fontFace=cv2.FONT_HERSHEY_PLAIN,  # フォント
		#   fontScale=1,                      # フォントサイズ
		#   color=(255, 0, 0),                # 色
		#   thickness=1,                      # 太さ
		#   lineType=cv2.LINE_AA              # タイプ
		# )

	# self.answer_direction.append(answer_direction)
	# self.answer_distance.append(answer_distance)


def draw_vector_8dir(image: ImageData, region: tuple, coords: list[int]) -> None:
	"""	土砂移動の矢印を描画

	Args:
			image (ImageData): 画像データ
			region (tuple): 注目領域の領域データ
			coords (list[int]): 流出先の座標
	"""
	# 精度評価データ
	answer_direction = []
	answer_distance  = []

	# 解像度
	resolution = 7.5

	# 各ラベルに対して
	for i, label in enumerate(coords):
		# 流出元の重心座標
		cy, cx   = region["cy"], region["cx"]

		# # 始点
		# cv2.circle(self.ortho, (cx, cy), 3, (0, 0, 255), thickness=5, lineType=cv2.LINE_8, shift=0)

		# # 終点
		# cv2.circle(self.ortho, (_cx, _cy), 3, (255, 0, 0), thickness=5, lineType=cv2.LINE_8, shift=0)

		# # 精度評価データ保存
		# answer_direction.append()
		# answer_distance.append(int(dist((cy, cx), (_cy, _cx)) * resolution))

		# 矢印の色を決定
		if (i == 0):
			color = (255,100,100)
		elif (i == 1):
			color = (0,0,255)
		elif (i == 2):
			color = (100,255,155)

		print(i, label, coords, color)

		# 矢印を描画
		cv2.arrowedLine(
			img=image.ortho,     	# 画像
			pt1=(cx, cy),       	# 始点
			pt2=(label[1], label[0]),     	# 終点
			# color=(20, 20, 180),  # 色
			color=color,  # 色
			thickness=2,        	# 太さ
		)


		# # 水平距離
		# dis = int(dist((cy, cx), (_cy, _cx)) * resolution)
		# # 水平方向の土砂移動を描画
		# cv2.putText(
		#   img=ortho,                        # 画像
		#   text="hor:"+str(dis)+"cm",        # テキスト
		#   org=(_cx+2, _cy+2),               # 位置
		#   fontFace=cv2.FONT_HERSHEY_PLAIN,  # フォント
		#   fontScale=1,                      # フォントサイズ
		#   color=(0, 255, 0),                # 色
		#   thickness=1,                      # 太さ
		#   lineType=cv2.LINE_AA              # タイプ
		# )

		# # 垂直距離
		# # TODO: DSMで良いか検討
		# dis = int(dsm[cy, cx][0] - dsm[_cy, _cx][0] * 100)
		# # 垂直方向の土砂変化標高
		# cv2.putText(
		#   img=ortho,                        # 画像
		#   text="ver:"+str(dis)+"cm",        # テキスト
		#   org=(_cx+2, _cy+14),              # 位置
		#   fontFace=cv2.FONT_HERSHEY_PLAIN,  # フォント
		#   fontScale=1,                      # フォントサイズ
		#   color=(255, 0, 0),                # 色
		#   thickness=1,                      # 太さ
		#   lineType=cv2.LINE_AA              # タイプ
		# )

	# self.answer_direction.append(answer_direction)
	# self.answer_distance.append(answer_distance)


def draw_mesh(mesh: CalcMovementMesh, image: ImageData) -> None:
	"""
	メッシュの格子線を描画

	mesh: メッシュデータ
	image_data: 画像等データ
	"""
	# x軸に平行な格子線を描画
	for y in range(mesh.mesh_height):
		cv2.line(
			img=image.ortho,   			# 画像
			pt1=(0, mesh.mesh_size * y),  # 始点
			pt2=(
				image.size_2d[1], 
				mesh.mesh_size * y
			),														# 終点
			color=(255, 255, 255),  			# 色
			thickness=2,        					# 太さ
		)

	# y軸に並行な格子線を描画
	for x in range(mesh.mesh_width):
		cv2.line(
			img=image.ortho,     		# 画像
			pt1=(mesh.mesh_size * x, 0),  # 始点
			pt2=(
				mesh.mesh_size * x, 
				image.size_2d[0]
			),														# 終点
			color=(255, 255, 255),  			# 色
			thickness=2,        					# 太さ
		)

	# 空画像を作成
	mesh_label = np.zeros(image.size_2d)

	# メッシュのラベル番号
	# mesh_label_y, mesh_label_x = 0, 0
	label = 0

	for y in range(mesh.mesh_height):
		for x in range(mesh.mesh_width):
			# ラベルを付与
			for coord in mesh.get_mesh_coords(image.size_3d, y, x):
				mesh_label[coord] = label

			

			# ラベルをインクリメント
			# mesh_label_x += 1
			label += 1
		
		# ラベルをインクリメント
		# mesh_label_y += 1
			
	# ラベルテーブルの保存
	mesh.mesh_label = mesh_label

	return


def draw_direction(mesh: CalcMovementMesh, image: ImageData, direction: float) -> None:
	""" メッシュに傾斜方位データを描画

	Args:
			image (ImageData): 画像データ
			mesh (CalcMovementMesh): メッシュデータ
			direction (float): 傾斜方位
	"""
	try:
		# 傾斜方位の角度データを三角関数用の表記に変更
		average_direction_trig = 0
		if (direction >= 90):
			average_direction_trig = direction - 90
		else:
			average_direction_trig = 270 + direction

		# 傾斜方位の座標取得
		y_coord = int((mesh.mesh_size // 2) * math.sin(math.radians(average_direction_trig))) + mesh.center_coord[0]
		x_coord = int((mesh.mesh_size // 2) * math.cos(math.radians(average_direction_trig))) + mesh.center_coord[1]

		# # 矢印を描画
		# cv2.arrowedLine(
		# 	img=image.ortho,	# 画像
		# 	pt1=(mesh.center_coord[1], mesh.center_coord[0]), # 始点
		# 	pt2=(x_coord, y_coord),	# 終点
		# 	color=(0, 0, 255),	# 色			
		# 	thickness=2,	# 太さ
		# )
	except Exception as e:
		print(e, ", average_direction: ", direction)

	return


def draw_min_height(mesh: CalcMovementMesh, image: ImageData, coord: tuple[int, int]) -> None:
	""" メッシュに最小標高値までの矢印を描画

	Args:
			image (ImageData): 画像データ
			mesh (CalcMovementMesh): メッシュデータ
			coord (tuple[int, int]): 
	"""
	# 傾きと切片
	# ac_steep, ac_intercept = (cy-ay)/(cx-ax),(cx*ay-ax*cy)/(cx-ax)

	# メッシュ格子と直線の交点座標を算出
	# coord = 

	try:
		# 矢印を描画
		cv2.arrowedLine(
			img=image.ortho,	# 画像
			pt1=(mesh.center_coord[1], mesh.center_coord[0]), # 始点
			pt2=(coord[1], coord[0]),	# 終点
			color=(0, 255, 0),	# 色			
			thickness=2,	# 太さ
		)
	except Exception as e:
		pass
		# print(e, ", coord: ", coord)

	return