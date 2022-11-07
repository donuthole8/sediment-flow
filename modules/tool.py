import cv2
import csv
import time
import random
import numpy as np
from functools import wraps
from modules.calc_movement_mesh import CalcMovementMesh

from modules.image_data import ImageData
from modules.calc_movement_mesh import CalcMovementMesh


def stop_watch(func: callable) -> callable:
	"""	関数の実行時間測定

	Args:
			func (callable): 測定対象の関数

	Returns:
			callable: 関数
	"""
	@wraps(func)
	def wrapper(*args, **kargs) :
		start = time.time()
		result = func(*args, **kargs)
		process_time =  time.time() - start
		print(f"-- {func.__name__} : {int(process_time)}[sec] / {int(process_time/60)}[min]")
		return result
	return wrapper


def save_resize_image(image: np.ndarray, path: str, size: tuple) -> None:
	"""	画像を縮小して保存

	Args:
			image (np.ndarray): 画像データ
			path (str): 保存先のパス
			size (tuple):  保存サイズ
	"""
	# 画像をリサイズ
	resize_img = cv2.resize(
		image, 
		size, 
		interpolation=cv2.INTER_CUBIC
	)

	# 画像を保存
	cv2.imwrite("./outputs/" + path, resize_img)

	return


def show_image_size(image: ImageData) -> None:
	"""	画像サイズの確認

	Args:
			image (imageData): 画像データ
	"""
	print("- uav-size  :", image.dsm_uav.shape)
	print("- heli-size :", image.dsm_heli.shape)
	print("- dem-size  :", image.dem.shape)
	print("- deg-size  :", image.degree.shape)
	print("- mask-size :", image.mask.shape)
	print("- img-size  :", image.ortho.shape)

	return


def csv2self(image: ImageData) -> None:
	""" PyMeanShiftで取得したcsvファイルをインスタンスに格納

	Args:
			image (imageData): 画像データ
	"""
	# 領域データ読み込み
	coords_list = load_csv("./area_data/pms_coords.csv")
	# pix_list    = load_csv("./area_data/pms_pix.csv")

	# データの保存
	image.pms_coords = coords_list
	# self.pms_pix    = pix_list

	return


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


def write_binfile(data: np.ndarray, path: str) -> None:
	"""	バイナリデータの書き出し

	Args:
			data (np.ndarray): 画像データ
			path (str): 保存パス
	"""

	f = open(path, 'w')
	for i in data:
		for j in i:
			f.write(str(j) + ' ')
		f.write(str('\n'))
	f.close()

	return


def load_csv(path: str) -> list[tuple]:
	""" cvsデータを読み込みヘッダを削除

	Args:
			path (str): csvファイルの入力パス

	Returns:
			list[tuple]: csvデータ
	"""
	# csvデータ読み込み
	with open(path, encoding='utf8', newline='') as f:
		area = csv.reader(f)
		area_list = [a for a in area]
		# ヘッダを削除
		area_list.pop(0)
		# # 背景領域を削除
		# area_list.pop(0)

		return area_list


def coordinates2contours(image: ImageData, coordinates: list[tuple]) -> list[tuple]:
	"""	領域の座標データから輪郭座標を取得

	Args:
			image (ImageData): 画像データ
			coordinates (list[tuple]): 注目領域の座標群

	Returns:
			list[tuple]: 輪郭座標
	"""
	# 注目領域のマスク画像を作成
	mask = draw_region(image, coordinates)

	# 輪郭抽出
	contours, _ = cv2.findContours(
		mask.astype(np.uint8), 
		cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_NONE
	)

	return [(c[0, 1], c[0, 0]) for c in contours[0]]


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


def decode_area(region: tuple) -> tuple[int, list, float]:
	""" 領域データをlabel, coords, areaに変換

	Args:
			region (tuple): PyMeanShiftで抽出した領域csvデータ

	Returns:
			tuple: ラベル番号・座標・面積
	"""
	# 領域ID
	label = int(region[0])
	# 座標
	coords = [
		(int(coord_str[1:-1].split(' ')[0]), int(coord_str[1:-1].split(' ')[1])) 
		for coord_str in region[1:-2]
	]
		# 面積
	area  = int(region[-2])

	return label, coords, area


def is_index(size: tuple[int, int, int], coordinate: tuple[int, int]) -> bool:
	"""	タプル型座標が画像領域内に収まっているかを判定

	Args:
			size (tuple[int, int, int]): 画像サイズ
			coordinate (tuple[int, int]): タプル型座標

	Returns:
			bool: 判別結果
	"""
	# (0 <= y < height) & (0 <= x < width)
	if 	(((coordinate[0] >= 0) and (coordinate[0] < size[0])) 
	and  ((coordinate[1] >= 0) and (coordinate[1] < size[1]))):
		return True
	else:
		return False


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


def draw_mesh(image: ImageData, mesh: CalcMovementMesh) -> None:
	"""
	メッシュの格子線を描画

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
