import cv2
import csv
import time
import random
import numpy as np
from functools import wraps

# from modules.operation import ImageOp


def stop_watch(func: callable) -> callable:
	"""
	関数の実行時間測定
	"""
	@wraps(func)
	def wrapper(*args, **kargs) :
		start = time.time()
		result = func(*args, **kargs)
		process_time =  time.time() - start
		print(f"-- {func.__name__} : {int(process_time)}[sec] / {int(process_time/60)}[min]")
		return result
	return wrapper


def save_resize_image(
	path: str, 
	image: np.ndarray, 
	size: tuple
) -> None:
	"""
	画像を縮小して保存

	path: 保存先のパス
	image: 画像データ
	size: 保存サイズ
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


def show_image_size(obj) -> None:
	"""
	画像サイズの確認

	obj: クラスオブジェクト
	"""
	print("- uav-size  :", obj.dsm_uav.shape)
	print("- heli-size :", obj.dsm_heli.shape)
	print("- dem-size  :", obj.dem.shape)
	print("- deg-size  :", obj.degree.shape)
	print("- mask-size :", obj.mask.shape)
	print("- img-size  :", obj.ortho.shape)

	return


def csv2self(self) -> None:
	"""
	PyMeanShiftで取得したcsvファイルをselfに格納
	"""
	# 領域データ読み込み
	coords_list = load_csv("./area_data/pms_coords.csv")
	# pix_list    = load_csv("./area_data/pms_pix.csv")

	# selfに格納
	self.pms_coords = coords_list
	# self.pms_pix    = pix_list

	return


def calc_min_max(dsm: np.ndarray) -> tuple[float, float]:
	# 最大値と最小値を算出
	_min, _max = np.nanmin(dsm), np.nanmax(dsm)

	return _min, _max


def calc_ave_sd(dsm: np.ndarray) -> tuple[float, float]:
	# 平均と標準偏差を算出
	ave, sd = np.nanmean(dsm), np.nanstd(dsm)

	return ave, sd


def and_operation(
	list1: list[int], 
	list2: list[int], 
	list3: list[int]
) -> list:
	"""
	条件を満たす領域の組を論理積を取り抽出

	list1: 条件1
	list2: 条件2
	list3: 条件3
	"""
	# 3つの条件の論理積を取る
	area_list = []

	# 領域数
	area_num = len(list1)

	# 3つの配列全てに存在する要素を抽出
	for i in range(area_num):
		and_area_list = set(list1[i]) & set(list2[i]) & set(list3[i])
		and_area_list = list(and_area_list)
		area_list.append(and_area_list)

	return area_list


def and_operation_2(list1: list[int], list2: list[int]) -> list[int]:
	"""
	条件を満たす領域の組を論理積を取り抽出

	list1: 条件1
	list2: 条件2
	"""
	# 2つの条件の論理積を取る
	area_list = []

	# 領域数
	area_num = len(list1)

	# 3つの配列全てに存在する要素を抽出
	for i in range(area_num):
		and_area_list = set(list1[i]) & set(list2[i])
		and_area_list = list(and_area_list)
		area_list.append(and_area_list)

	return area_list


def draw_color(
	img: np.ndarray, 
	idx: np.ndarray, 
	color: list[int]
) -> np.ndarray:
	"""
	指定インデックスの領域を着色

	img: 着色する画像
	idx: 着色する領域
	color: 着色する色のRGBデータ
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
	"""
	バイナリデータの書き出し

	data: 画像データ
	path: 保存パス
	"""
	f = open(path, 'w')
	for i in data:
		for j in i:
			f.write(str(j) + ' ')
		f.write(str('\n'))
	f.close()

	return


def load_csv(path: str) -> list[tuple]:
	"""
	cvsデータを読み込みヘッダを削除

	path: パス
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


def coordinates2contours(self, coordinates: list[tuple]) -> list[tuple]:
	"""
	領域の座標データから輪郭座標を取得

	coordinates: 注目領域の座標群
	"""
	# 注目領域のマスク画像を作成
	mask = draw_region(self, coordinates)

	# 輪郭抽出
	contours, _ = cv2.findContours(
		mask.astype(np.uint8), 
		cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_NONE
	)

	return [(c[0, 1], c[0, 0]) for c in contours[0]]


def draw_region(self, coords: list[tuple]) -> list[tuple]:
	"""
	与えられた座標を領域とし特定画素で埋める

	coords: 領域の座標群
	"""
	# キャンパス描画
	campus = np.zeros((self.size_2d[1], self.size_2d[0]))

	for coord in coords:
		# 領域座標を白画素で埋める
		campus[coord] = 255

	return campus


def draw_label(
	self, 
	label_img: np.ndarray, 
	coords: list[tuple]
) -> tuple[np.ndarray, np.ndarray]:
	"""
	与えられた座標を領域とし特定画素で埋める

	label_img: ラベル画像
	coords: 領域の座標群
	"""
	# キャンパス描画
	campus = np.zeros((self.size_2d[1], self.size_2d[0]))

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


def decode_area(region: tuple) -> tuple:
	"""
	領域データをlabel, coords, areaに変換

	region: PyMeanShiftで抽出した領域csvデータ
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


def is_index(self, coordinate: tuple[int, int]) -> bool:
	"""
	タプル型座標が画像領域内に収まっているかを判定

	coordinate: タプル型座標
	"""
	# (0 <= x < width) & (0 <= y < height)
	if (	((coordinate[0] >= 0) and (coordinate[0] < self.size_3d[0])) 
		and ((coordinate[1] >= 0) and (coordinate[1] < self.size_3d[1]))):
			return True
	else:
		return False


def draw_vector(
	self, 
	region: tuple, 
	labels: list[int]
) -> None:
	"""
	土砂移動の矢印を描画

	region: 注目領域の領域データ
	labels: 流出先の領域ラベルID
	"""
	# 各ラベルに対して
	for label in labels:
		# 流出元の重心座標
		cy, cx   = region["cy"], region["cx"]

		# 流出先の重心座標
		_cy, _cx = self.region[label]["cy"], self.region[label]["cx"]
		
		# 矢印を描画
		cv2.arrowedLine(
			img=self.ortho,     # 画像
			pt1=(cx, cy),       # 始点
			pt2=(_cx, _cy),     # 終点
			color=(20,20,180),  # 色
			thickness=2,        # 太さ
			tipLength=0.4       # 矢先の長さ
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
