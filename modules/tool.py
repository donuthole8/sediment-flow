import cv2
import csv
import time
import numpy as np
from PIL import Image
from functools import wraps


def stop_watch(func):
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


def save_resize_image(path, image, size):
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


def show_image_size(obj):
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


def show_area_num():
	"""
	マスク画像の領域数を表示
	"""
	# 領域データ読み込み
	with open("./area_data/region.csv", encoding='utf8', newline='') as f:
		area = csv.reader(f)
		area_list = [a for a in area]

	# 領域数の表示
	print("- area-num :", len(area_list))

	return


def contours2csv(contours, mask, area_th, scale):
	"""
	輪郭データをcsvに保存

	contours: 輪郭データ
	mask: マスク画像
	area_th: 面積の閾値
	scale: 拡大倍率
	"""
	# 画像の高さと幅を取得
	h, w = mask.shape

	# 黒画像
	campus = np.zeros(mask.shape)

	# TODO: スケール分は考慮して割る必要あり
	area_id = 0
	with open('./area_data/area.csv', 'w') as fa:
		with open('./area_data/l-centroid.csv', 'w') as fc:
			# ヘッダを追加
			columns_list = ['id', 'area', 'x_center_of_gravity', 'y_center_of_gravity', 'arc_length']
			columns_str = ','.join(columns_list)
			fa.write(columns_str + '\n')
			fc.write('contours\n')

			for i, contour in enumerate(contours):
				# 面積
				area = cv2.contourArea(contour)
				area = int(area) / scale

				# 輪郭の重心
				M = cv2.moments(contour)
				try:
					cx = int((M["m10"] / M["m00"]) / scale)
					cy = int((M["m01"] / M["m00"]) / scale)
				except:
					cx, cy = 0, 0

				# 輪郭（境界）の長さ
				arclen = (cv2.arcLength(contours[i],True) / scale)

				# 閾値以上の面積の場合画像に出力
				if (area >= area_th):
					normed_mask = cv2.drawContours(campus, contours, i, 255, -1)
					# csvファイルに保存
					data_list = [area_id, area, cx, cy, arclen]
					fa.write(str(data_list) + '\n')
					contour_list = [[c[0][0], c[0][1]] for c in contour]
					fc.write(str(contour_list) + '\n')
					area_id += 1

	# スケールを戻す
	normed_mask = cv2.resize(normed_mask, (int(w/scale), int(h/scale)))

	cv2.imwrite("./output/normed_mask.png", normed_mask)

	return normed_mask


def labeling2csv(area_th, stats, centroids, markers):
	"""
	閾値以下の面積を除去し輪郭データをcsvに保存

	area_th: 面積閾値
	stats: オブジェクトデータ
	centroids: 重心データ
	markers: ラベリングデータ
	"""
	_stats, _centroids, _markers = [], [], []
	count_label = 0

	# csvファイルに保存
	with open('./area_data/l-centroid.csv', 'w') as f:
		writer = csv.writer(f)
		# ヘッダを追加
		columns_list = ['id', 'area', 'x_centroid', 'y_centroid', 'witdh', 'hight']
		writer.writerow(columns_list)

		# for (stat, centroid, marker) in zip(stats, centroids, markers):
		for (stat, centroid) in zip(stats, centroids):
			# 閾値以下の面積を除去
			if (stat[4] > area_th):
				# 領域データを追加
				data_list = [
					count_label,       # ID
					stat[4],           # 面積
					int(centroid[0]),  # 重心x座標
					int(centroid[1]),  # 重心y座標
					stat[2],           # 横サイズ
					stat[3]            # 縦サイズ
				]
				
				# 返却用のデータ
				_stats.append(stat)
				_centroids.append(centroid)
				# _markers.append(marker)
				count_label += 1

				writer.writerow(data_list)

	return _stats, _centroids


@stop_watch
def labeling2centroid():
	# mask = cv2.cvtColor(cv2.resize(mask, (4,3)), cv2.COLOR_RGB2GRAY)
	# print(mask)
	# print(type(mask))

	# カラーラベリングデータ読み込み
	# label_list = np.zeros_like(mask)
	label_list = []
	label_line = []
	# with open("./area_data/dummy.txt", encoding='utf8', newline='') as f:
	with open("./area_data/labeee.csv", encoding='utf8', newline='') as f:
		for i, line in enumerate(f):
			# 空白で区切る
			line = line.split(",")
			# 最後尾要素の改行除去
			line[-1] = line[-1].replace("\n", "")
			# int型に変換
			line = [int(l) for l in line]
			
			# label_list[i] = line
			label_list.append(line)
			label_line += line

	# 領域数
	label_num = np.max(label_list)

	# 画像横サイズ
	w = len(label_list[0])

	# area_list = []
	with open('./area_data/lcc-centroid.csv', 'w') as f:
		writer = csv.writer(f)
		# ヘッダを追加
		columns_list = ['id', 'area', 'x_centroid', 'y_centroid']
		writer.writerow(columns_list)

		# id, 領域数, 重心
		for id in range(2, label_num + 1):
			# 面積
			area = label_line.count(id)

			# 重心
			# TODO: 重心を求めたい
			# idx = [i for i,x in enumerate(label_line) if x == id]
			idx = label_line.index(id)
			x, y = idx % w, idx // w

			# データを追加
			# area_list.append([id, area, x, y])
			writer.writerow([id, area, x, y])

	return 


def calc_min_max(dsm):
	# 最大値と最小値を算出
	_min, _max = np.nanmin(dsm), np.nanmax(dsm)

	return _min, _max


def calc_ave_sd(dsm):
	# 平均と標準偏差を算出
	ave, sd = np.nanmean(dsm), np.nanstd(dsm)

	return ave, sd


def and_operation(list1, list2, list3):
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


def and_operation_2(list1, list2):
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


def draw_color(img, idx, color):
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


def write_binfile(data, path):
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


def img2cvs(path):
	"""
	画素値をcsvにして保存

	path: 画像パス
	"""
	# 画像読込
	img = Image.open(path)

	# モノクロ画像へ変換
	img = img.convert("L")
	width, height = img.size
	# 画像の輝度値をlistで取得
	data = list(img.getdata())

	# 輝度値をcsvファイルで保存
	with open('./area_data/image_data.csv', 'w', newline='') as csvfile:
		spamwriter  = csv.writer(csvfile)
		# 画像データを一行ごと書き込み
		x = 0
		for y in range(height):
			# 一行分のデータ
			line_data = data[x:x+width]
			# 一行分のデータを書き込み
			spamwriter.writerow(line_data)
			x += width


# テスト用メイン関数
if __name__ == "__main__":
	l1 = [[1,2],[0],[]]
	l2 = [[2],  [], []]
	l3 = [[2],  [2],[]]
	l = and_operation(l1, l2, l3)


def load_csv(path):
	"""
	cvsデータの読み込み

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


def decode_area(region):
	"""
	領域データをlabel, cords, areaに変換

	region: PyMeanShiftで抽出した領域csvデータ
	"""
	# 領域ID
	label = int(region[0])
	# 面積
	area  = int(region[-2])
	# 座標
	cords = [
		(int(cord_str[1:-1].split(' ')[0]), int(cord_str[1:-1].split(' ')[1])) 
		for cord_str in region[1:-2]
	]

	return label, cords, area
