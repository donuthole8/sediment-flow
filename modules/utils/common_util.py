import time
from functools import wraps


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
