import csv
from modules.image_data import ImageData


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
