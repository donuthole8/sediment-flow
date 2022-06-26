import cv2
from cv2 import norm
import numpy as np
from torch import div

from modules import tif
from modules import tool
from modules import driver

# 入力画像パス
path1 = './inputs/dsm_uav.tif'
path2 = './inputs/dsm_heli.tif'
path3 = './inputs/dem.tif'
path4 = './inputs/degree.tif'
# path5 = './inputs/mask.png'
path5 = './inputs/manual_mask.png'
path6 = './inputs/koyaura.tif'


# TODO: ラプラシアンフィルタとかを領域に使って勾配を求める
# TODO: テクスチャかDLで建物検出


@tool.stop_watch
def main():
	"""
	メイン関数
	"""
	# 入力画像読み込み
	# TODO: マスク画像・DSM・DEMをオリジナルサイズのものを作成（実際のサイズは22610x20662画素）
	# TODO: 土砂マスク画像の作成に中山さんの手法を適用する・海領域の除去・影領域の対処・前後差分の検討
	# FIXME: 画素値が0-255に正規化されている
	dsm_uav  = tif.load_tif(path1).astype(np.float32)
	dsm_heli = tif.load_tif(path2).astype(np.float32)
	dem_org  = tif.load_tif(path3).astype(np.float32)
	deg      = tif.load_tif(path4).astype(np.float32)
	mask     = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)

	# 画像サイズの確認
	print("# 入力画像のサイズ確認")
	tool.show_image_size(
			dsm_uav, 
			dsm_heli, 
			dem_org, 
			deg, 
			mask
	)

	# DEMより傾斜データを抽出
	print("# DEMより傾斜データを抽出")
	grad = driver.dem2gradient(dem_org, 5)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	dsm_uav, dsm_heli, dem, deg, mask = driver.resampling_dsm(
		dsm_uav, 
		dsm_heli, 
		dem_org, 
		deg,
		mask
	)

	# 土砂マスクの前処理
	# TODO: 精度向上させる
	print("# マスク画像の前処理")
	# normed_mask = driver.norm_mask(mask)
	# 作成済みのマスク画像を使用
	# normed_mask = cv2.imread("./outputs/normed_mask.png")
	normed_mask = cv2.imread("./inputs/manual_mask.png")
	# マスク画像の前処理無し
	# normed_mask = driver.use_org_mask(mask)

	# 領域分割
	print("# 土砂領域の領域分割")
	# div_img = driver.divide_area(path6, 3, 4.5, 100)
	div_img = cv2.imread("./outputs/__meanshift.png")

	# ラベリング
	print("# 土砂マスク中の領域のみでラベリング")
	# TODO: 出力画像がおかしい
	# driver.labeling_color_v1(normed_mask, div_img)
	# driver.labeling_color(normed_mask, div_img)
	driver.labeling_bin(normed_mask, div_img)

	# # 土砂マスク中の領域のみを算出
	# print("# 土砂マスク中の領域の輪郭抽出")
	# driver.extract_contours(normed_mask, div_img)

	# 標高モデルのマッチング
	# TODO: 絶対値で算出できるよう実装を行う
	print("# 標高値の正規化")
	dsm_uav, dsm_heli = driver.norm_elevation(
		dsm_uav,
		dsm_heli,
		dem
	)

	# 土砂マスクを利用し堆積差分算出
	print("# 堆積差分算出")
	dsm_sub = driver.calc_sedimentation(
		dsm_uav,
		dsm_heli,
		normed_mask
	)

	# 傾斜方位の正規化（0-255 -> 0-360）
	print("# 傾斜方向の正規化")
	deg = driver.norm_degree(deg)

	# 土砂移動推定
	print("# 土砂移動推定")
	driver.calc_movement(
		dsm_sub,
		dem,
		deg,
		grad, 
		dsm_uav,
		path6
	)


# メイン関数
if __name__ == "__main__":
	main()

	# process.test()
	# tool.labeling2centroid()
	# driver.labeling_bin()
