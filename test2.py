import cv2
import numpy as np

from modules import tif
from modules import tool
from modules import driver


# トリミングしたテスト用画像
path1 = './inputs_trim/dsm_uav_re.tif'
path2 = './inputs_trim/dsm_heli.tif'
path3 = './inputs_trim/dem.tif'
path4 = './inputs_trim/degree.tif'
# path5 = './inputs_trim/mask.png'
# path5 = './inputs_trim/manual_mask.png'
path5 = './inputs_trim/normed_mask.png'
path6 = './inputs_trim/uav_img.tif'
path7 = './inputs_trim/heli_img.tif'


# # リサイズしたテスト用画像
# path1 = './inputs_re/dsm_uav.tif'
# path2 = './inputs_re/dsm_heli.tif'
# path3 = './inputs_re/dem.tif'
# path4 = './inputs_re/degree.tif'
# # path5 = './inputs_re/mask.png'
# path5 = './inputs_re/manual_mask.png'
# path6 = './inputs_re/uav_img.tif'
# path7 = './inputs_re/heli_img.tif'


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
	img      = cv2.imread(path6)

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

	# 画像サイズの確認
	print("# 入力画像のサイズ確認")
	tool.show_image_size(
			dsm_uav, 
			dsm_heli, 
			dem, 
			deg, 
			mask
	)
	print("- img-size  :", img.shape)

	# 土砂マスクの前処理
	# TODO: 精度向上させる
	print("# マスク画像の前処理")
	normed_mask = driver.norm_mask(mask)
	# 作成済みのマスク画像を使用
	# normed_mask = cv2.imread("./outputs/normed_mask.png")
	# normed_mask = cv2.imread("./inputs/manual_mask.png")
	# normed_mask = mask
	# マスク画像の前処理無し
	# normed_mask = driver.use_org_mask(mask)

	# 土砂マスク
	print("# 土砂マスクによる土砂領域抽出")
	# img = process.extract_sediment(img, mask)
	# uav = process.remove_black_pix(uav, "./outputs/vegitation.png")
	# heli = process.remove_black_pix(heli, "./outputs/vegitation.png")

	# 領域分割
	print("# 土砂領域の領域分割")
	div_img = driver.divide_area(img, 3, 4.5, 100)
	# div_img = driver.divide_area(img, 15, 4.5, 300)
	# div_img = driver.divide_area(img, 2, 2, 20)
	# div_img = cv2.imread("./outputs/meanshift.png")

	# 輪郭・重心データ抽出
	print("# 領域分割・土砂マスク済み画像から輪郭データ抽出")
	driver.calc_contours((div_img.shape[0], div_img.shape[1]))

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
		path6,
	)


# メイン関数
if __name__ == "__main__":
	# テスト実行
	main()
