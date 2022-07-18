import cv2
from cv2 import norm
import numpy as np
from torch import div

from modules import tif
from modules import piv
from modules import tool
from modules import driver
from modules import process
from modules import track
from modules import test_code
from modules import flow_r


# トリミングしたテスト用画像
path1 = './inputs_trim/dsm_uav_re.tif'
path2 = './inputs_trim/dsm_heli.tif'
path3 = './inputs_trim/dem.tif'
path4 = './inputs_trim/degree.tif'
path5 = './inputs_trim/mask.png'
# path5 = './inputs/manual_mask.png'
path6 = './inputs_trim/uav_img_re.tif'
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


def piv_ana():
	# 読み込み
	# uav  = cv2.imread(path6)
	# heli = cv2.imread(path7)
	# uav  = tif.load_tif(path1).astype(np.float32)
	# heli = tif.load_tif(path2).astype(np.float32)
	# mask = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)

	# # 航空画像のDSMとDEMの切り抜き・リサンプリング
	# print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	# uav, heli, _, _, mask = driver.resampling_dsm(uav, heli, uav, uav, mask)

	# # 土砂マスク
	# print("# 土砂マスクによる土砂領域抽出")
	# # uav  = process.extract_sediment(uav,  mask)
	# # heli = process.extract_sediment(heli, mask)
	# uav = process.remove_black_pix(uav, "./outputs/vegitation.png")
	# heli = process.remove_black_pix(heli, "./outputs/vegitation.png")

	# PIV解析
	print("# PIV解析")
	piv.piv_analysis([heli, uav])
	# piv.open_piv([heli, uav])


def open_piv():
	# 読み込み
	uav, heli = cv2.imread(path1).astype(np.float32), cv2.imread(path2).astype(np.float32)
	# heli, uav = cv2.imread(path1).astype(np.float32), cv2.imread(path2).astype(np.float32)
	mask = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	uav, heli, _, _, mask = driver.resampling_dsm(uav, heli, uav, uav, mask)
	uav, heli = cv2.split(uav)[0], cv2.split(heli)[0]

	# 土砂マスク
	print("# 土砂マスクによる土砂領域抽出")
	uav  = process.extract_sediment(uav,  mask)
	heli = process.extract_sediment(heli, mask)

	# PIV解析
	print("# PIV解析")
	# piv.open_piv([heli, uav])
	piv.open_piv(uav, heli, path1)


def resize():
	test_code.resize_img(path6, "./inputs_trim/uav_img_re.tif", (1000 ,1000))


def mask():
	test_code.make_mask('./inputs_trim/answer.tif')


def flow_r_ana():
	flow_r.main()


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

	# 画像サイズの確認
	print("# 入力画像のサイズ確認")
	tool.show_image_size(
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
	# normed_mask = cv2.imread("./inputs/manual_mask.png")
	normed_mask = mask
	# マスク画像の前処理無し
	# normed_mask = driver.use_org_mask(mask)

	# 領域分割
	print("# 土砂領域の領域分割")
	# div_img = driver.divide_area(path6, 3, 4.5, 100)
	div_img = cv2.imread("./outputs/meanshift.png")

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


# オプティカルフロー
def opt_flow():
	# 読み込み
	uav, heli = cv2.imread(path1).astype(np.float32), cv2.imread(path2).astype(np.float32)
	mask = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)
	dem  = tif.load_tif(path3).astype(np.float32)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	uav, heli, dem, _, mask = driver.resampling_dsm(uav, heli, dem, dem, mask)
	uav, heli = cv2.split(uav)[0], cv2.split(heli)[0]

	# 標高モデルのマッチング
	# TODO: 絶対値で算出できるよう実装を行う
	print("# 標高値の正規化")
	dsm_uav, dsm_heli = driver.norm_elevation(uav, heli, dem)

	# 土砂マスク
	print("# 土砂マスクによる土砂領域抽出")
	uav  = process.extract_sediment(uav,  mask)
	heli = process.extract_sediment(heli, mask)

	# 型変換
	uav, heli = uav.astype(np.uint8), heli.astype(np.uint8)

	# # サンプル画像
	# uav, heli = cv2.imread("test1.TIF"), cv2.imread("test2.TIF")
	# uav, heli = cv2.cvtColor(uav, cv2.COLOR_BGR2GRAY), cv2.cvtColor(heli, cv2.COLOR_BGR2GRAY)
	# heli = cv2.resize(heli, (uav.shape[1],uav.shape[0]))

	# print(max(uav), min(uav))
	# print(max(heli), min(heli))

	# オプティカルフロー
	track.lucas_kanade(uav, heli)


def labeling():
	# 入力画像読み込み
	dsm_uav  = tif.load_tif(path1).astype(np.float32)
	dsm_heli = tif.load_tif(path2).astype(np.float32)
	dem_org  = tif.load_tif(path3).astype(np.float32)
	deg      = tif.load_tif(path4).astype(np.float32)
	mask     = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)

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
	# normed_mask = cv2.imread("./inputs/manual_mask.png")
	normed_mask = mask
	# マスク画像の前処理無し
	# normed_mask = driver.use_org_mask(mask)

	# 領域分割
	print("# 土砂領域の領域分割")
	# img = tif.load_tif(path6)
	img = cv2.imread(path6)
	img = cv2.resize(img, (dsm_uav.shape[1], dsm_uav.shape[0]))
	div_img, num = driver.divide_area(img, 3, 4.5, 100)
	# div_img = cv2.imread("./outputs/meanshift.png")

	# ラベリング
	print("# ラベリング")
	# driver.labeling_color(normed_mask, div_img)
	driver.labeling_color_v1(mask, img)


def pymeanshift():
	# 領域分割
	print("# 土砂領域の領域分割")
	# img = cv2.imread(path6)
	img = cv2.imread("./inputs_trim/Google.png")

	# print("size: ", (img.shape[1], img.shape[0]))
	img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)))

	div_img, num = driver.divide_area(img, 3, 4.5, 100)

	print("size: ", (img.shape[1], img.shape[0]))

# Connetがあやしい
# TransitiveClosureもちょい
# PruneはちがうかもMinがあｒ

# メイン関数
if __name__ == "__main__":
	# テスト実行
	# main()

	# マスク画像作成
	# mask()

	# PIV解析
	# piv_ana()
	# open_piv()

	# 画像サイズ変更
	# resize()

	# Flow-R
	# flow_r_ana()
	
	# オプティカルフロー
	# opt_flow()

	# ラベリング
	# labeling()

	# PyMeanShift
	pymeanshift()
	
