import cv2
import numpy as np

from modules import tool
from modules import operation


# # 本番用画像
# path1 = './inputs/dsm_uav.tif'
# path2 = './inputs/dsm_heli.tif'
# path3 = './inputs/dem.tif'
# path4 = './inputs/degree.tif'
# # path5 = './inputs_trim/mask.png'
# # path5 = './inputs_trim/manual_mask.png'
# path5 = './inputs/mask.png'
# path6 = './inputs/uav_img.tif'
# path7 = './inputs/heli_img.tif'


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

path_list = [path1, path2, path3, path4, path5, path6]


# TODO: ラプラシアンフィルタとかを領域に使って勾配を求める
# TODO: テクスチャかDLで建物検出
# ラベリングの改良
# ラベリングについて領域サイズを一定に
# 領域同士が隣接している領域を輪郭データ等で算出
# 傾斜方向が上から下である領域を平均標高値や傾斜方向で算出
# 建物領域にも矢印があるので除去など


@tool.stop_watch
def main() -> None:
	"""
	メイン関数
	"""
	# クラス初期化
	image_op = operation.ImageOp(path_list)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	image_op.resampling_dsm()

	# 土砂マスクの前処理
	# TODO: 精度向上させる
	print("# マスク画像の前処理")
	image_op.norm_mask(16666, 3)	# 面積の閾値, 拡大倍率
	# image_op.mask = cv2.imread("./outputs/normed_mask.png")

	# 領域分割
	print("# オルソ画像の領域分割")	# 空間半径,範囲半径,最小密度
	# image_op.divide_area(3, 4.5, 100)
	image_op.div_img = cv2.imread("./outputs/meanshift.png").astype(np.float32)

	# 輪郭・重心データ抽出・ラベル画像作成
	print("# 領域分割結果から領域データ抽出・ラベル画像の生成")
	image_op.calc_contours()

	# 標高値の正規化
	print("# 標高値の正規化")
	image_op.norm_elevation_meter()

	# 土砂マスクを利用し堆積差分算出
	print("# 堆積差分算出")
	image_op.calc_sedimentation()

	# 土砂移動推定
	print("# 土砂移動推定")
	image_op.calc_movement()


# メイン関数
if __name__ == "__main__":
	# テスト実行
	main()
