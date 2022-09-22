"""
災害前のDSMに航空画像でなく国土地理院DEMを使用
"""
import cv2
import numpy as np

from modules import tool
from modules import operation


# # 本番用画像
# path1 = './inputs/dsm_uav.tif'
# path2 = './inputs/dem.tif'
# path3 = './inputs/dem.tif'
# path4 = './inputs/degree.tif'
# # path5 = './inputs_trim/mask.png'
# # path5 = './inputs_trim/manual_mask.png'
# path5 = './inputs/mask.png'
# path6 = './inputs/uav_img.tif'
# path7 = './inputs/heli_img.tif'


# トリミングしたテスト用画像
path1 = './inputs_trim/dsm_uav_re.tif'
path2 = './inputs_trim/dem.tif'
path3 = './inputs_trim/dem.tif'
path4 = './inputs_trim/degree.tif'
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
def main():
	"""
	メイン関数
	"""
	# TODO: 建物輪郭データとか？？読み込み？？
	# TODO: 建物使うべき？？

	# TODO: 土砂マスク画像の作成に中山さんの手法を適用する・海領域の除去・影領域の対処・前後差分の検討
	# FIXME: 画素値が0-255に正規化されている
	# クラス初期化
	image_op = operation.ImageOp(path_list)

	# # DEMより傾斜データを抽出
	# # NOTE: リサンプリング後に行った方が良いかも
	# print("# DEMより傾斜データを抽出")
	# image_op.dem2gradient(10)	# メッシュサイズ

	# # 傾斜方位の正規化（0-255 -> 0-360）
	# print("# 傾斜方向の正規化")
	# image_op.norm_degree()

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	image_op.resampling_dsm()

	# 画像サイズの確認
	print("# 入力画像のサイズ確認")
	tool.show_image_size(image_op)

	# 土砂マスクの前処理
	# TODO: 精度向上させる
	print("# マスク画像の前処理")
	image_op.norm_mask(16666, 3)	# 面積の閾値, 拡大倍率
	# image_op.mask = cv2.imread("./outputs/normed_mask.png")

	# 土砂マスク
	print("# 土砂マスクによる土砂領域抽出")
	image_op.extract_sediment()

	# 領域分割
	# NOTE: 領域分割画像のみ取得する（ラベル画像・領域数必要無い）場合PyMeanShiftを変更し処理時間を短縮できるかも
	print("# オルソ画像の領域分割")	# 空間半径,範囲半径,最小密度
	image_op.divide_area(3, 4.5, 100)
	# image_op.divide_area(5, 4.5, 100)
	# image_op.divide_area(15, 4.5, 300)
	# image_op.divide_area(2, 2, 20)
	# image_op.div_img = cv2.imread("./outputs/meanshift.png").astype(np.float32)

	# TODO: 大きすぎた領域のみさらに領域分割する

	# 輪郭・重心データ抽出・ラベル画像作成
	print("# 領域分割結果から領域データ抽出・ラベル画像の生成")
	image_op.calc_contours()

	# 標高値の正規化
	# TODO: 絶対値で算出できるよう実装を行う
	print("# 標高値の正規化")
	# image_op.norm_elevation_0to1()
	image_op.norm_elevation_meter()

	# # 標高座標の最適化
	# # TODO: 実装する,論文手法
	# print("# 標高座標の最適化")
	# image_op.norm_cord()

	# 建物領域の検出
	print("# 建物領域を検出する")
	image_op.extract_building()

	# TODO: 建物領域の標高値を地表面と同じ標高値にする
	print("# 建物領域の標高値を地表面標高値に補正")
	image_op.norm_building()

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