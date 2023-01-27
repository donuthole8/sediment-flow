import cv2
import numpy as np

from modules.utils import common_util
from modules.utils import image_util
from modules.image_data import ImageData
from modules.calc_geo_data import CalcGeoData
from modules.resampling import Resampling
from modules.mask_processing import MaskProcessing
from modules.region_processing import RegionProcessing
from modules.analyze_image import AnalyzeImage
from modules.calc_sedimentation import CalcSedimentation
from modules.calc_movement_mesh import CalcMovementMesh
from modules.accuracy_valuation import AccuracyValuation


# 画像のファイルパス
path1 = './inputs/atami/trim/dsm_raw.tif'
path2 = './inputs/atami/trim/dem.tif'
path3 = './inputs/atami/trim/dem.tif'
path4 = './inputs/atami/trim/degree.tif'
path5 = './inputs/atami/trim/normed_mask.png'
path6 = './inputs/atami/trim/ortho_img.tif'
path7 = './inputs/atami/trim/building_polygon.png'

path_list = [path1, path2, path3, path4, path5, path6, path7]


# TODO: ラプラシアンフィルタとかを領域に使って勾配を求める
# TODO: テクスチャかDLで建物検出
# 領域同士が隣接している領域を輪郭データ等で算出
# 傾斜方位が上から下である領域を平均標高値や傾斜方位で算出
# 建物領域にも矢印があるので除去など


@common_util.stop_watch
def main() -> None:
	"""
	メイン関数
	"""
	# クラス初期化
	image = ImageData(path_list)

	# 傾斜方位データの正規化（0-255 -> 0-360）
	# TODO: 値がおかしい
	print("# 傾斜方位データの正規化")
	CalcGeoData().norm_degree_v2(image)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 災害前DEM切り抜き・解像度のリサンプリング")
	Resampling(image)

	# 領域分割
	# NOTE: 領域分割画像のみ取得する（ラベル画像・領域数必要無い）場合PyMeanShiftを変更し処理時間を短縮できるかも
	print("# オルソ画像の領域分割")
	# RegionProcessing().area_division(image, 3, 4.5, 100)
	RegionProcessing().area_division(image, 10, 10, 300)
	# image.div_img = cv2.imread("./outputs/meanshift.png").astype(np.float32)

	# 輪郭・重心データ抽出・ラベル画像作成
	# TODO: 大きすぎた領域のみさらに領域分割する
	# NOTE: 処理が重い
	print("# 領域分割結果から領域データ抽出・ラベル画像の生成")
	RegionProcessing().get_region_data(image)

	# 土砂マスクの前処理
	# TODO: 土砂マスク画像の作成に中山さんの手法を適用する・海領域の除去・影領域の対処・前後差分の検討
	print("# マスク画像の前処理")
	# NOTE: こっちでマスク画像作成するとエラーになる
	# NOTE: 別リポジトリで作成
	# MaskProcessing().norm_mask(image, 16666, 3)
	image.mask = cv2.imread("./inputs/atami/trim/normed_mask.png")

	# 土砂マスク
	# TODO: 隣接領域抽出のコスト削減のためにこれを行う
	print("# 土砂マスクによる土砂領域抽出")
	MaskProcessing().apply_mask(image)

	# 標高値の正規化
	# TODO: 絶対値で算出できるよう実装を行う
	print("# 標高値の正規化")
	CalcGeoData().norm_elevation_meter(image)
	# CalcGeoData().norm_elevation_sd(image)
	# CalcGeoData().norm_elevation_0to1(image)

	# # 標高座標の最適化
	# # TODO: 論文手法を実装する
	# print("# 標高座標の最適化")
	# CalcGeoData().norm_coord(image)

	# # テクスチャ解析
	# print("# テクスチャ解析")
	# # AnalyzeImage().texture_analysis(image)
	# image.dissimilarity = cv2.imread(path8, cv2.IMREAD_ANYDEPTH).astype(np.float32)

	# # エッジ抽出
	# print("# エッジ抽出")
	# AnalyzeImage().edge_analysis(image)

	# 建物領域の検出
	print("# 建物領域を検出する")
	RegionProcessing().extract_building(image)
	# image.bld_mask = cv2.imread(path9)

	# 建物領域の標高値補正
	print("# 建物領域の標高値を地表面標高値に補正")
	# TODO: 建物領域の標高値を地表面と同じ標高値にする
	RegionProcessing().norm_building(image)

	# 土砂マスクを利用し堆積差分算出
	print("# 堆積差分算出")
	CalcSedimentation(image)

	# メッシュベースでの土砂移動推定
	print("# メッシュベースでの土砂移動推定")
	calc_movement_result = CalcMovementMesh(100, image.size_2d).main(image)

	# 精度評価
	print("# 精度評価")
	AccuracyValuation(calc_movement_result).main()


# メイン関数
if __name__ == "__main__":
	# テスト実行
	main()
