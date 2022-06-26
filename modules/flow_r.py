import cv2
import numpy as np

from modules import tif
from modules import tool
from modules import driver
from modules import process
from modules import test_code


path1 = './inputs_re/dsm_uav.tif'
path2 = './inputs_re/degree.tif'
path3 = './inputs_re/manual_mask.png'
path4 = './inputs_re/uav_img.tif'


def estimate_flow():
	# 斜面崩壊領域を処理対象としてforを回す
	pass

	



def main():
	"""
	Flow-R・中山さん手法による土砂流出予想

	①処理前の斜面崩壊領域の検出結果
	②標高値(DEM)
	③傾斜方位
	アルゴリズムはめっちゃ単純で、
	ⅰ. すべての斜面崩壊領域を処理対象としてfor文を回す
	ⅱ. 注目画素から傾斜方向 + 隣接2方向の画素の標高値を調べる（傾斜方向はQgisの出力を近傍8画素に割り当て,　多分↑が0°になってるので上方向は 337.5° ~ 360° & 0° - 22.5°みたいな？）
	ⅲ. 標高値が注目画素よりも小さければそこも斜面崩壊領域としてもう一回ⅱをやる
	を繰り返してたと思う。（もとは傾斜とか条件つけてたけど全然領域が広がらなかったので標高の大小だけにした気がする）
	正直①のほうに時間かけすぎてかなり適当なので、一番直すべきだったのがここかも
	（参考にならなくて申し訳ない）
	"""
	# 画像の読み込み
	dsm  = tif.load_tif(path1).astype(np.float32)
	deg  = tif.load_tif(path2).astype(np.float32)
	mask = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
	img  = cv2.imread(path4)

	# 航空画像のDSMとDEMの切り抜き・リサンプリング
	print("# 航空画像のDSM・DEM切り抜き・解像度のリサンプリング")
	dsm, _, _, deg, mask = driver.resampling_dsm(dsm, dsm, deg, deg, mask)

	# 土砂領域以外の除去
	print("# 植生領域の除去")
	# process.remove_vegitation(img)
	# img = process.remove_black_pix()

	# カラー画像の領域分割
	print("# 土砂領域の領域分割")
	img = driver.divide_area(img, 3, 4.5, 100)
	# img = cv2.imread("./outputs/meanshift.png")

	# 斜面崩壊領域データをすべて抽出
	print("# 土砂マスク中の領域のみでラベリング")
	# driver.labeling_color_v1(mask, img)
	driver.labeling_bin(mask, img)

	# 流出推定
	estimate_flow()
