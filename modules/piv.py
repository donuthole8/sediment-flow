import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
from openpiv import tools, pyprocess, validation, filters, scaling

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import imageio


# PIV解析をする関数
def piv_analysis(img_list, wsize=32, overlap=0, threshold=30):
	count = 0
	# path_list = ["./inputs/img1.png","./inputs/img2.png"]

	# 全画像ファイルをリストしてPIV計算を実施
	for i in range(len(img_list)-1):
		count += 1
		vectors_amps = []                                     # ベクトルの大きさ情報を格納する空配列
		coordinates = []                                      # 座標情報を格納する空配列
		correlation_coef = []                                 # 相関係数情報を格納する空配列

		# グレースケール画像で2枚(i, i+1)の画像を読み込み
		# img1 = cv2.imread(path_list[i], 0)
		# img2 = cv2.imread(path_list[i+1], 0)
		img1, img2 = img_list[i], img_list[i+1]

		# 画像サイズを取得
		h, w, c1 = img1.shape
		h2, w2, c2 = img2.shape

		# ファイルサイズが揃っていない場合はエラー
		if h != h2 or w != w2:
			print('Error:img(i).shape != img(i+1).shape.')
			print('Align image size.')
			break

		w_st = int(w / (wsize - overlap))                     # 幅のストライドを計算
		h_st = int(h / (wsize - overlap))                     # 高さのストライドを計算

		# 画像を走査しながら処理をする（h_st, w_stのサイズでループ）
		for k in range(h_st-1):
			for j in range(w_st-1):
				# テンプレートマッチング部分
				# 抽出範囲の点（左上、右上）を計算
				p_h1 = k * (wsize - overlap)
				p_h2 = p_h1 + wsize
				p_w1 = j * (wsize - overlap)
				p_w2 = p_w1 + wsize

				# パターンマッチングから移動先座標を計算
				template = img1[p_h1:p_h2, p_w1:p_w2]                    # img[i]からテンプレート画像を抽出
				method = cv2.TM_CCOEFF_NORMED                            # NCC(正規化相互相関係数)を選択
				res = cv2.matchTemplate(img2, template, method)          # img[i+1]に対するテンプレートマッチングの結果
				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # 最小値と最大値、その位置を取得

				# テンプレート画像の中心座標を計算
				before_w = int(p_w1 + (p_w2 - p_w1) / 2)                 # 探査前のx座標
				before_h = int(p_h1 + (p_h2 - p_h1) / 2)                 # 探査前のy座標
				after_w = int(max_loc[0] + wsize / 2)                    # 探査後のx座標
				after_h = int(max_loc[1] + wsize / 2)                    # 探査後のy座標

				# 評価指標を計算
				dx = after_w - before_w                                  # x増分値
				dy = after_h - before_h                                  # y増分値
				vector_amp = np.sqrt(dx ** 2 + dy ** 2)                  # ベクトルの大きさ
				coordinate = [before_w, before_h, dx, dy]                # 座標と増分値セット

				# データ格納
				vectors_amps.append(vector_amp)
				coordinates.append(coordinate)
				correlation_coef.append(max_val)
				# ----------------------------------------------------------------------------------

		# ここからグラフ描画-------------------------------------
		# フォントの種類とサイズを設定する。
		plt.rcParams['font.size'] = 14
		plt.rcParams['font.family'] = 'Times New Roman'

		# 目盛を内側にする。
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'

		# グラフの上下左右に目盛線を付ける。
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.yaxis.set_ticks_position('both')
		ax1.xaxis.set_ticks_position('both')

		# 背景画像の用意と表示設定
		ax1.axis('off')
		ax1.imshow(img2, cmap='gray')           # 背景画像
		cm = plt.get_cmap('jet')                # カラーマップ
		vsf = 2                                 # ベクトル表示のスケールファクタ

		# 誤ベクトル処理（一度に一定ピクセル（threshold）以上のベクトルは長さを0にする）
		for m in range(len(vectors_amps)):
			print(m, '/', len(vectors_amps)-1)
			if vectors_amps[m] >= threshold:
				coordinates[m][2] = 0
				coordinates[m][3] = 0
				vectors_amps[m] = 0

		# 誤ベクトル処理後のベクトルをMin-Max正規化(cmapでベクトルに色を付けるためだけの変数)
		norm_vectors = (vectors_amps - np.min(vectors_amps)) / (np.max(vectors_amps) - np.min(vectors_amps))

		# 画像にベクトルをプロットする
		for n in range(len(vectors_amps)):
			print(n, '/', len(vectors_amps)-1)          # 進捗表示

			# 長さ0以外の場合に図にベクトル(dx, dyにそれぞれスケールを乗じた後のベクトル)を描画
			if vectors_amps[n] != 0:
				ax1.arrow(x=coordinates[n][0], y=coordinates[n][1],
						  dx=vsf * coordinates[n][2], dy=vsf * coordinates[n][3],
						  width=0.01, head_width=10, color=cm(norm_vectors[n]))

		# レイアウトタイト設定
		fig.tight_layout()

		# out_dirで指定したフォルダが無い時に新規作成
		if os.path.exists("./piv_out"):
			pass
		else:
			os.mkdir("./piv_out")

		# 画像保存パスを連番で準備
		path = os.path.join(*["./piv_out", str("{:05}".format(count)) + '.png'])

		# 画像を保存する
		plt.savefig(path)

		# 画像作成の進捗を表示
		print(count, '/', len(img_list) - 1)

	return


def open_piv(frame_a, frame_b, path):
	"""
	OpenPIVによるPIV解析

	frame_a: 災害後DSM
	frame_b: 災害前DSM
	paht: 画像パス
	"""
	# パラメータ
	winsize = 32 			# pixels, interrogation window size in frame A
	searchsize = 38 	# pixels, search area size in frame B
	overlap = 17 			# pixels, 50% overlap
	dt = 0.02 				# sec, time interval between the two frames

	# 速度ベクトルのu,v成分,各ベクトルの相互相関マップの信号とノイズ比
	u0, v0, sig2noise = pyprocess.extended_search_area_piv(
			frame_a.astype(np.int32),
			frame_b.astype(np.int32),
			window_size=winsize,
			overlap=overlap,
			dt=dt,
			search_area_size=searchsize,
			sig2noise_method='peak2peak',
	)

	# 各セルの中心座標を算出
	x, y = pyprocess.get_coordinates(
			image_size=frame_a.shape,
			search_area_size=searchsize,
			overlap=overlap,
	)

	# マスク画像を作成
	u1, v1, mask = validation.sig2noise_val(
			u0, v0,
			sig2noise,
			threshold = 1.05,
	)

	# アウトライア（異常ベクトル）を近傍平均ベクトルにて置換
	u2, v2 = filters.replace_outliers(
			u1, v1,
			method='localmean',
			max_iter=3,
			kernel_size=3,
	)

	# ピクセルをmmへ変換
	x, y, u3, v3 = scaling.uniform(
			x, y, u2, v2,
			scaling_factor = 96.52,  # 96.52 pixels/millimeter
	)

	# 座標原点を変換
	x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

	# ASCIIにて保存
	tools.save(x, y, u3, v3, mask, 'exp1_001.txt')

	# ベクトルをプロット（赤色はS2Nが閾値以下）
	fig, ax = plt.subplots(figsize=(8,8))
	tools.display_vector_field(
			'exp1_001.txt',
			ax=ax, scaling_factor=96.52,
			scale=50, # scale defines here the arrow length
			width=0.0035, # width is the thickness of the arrow
			on_img=True, # overlay on the image
			image_name=path,
	)
	
