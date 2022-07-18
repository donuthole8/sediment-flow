import cv2
import numpy as np
from scipy import ndimage


def lucas_kanade(gray2, gray1):
	# Shi-Tomasi法のパラメータ（コーナー：物体の角を特徴点として検出）
	ft_params = dict(maxCorners=10000,  # 特徴点の最大数
		qualityLevel=0.3,  # 特徴点を選択するしきい値で、高いほど特徴点数は厳選されて減る
		minDistance=7,  # 特徴点間の最小距離 (特徴点から近い点は、特徴点としない)
		blockSize=7)  # 特徴点の計算に使うブロック（周辺領域）サイズ

	# Lucas-Kanade法のパラメータ（追跡用）
	lk_params = dict(winSize=(15, 15),  # オプティカルフローの推定の計算に使う周辺領域サイズ
		maxLevel=2,  # ピラミッド数 (デフォルト0：2なら1/4画像まで使用)
		criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # 探索アルゴリズムの終了条件

	# 最初のフレームを取得してレースケール変換
	# ret, frame = cap.read()

	# Shi-Tomasi法で特徴点の検出
	ft1 = cv2.goodFeaturesToTrack(
		gray1, mask=None, **ft_params
	)

	# 出力用の配列を生成
	img = cv2.merge((gray1, gray1, gray1))
	print(img.shape)



	## 本当は繰り返し
	# 次のフレームを取得し、グレースケールに変換
	# ret, frame = cap.read()
	# gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	print(gray1.shape, gray2.shape)

	# Lucas-Kanade法でフレーム間の特徴点のオプティカルフローwp計算
	ft2, status, err = cv2.calcOpticalFlowPyrLK(
			gray1, gray2, ft1, None, **lk_params)

	# オプティカルフローを検出した特徴点を取得（1なら検出）
	good1 = ft1[status == 1]  # 1フレーム目
	good2 = ft2[status == 1]  # 2フレーム目

	# 特徴点とオプティカルフローをフレーム・マスクに描画
	for i, (pt2, pt1) in enumerate(zip(good2, good1)):
			x1, y1 = pt1.ravel()  # 1フレーム目の特徴点座標
			x2, y2 = pt2.ravel()  # 2フレーム目の特徴点座標

			x1, y1 = int(x1), int(y1)
			x2, y2 = int(x2), int(y2)

			print((x1,y1),(x2,y2))

			# 軌跡を描画(過去の軌跡も残すためにmaskに描く)
			img_line = cv2.line(img, (x2, y2), (x1, y1), [0, 0, 200], 1)

	cv2.imwrite("line.png", img_line)
	# cv2.imwrite("circle.png", img_circle)

	return




		# cap.release()

		# # 次のフレーム、ポイントの準備
		# gray1 = gray2.copy() # 次のフレームを最初のフレームに設定
		# ft1 = good2.reshape(-1, 1, 2) # 次の点を最初の点に設定

		# # qキーが押されたら途中終了
		# if cv2.waitKey(30) & 0xFF == ord('q'):
		# 		break

