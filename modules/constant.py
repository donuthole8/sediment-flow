import numpy as np


# TODO: 画像サイズ，領域データ，傾斜方向データ，精度評価データを追加

# 解像度（cm） この画像はまた違う
RESOLUTION: float = 7.5


# (w, h) = 12 x 13
# direction: 方向 0<= dir <360
# distance: 水平移動距離 0<= dis <=70.7 (50 ** 2) ,,,
# 　※7月11日と被災前の比較（被災前写真：平成21年4月撮影）で作成

## 違うかも
# 小屋浦切り抜き画像正解データ
# 約 400m x 400m
# 5mメッシュ -> 80 x 80
CORRECT_DATA_TRIM_100: list[dict[int, float]] = [
  # 1 - 3 rows
  [
  {"direction": 170, "distance": 5}, {"direction": 175, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction":220, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
  ], [
  {"direction": 160, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": 220, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], [
  {"direction": 160, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": 200, "distance": 5},
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], 

  # 4 - 6 rows
  [
  {"direction": 135, "distance": 5}, {"direction": 175, "distance": 5}, {"direction": 190, "distance": 5},
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": 190, "distance": 5},
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], [
  {"direction": np.nan, "distance": np.nan}, {"direction": 170, "distance": 5}, {"direction": 200, "distance": 5},
  {"direction": 225, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": 190, "distance": 5}, {"direction": 200, "distance": 5}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan},
  ], [
  {"direction": np.nan, "distance": np.nan}, {"direction": 225, "distance": 5}, {"direction": 225, "distance": 5},
  {"direction": 230, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": 225, "distance": 5}, {"direction": 220, "distance": 5}, {"direction": 250, "distance": 5}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan},
  ],

  # 7 - 9 rows
  [
  {"direction": 250, "distance": 5}, {"direction": 250, "distance": 5}, {"direction": 250, "distance": 5},
  {"direction": 245, "distance": 5}, {"direction": 245, "distance": 5}, {"direction": 225, "distance": 5},
  {"direction": 225, "distance": 5}, {"direction": 225, "distance": 5}, {"direction": 225, "distance": 5},
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan},
  ], [
  {"direction": 250, "distance": 5}, {"direction": 250, "distance": 5}, {"direction": 260, "distance": 5},
  {"direction": 260, "distance": 5}, {"direction": 260, "distance": 5}, {"direction": 260, "distance": 5},
  {"direction": 260, "distance": 5}, {"direction": 250, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], [
  {"direction": 265, "distance": 5}, {"direction": 275, "distance": 5}, {"direction": 270, "distance": 5},
  {"direction": 275, "distance": 5}, {"direction": 270, "distance": 5}, {"direction": 270, "distance": 5},
  {"direction": 250, "distance": 5}, {"direction": 225, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], 

  # 10 - 13 row
  [
  {"direction": 290, "distance": 5}, {"direction": 290, "distance": 5}, {"direction": 275, "distance": 5},
  {"direction": 280, "distance": 5}, {"direction": 275, "distance": 5}, {"direction": 290, "distance": 5},
  {"direction": 275, "distance": 5}, {"direction": 275, "distance": 5}, {"direction": 275, "distance": 5},
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], [
  {"direction": 290, "distance": 5}, {"direction": 290, "distance": 5}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": 325, "distance": 5}, {"direction": 320, "distance": 5}, {"direction": 325, "distance": 5},
  {"direction": 295, "distance": 5}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], [
  {"direction": np.nan, "distance": np.nan}, {"direction": 325, "distance": 5}, {"direction": 325, "distance": 5},
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], [
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, {"direction": np.nan, "distance": np.nan}, 
  ], 
]
