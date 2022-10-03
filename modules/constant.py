import numpy as np
# TODO: 画像サイズ，領域データ，傾斜方向データ，精度評価データを追加

# 解像度（cm） この画像はまた違う
RESOLUTION = 7.5

# 小屋浦切り抜き画像正解データ
# 約 400m x 400m
# 5mメッシュ -> 80 x 80

# direction: 方向 0<= dir <360
# distance: 水平移動距離 0<= dis <=7.07,,,
CORRECT_DATA_TRIM = [
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
  [{"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5}, {"direction": 170, "distance": 5},], 
]




# CORRECT_DATA = [
#   [
#     {"direction": 135, "distance": 5}, 
#     {"direction": 170, "distance": 3}, 
#     {"direction": np.nan, "distance": np.nan}, 
#     ...
#   ],
#   [...],
#   [...],
#   ...
# ]
