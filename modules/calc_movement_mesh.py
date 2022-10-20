import cv2
import numpy as np

from modules import operation
from modules import tool


class CalcMovementMesh():
  def __init__(self, mesh_size: int) -> None:
    # メッシュサイズ
    self.mesh_size = mesh_size
    
    return

  def main(
    self, 
    image_data: operation.ImageOp, 
  ) -> None:
    """
    メッシュベースでの土砂移動の推定

    image_data: 画像等のデータ
    """
    print("image_op", image_data)
    # メッシュサイズよりメッシュの高さと幅を取得
    mesh_height = (image_data.size_2d[1] // self.mesh_size) + 1
    mesh_width  = (image_data.size_2d[0] // self.mesh_size) + 1

    print("(h, w)", mesh_height, mesh_width)

    # メッシュの格子線を描画
    tool.draw_mesh(
      image_data,
      self.mesh_size,
      mesh_height, 
      mesh_width,
    )
    
    # メッシュ中心が土砂の場合各注目領域に対して処理を行う
    for y in range(mesh_height):
      for x in range(mesh_width):


        # print()
        # print("(y, x) ", y, x)

        # 注目メッシュの中心座標を取得
        center = (
          (self.mesh_size // 2) + y * self.mesh_size, 
          (self.mesh_size // 2) + x * self.mesh_size
        )

        # 注目メッシュの中心座標が土砂領域か判別
        # if not (self.is_sedimentation(image_data, center)):
        if (self.is_sedimentation_mask(image_data, center)):
          pass

          # てん
          cv2.circle(image_data.ortho, (center[1], center[0]), 3, (0, 0, 255), thickness=5, lineType=cv2.LINE_8, shift=0)

    cv2.imwrite("./outputs/mesh.png", image_data.ortho)


  def is_sedimentation(
    self, 
    image_data: operation.ImageOp, 
    center: tuple[int, int],
  ) -> bool:
    """
    土砂領域かどうかを判別

    image_data: 画像等のデータ
    center: 該当領域の中心座標
    """
    # Lab表色系に変換
    Lp, ap, bp = cv2.split(
      cv2.cvtColor(
        image_data.div_img.astype(np.uint8), 
        cv2.COLOR_BGR2Lab)
    )
    Lp, ap, bp = Lp * 255, ap * 255, bp * 255

    # 土砂の判別
    if (Lp[center] > 125) & (ap[center] > 130):
      return True
    else:
      return False


  def is_sedimentation_mask(
    self, 
    image_data: operation.ImageOp, 
    center: tuple[int, int],
  ) -> bool:
    """
    土砂マスク画像を用いて土砂領域かどうかを判別

    image_data: 画像等のデータ
    center: 該当領域の中心座標
    """
    # マスク画像より土砂の判別
    if (image_data.mask[center][0] == 0):
      return True
    else:
      return False






    # for i, region in enumerate(image_data.region):
    #   # TODO: 順番を考えることによって処理を減らせそう
    #   # TODO: 最初に4つの処理で共通に必要なデータを取得することでメモリ使用等を減らせそう
    #   # NOTE: detect_flowのように10方向で精度評価＋できればベクトル量（流出距離も）ラベル画像の領域単位で，そこからどこに流れてそうか正解画像（矢印図）を作成
    #   # NOTE: できればオプティカルフローやPIV解析,3D-GIV解析，特徴量追跡等も追加する

    #   # cy, cx   = region["cy"], region["cx"]
    #   # # 始点
    #   # cv2.circle(self.ortho, (cx, cy), 3, (0, 0, 255), thickness=5, lineType=cv2.LINE_8, shift=0)


    #     # 傾斜方向と隣接2方向の3方向に対しての隣接領域を取得
    #     # TODO: 距離が小さすぎる矢印を除去する
    #     # FIXME: 座標とかがおかしいので修正
    #     # FIXME: 終点が全部上端になってる？？

    #     labels = process.extract_neighbor(self, region)

    #     # 傾斜方向が上から下の領域を抽出
    #     labels = process.extract_downstream(self, region, labels)

    #     # 侵食と堆積の組み合わせの領域を抽出
    #     labels = process.extract_sediment(self, region, labels)

    #     # 矢印の描画
    #     # FIXME: 移動ベクトルのやじるしの傘部分の大きさ一定にしたい
    #     # tool.draw_vector(self, region, labels)
    #     tool.draw_vector(self, region, labels)

    #     # # 移動量・移動方向を保存
    #     # process.save_vector(self, region, sediment_labels)

    #     # if (i > 100):
    #     # 	cv2.imwrite("./outputs/map_v2.png", self.ortho)
    #     # 	return

    # # 土砂移動図の作成
    # cv2.imwrite("./outputs/map_v2_point.png", self.ortho)

    # return

