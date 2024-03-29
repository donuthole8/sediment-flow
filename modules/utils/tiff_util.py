import cv2
import numpy as np

from osgeo import gdal
# from osgeo import osr



def load_tif(path: str) -> np.ndarray:
  """ tifファイルの読み込み

  Args:
      path (str): tifファイルのファイルパス

  Returns:
      np.ndarray: tif画像
  """
  src = gdal.Open(path)

  xsize = src.RasterXSize
  ysize = src.RasterYSize
  band = src.RasterCount

  # 第1-4バンド
  b1 = src.GetRasterBand(1).ReadAsArray()
  try:
    b2 = src.GetRasterBand(2).ReadAsArray()
    b3 = src.GetRasterBand(3).ReadAsArray()
    b4 = src.GetRasterBand(4).ReadAsArray()
  except:
    pass

  # データタイプ番号
  # dtid = src.GetRasterBand(1).DataType

  # 出力画像
  # output = gdal.GetDriverByName('GTiff').Create('./results/geo.tif', xsize, ysize, band, dtid)

  # 座標系指定
  # output.SetGeoTransform(src.GetGeoTransform())

  # 空間情報を結合
  # output.SetProjection(src.GetProjection())
  # output.GetRasterBand(1).WriteArray(b1)
  # output.GetRasterBand(2).WriteArray(b2)
  # output.GetRasterBand(3).WriteArray(b3)
  # output.GetRasterBand(4).WriteArray(b4)
  # output.FlushCache()
  # output = None

  try:
    return cv2.merge((b1,b2,b3))
  except:
    print("1 band")
    return b1


# def write_tiffile(res,read_file,write_file):
#   """
#   tifファイルの保存
#   """
#   # Nodataの消去
#   # idx = np.where((res == 0) or (res == 12623735.))
#   # res[idx] = -1

#   # tif画像テンプレ読み込み
#   src = gdal.Open(read_file)

#   # 画像サイズ・バンド数
#   xsize = src.RasterXSize
#   ysize = src.RasterYSize
#   band = src.RasterCount

#   # 第1-4バンド
#   b3,b2,b1 = res,res,res
#   b4 = src.GetRasterBand(4).ReadAsArray()

#   # データタイプ番号（32-bit float）
#   dtid = 6

#   # 出力画像
#   output = gdal.GetDriverByName('GTiff').Create(write_file, xsize, ysize, band, dtid)

#   # 座標系指定
#   output.SetGeoTransform(src.GetGeoTransform())

#   # 空間情報を結合
#   output.SetProjection(src.GetProjection())
#   output.GetRasterBand(1).WriteArray(b1)
#   output.GetRasterBand(2).WriteArray(b2)
#   output.GetRasterBand(3).WriteArray(b3)
#   output.GetRasterBand(4).WriteArray(b4)
#   output.FlushCache()


def save_tif(image: np.ndarray, path: str) -> None:
  """ tiff画像を保存

  Args:
      image (np.ndarray): tifデータ
      path (str): 書き込みパス
  """
  cv2.imwrite("./outputs/" + path, image)


def _save_tif(data: np.ndarray, load_path: str, save_path: str) -> None:
  """ tifファイルの書き込み

  Args:
      data (np.ndarray): 書き込みデータ 
      load_path (str): 入力tifファイルのパス
      save_path (str): 出力tifファイルのパス
  """
  # 入出力パス
  # load_path = "./inputs/"  + load_path
  # save_path = "./outputs/" + save_path

  # tif画像テンプレ読み込み
  src = gdal.Open(load_path)

  # 画像サイズ・バンド数
  xsize = src.RasterXSize
  ysize = src.RasterYSize
  band = src.RasterCount

  # 第1-4バンド
  try:
    b3, b2, b1 = cv2.split(data)
  except Exception as e:
    b3, b2, b1 = data, data, data

  try:
    b4 = src.GetRasterBand(4).ReadAsArray()
  except:
    pass

  # データタイプ番号（32-bit float）
  dtid = 6
  
  # 出力画像
  output = gdal.GetDriverByName('GTiff').Create(save_path, xsize, ysize, band, dtid)

  # 座標系指定
  output.SetGeoTransform(src.GetGeoTransform())
  
  # # 空間参照情報
  # srs = osr.SpatialReference()
  # # 空間情報を結合
  # output.SetProjection(srs.ExportToWkt())

  # 空間情報を結合
  output.SetProjection(src.GetProjection())
  try:
    output.GetRasterBand(1).WriteArray(b1)
    output.GetRasterBand(2).WriteArray(b2)
    output.GetRasterBand(3).WriteArray(b3)
    output.GetRasterBand(4).WriteArray(b4)
  except Exception as e:
    output.GetRasterBand(1).WriteArray(b1)
  output.FlushCache()


def get_band4(path: str) -> np.ndarray:
  """ 第4バンドデータを取得

  Args:
      path (str): tifファイルのパス

  Returns:
      np.ndarray: 第4バンドデータ
  """
  src = gdal.Open(path)

  # 第4バンド
  return src.GetRasterBand(4).ReadAsArray()


def set_band(band, tif):
  """
  バンドデータを変更
  """
  pass