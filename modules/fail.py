class sa():
	def a():





		# # 最大で(メッシュサイズ*√2)分だけループを回す
		# for i in range(self.mesh_size * (2 ** 0.5)):
		# 	print()
		# 	print(i)

		# 	# 注目画素の標高値を取得
		# 	target_height = image_data.dsm_uav[center]

		# 	while (1):
		# 		# 注目画素の傾斜方向の画素と隣接2画素の傾斜方向を取得
		# 		directions = self.get_directions(image_data.degree[center])

		# 		# 注目画素の傾斜方向の画素と隣接2画素の標高値を取得
		# 		heights = self.get_heights(image_data, directions, center)

		# 		print("dir", directions)
		# 		print("hei", heights)

		# 		# 注目画素の標高値と傾斜方向の画素値を比較
		# 		for 

		# 		if ():

		print("center   ::", center)
		if (center == (1050, 50)):

			# 土砂候補画素の座標
			y_coord, x_coord = center
			sediment_coords = [(y_coord, x_coord)]

			temp_coords = [(y_coord, x_coord)]


			print()
			print()
			print()
			print()
			print()
			print("center   ::", center)
			
			# 追跡土砂がなくなるまでループを回す
			# for temp_coord in temp_coords:
			while (len(temp_coords) > 0):

				# 注目画素の座標を取得
				y_coord, x_coord = temp_coords[0]

				# 一時配列から注目座標を削除
				temp_coords.pop(0)

				# 注目画素の標高値を取得
				target_height = image_data.dsm_uav[y_coord, x_coord]

				# 座標群に入っていなかったら処理を終了
				if (not (y_coord, x_coord) in coords):
					print("break")
					break

				# 注目画素の傾斜方向の画素と隣接2画素の傾斜方向を取得
				directions = self.get_directions(image_data.degree[center])

				# 注目画素の傾斜方向の画素と隣接2画素の標高値を取得
				heights = self.get_heights(image_data, directions, center)

				print()
				print("dir", directions)
				print("t-hei", target_height)
				print("hei", heights)

				# 注目画素の標高値と傾斜方向の画素値を比較
				for direction, height in zip(directions, heights):
					print("aaaaaaaaaaa")
					if (height <= target_height):
						temp_coords.append((
							y_coord + direction[0], 
							x_coord + direction[1]
						))
						sediment_coords.append((
							y_coord + direction[0], 
							x_coord + direction[1]
						))
						print("temp", temp_coords)
						print("temp", len(temp_coords))

				
				print("sediment::", sediment_coords)

			# # 矢印を描画
			# cv2.arrowedLine(
			# 	img=image_data.ortho,     	# 画像
			# 	pt1=(center[1], center[0]), # 始点
			# 	pt2=(sediment_coords[0][1], sediment_coords[0][0]),			# 終点
			# 	color=(0, 0, 255),  				# 色			
			# 	thickness=5,        				# 太さ
			# )
				

					

			


		# # directions = get_directions(self.degree[center])

		# # 傾斜方向と隣接2方向の隣接領域を取得
		# neighbor_region_coords = []

		# for direction in directions:
		# 	# 輪郭の一番端座標からDIRECTION[i]の方向の座標を取得
		# 	neighbor_coordinate = get_neighbor_coordinate(
		# 		direction, 
		# 		contour_coordinates, 
		# 		(region["cy"], region["cx"])
		# 	)
		# 	neighbor_region_coords.append(neighbor_coordinate)

		# return neighbor_region_coords


	def get_directions(self, deg: float) -> tuple[tuple]:
		"""
		傾斜方向を画素インデックスに変換し傾斜方向と隣接2方向を取得

		deg: 角度
		"""
		# 注目画素からの移動画素
		if   (math.isnan(deg)):
			# NOTE: 返り値違うかも
			return np.nan, np.nan
		elif (deg > 337.5) or  (deg <= 22.5):
			return self.DIRECTION[7], self.DIRECTION[0], self.DIRECTION[1]
		elif (deg > 22.5)  and (deg <= 67.5):
			return self.DIRECTION[0], self.DIRECTION[1], self.DIRECTION[2]
		elif (deg > 67.5)  and (deg <= 112.5):
			return self.DIRECTION[1], self.DIRECTION[2], self.DIRECTION[3]
		elif (deg > 112.5) and (deg <= 157.5):
			return self.DIRECTION[2], self.DIRECTION[3], self.DIRECTION[4]
		elif (deg > 157.5) and (deg <= 202.5):
			return self.DIRECTION[3], self.DIRECTION[4], self.DIRECTION[5]
		elif (deg > 202.5) and (deg <= 247.5):
			return self.DIRECTION[4], self.DIRECTION[5], self.DIRECTION[6]
		elif (deg > 247.5) and (deg <= 292.5):
			return self.DIRECTION[5], self.DIRECTION[6], self.DIRECTION[7]
		elif (deg > 292.5) and (deg <= 337.5):
			return self.DIRECTION[6], self.DIRECTION[7], self.DIRECTION[0]


	def get_heights(
		self, 
		image_data, 
		directions: tuple[tuple],
		coord: tuple[int, int]
	) -> tuple[int, int, int]:
		"""
		注目座標からの標高値を取得

		image_data: 画像等のデータ
		directions: 注目座標からの傾斜方向
		coord: 注目座標
		"""
		heights = []

		for direction in directions:
			# 注目画素からの傾斜方向の標高値を取得
			heights.append(image_data.dsm_uav[
				coord[0] + direction[0], 
				coord[1] + direction[1]
			])

		return heights
