#
# ズレ量を計算して、位置を整えるプログラム例
#

import cv2
import math
import numpy as np
import numpy.linalg as LA

# 入出力ファイルは決め打ち
src_path = "./src_image1.bmp"
dest_path = "./dest_image1.bmp"

# ① 画像を読み込む
src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

# ② 画像の中心座標を求める
height,width  = src_img.shape[:2]
gy = height / 2
gx = width / 2
print("画像の中心：y={0},x={1}\n".format(gy, gx))

# ③ オブジェクトの重心を求める
object_g = np.array(np.where(src_img == 255)).mean(axis=1)
print("オブジェクトの中心座標：y={0}, x={1}\n".format(object_g[0], object_g[1]))

# ④ 重心のズレを補正する
dy = gy - object_g[0]
dx = gx - object_g[1]
print("中心座標とのズレ：y={0}, x={1}\n".format(dy, dx))

mat_g = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
affine_img_g = cv2.warpAffine(src_img, mat_g, (width,height))

# ⑤ 確度のズレを計算する
index_vector = np.array(np.where(affine_img_g == 255))
Cov = np.cov(index_vector)
eigne_value, eigen_vector = LA.eig(Cov)

mat_delta = np.array([[eigen_vector[0][0], eigen_vector[0][1], (1.0 - eigen_vector[0][0]) * gx - eigen_vector[0][1] * gy], 
 [eigen_vector[1][0], eigen_vector[1][1], (1.0 - eigen_vector[0][0])*gy + eigen_vector[0][1]*gx]], dtype=np.float32)
print("変換行列")
print(mat_delta)

affine_img_delta = cv2.warpAffine(affine_img_g, mat_delta, (width, height))
cv2.imwrite(dest_path, affine_img_delta)