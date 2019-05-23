import cv2
import os
from PIL import Image
import numpy as np

# ...................................提取通道.....................................
# def pc_trainimage():
#     train_pic = []
#     path = './DRIVE/training/images/'
#     for pic in os.listdir(path):
#         train_pic.append(pic)
#     print(train_pic)
#     for i in range(len(train_pic)):
#         pic = cv2.imread(path + train_pic[i])
#         a = pic[:, :, 0]
#         b = pic[:, :, 1]
#         c = pic[:, :, 2]
#         cv2.imwrite(path + str(i+21) + '_training.tif', a)
#         cv2.imwrite(path + str(i + 41) + '_training.tif', b)
#         cv2.imwrite(path + str(i + 61) + '_training.tif', c)
#
# pc_trainimage()

# ..............................改名.....................................................
# path = './DRIVE/training/1st_manual/'
#
# # 获取该目录下所有文件，存入列表中
# f = os.listdir(path)
#
# n = 0
# for i in f:
#     # 设置旧文件名（就是路径+文件名）
#     oldname = path + f[n]
#
#     # 设置新文件名
#     newname = path + str(n + 41) + '_manual1.gif'
#
#     # 用os模块中的rename方法对文件改名
#     os.rename(oldname, newname)
#     print(oldname, '======>', newname)
#
#     n += 1

a = cv2.imread('a.tif')
print(a.shape)
anp = np.array(a)
print(anp.shape)
an = anp[:, :, 0]
print(an.shape)