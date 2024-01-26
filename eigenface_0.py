# 固有顔(Eigenface)を用いた顔認証シミュレーション

import os
import cv2
import numpy as np
from pylab import *

# 顔画像データベース
Database = 'att_faces'

# 被験者数
person = 40

# 学習用画像のid
training_ids = [1,2,3,4,5,6,7,8,9]
# テスト用画像のid
test_ids = [10]

# 学習用顔画像の枚数（1人あたり）
training_faces_count = len(training_ids)
# テスト用顔画像の枚数（1人あたり）
test_faces_count = len(test_ids)
# 学習用顔画像の全枚数
l = training_faces_count * person


# 画像サイズ（縦）
height = 112
# 画像サイズ（横）
width = 92
# 画素数
pixels = height * width

# 累積寄与率(cumulative contribution ratio)の閾値
ccr_th = 0.85

# 識別関数
def classify(path, mean_img_col, evectors, W):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_col = np.array(img, dtype='float64').flatten()
    img_col -= mean_img_col
    img_col = np.reshape(img_col, (pixels, 1))

    S = np.dot(np.transpose(evectors),img_col)
    diff = abs(W - S)
    norms = np.linalg.norm(diff, axis=0)

    # 最も類似した顔画像のidを求める
    closest_face_id = np.argmin(norms) 

    return closest_face_id

############################################################
print("学習開始")

# 各列が1枚の顔画像に対応する行列Lを定義
L = np.empty(shape=(pixels, l), dtype='float64')

cur_img = 0
for face_id in range(1, person+1):
    for training_id in training_ids:
        path = Database + "/" + "s" + str(face_id).zfill(2) +  "/" + str(training_id) + ".pgm"
        # 顔画像をグレイスケール画像として読み込み
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 2次元配列を1次元配列に変換
        img_col = np.array(img, dtype='float64').flatten()
        # 1次元配列をLに格納
        img_col = img_col.flatten()

        L[:,cur_img] = img_col
        cur_img += 1

# Lより平均画像を求める
print("L平均画像のサイズ={}".format(L.shape))
mean_img_col = np.mean(L,axis = 1)

# Lの各画像から平均画像を引く
for j in range(0, l):
     L[:,j] = L[:,j] - mean_img_col

# Lから共分散行列Cを求める
C = np.dot(np.transpose(L),L)
print("共分散行列Cのサイズ={}".format(C.shape))
# 固有値と固有ベクトルを求める（実対称行列用のeighを使用）
evalues, evectors = np.linalg.eig(C)

# 固有値と固有ベクトルを降順に並べ替える
sort_indices = np.argsort(evalues)[::-1]
evalues = evalues[sort_indices]
evectors = evectors[:,sort_indices]
print("固有値のサイズ={}".format(evalues.shape))
print("u固有ベクトル={}".format(evectors.shape))

# 累積寄与率が閾値を超えるのに必要な固有値の最小数を求める
evalues_sum = np.sum(evalues)
evalues_count = 0
ccr = 0.0
for evalue in evalues :
    evalues_count += 1
    ccr += evalue/evalues_sum
    if ccr > ccr_th :
        break
print("累積寄与率={}".format(ccr))
print("固有値の最小数={}".format(evalues_count))



evalues = evalues[:evalues_count]
evectors = np.dot(L,evectors[:,:evalues_count])
print("v固有ベクトル={}".format(evectors.shape))
norms = np.linalg.norm(evectors, axis=0)
evectors = evectors / norms

# 重み係数を求める
W = np.dot(np.transpose(evectors),L)
print("重み係数のサイズ={}".format(W.shape))

np.savetxt("evalues.dat", evalues)
np.savetxt("evectors.dat", evectors)
np.savetxt("weights.dat", W)

# ###########################################################
print("テスト開始")

test_count = test_faces_count * person
test_correct = 0

#lnはテスト画像の配列サイズ
ln = len(training_ids)
print("被験者id　識別結果")
for face_id in range(1, person+1):
    for test_id in test_ids:
        path = Database + "/" + "s" + str(face_id).zfill(2) +  "/" + str(test_id) + ".pgm"
        result_id = classify(path, mean_img_col, evectors, W)

        result = int(result_id / ln)  + 1 == face_id 


        print(face_id, result_id,result)
        if result == True:
            test_correct += 1

accuracy = test_correct / test_count
print(accuracy)
print("Correct: ", end="")
print(str(test_correct) + "/" + str(test_count) + " = ", end="")
print(str(accuracy) + " %")

# print(evectors[:,1].reshape(img.shape).shape)
# print(img.shape)
# 平均顔と固有顔の表示
figure()
gray()
subplot(2, 5, 1)
imshow(mean_img_col.reshape(height, width))
for i in range(9):
    subplot(2, 5, i+2)
    imshow(evectors[:,i].reshape(img.shape),cmap = "gray")

# show()
