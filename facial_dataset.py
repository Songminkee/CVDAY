import cv2
import os
import numpy as np

def dataset_generate():
    A_path = 'C:/Users/tissu/Projects/datasets/test2017/'
    # B_path = 'C:/Users/tissu/Projects/datasets/FASSEG-repository-master/train_labels/'

    res_path = 'C:/Users/tissu/Projects/datasets/BWtoColor_sets/'

    A_list = os.listdir(A_path)
    # B_list = os.listdir(B_path)

    if not os.path.isdir(res_path):
        os.mkdir(res_path)

    for i in range(0, len(A_list)):
        A_src = cv2.imread(os.path.join(A_path, A_list[i]))
        B_src = cv2.imread(os.path.join(A_path, A_list[i]), cv2.IMREAD_GRAYSCALE)

        A_res = cv2.resize(A_src, dsize=(256, 256))
        B_res = cv2.resize(B_src, dsize=(256, 256))
        B_res = cv2.cvtColor(B_res, cv2.COLOR_GRAY2BGR)

        res = np.concatenate((A_res, B_res), axis=1)

        # res = cv2.hconcat([A_res, B_res])

        # cv2.imshow('dwd', res)
        # cv2.waitKey()

        cv2.imwrite(os.path.join(res_path, A_list[i]), res)

def input_generator():
    in_path = 'C:/Users/tissu/Projects/cv_proj/pix2pix-tensorflow/facial_test/joonggi.jpg'

    in_img = cv2.imread(in_path)

    in_resize = cv2.resize(in_img, dsize=(256, 256))

    black_img = np.zeros((256, 256, 3), np.uint8)

    res = cv2.hconcat([in_resize, black_img])

    cv2.imwrite(os.path.join("C:/Users/tissu/Projects/cv_proj/pix2pix-tensorflow/facial_test/", "res.png"), res)

dataset_generate()