
# coding: utf-8

# # Face Recognition
# This module works to detect faces using the 'detect_face.py' file and pre-trained models of facenet
# https://github.com/davidsandberg/facenet <br><br>
# Goals:
# 1. Load facenet and supporting modules without error
# 2. Load images & Prepare
# 3. Load pretrained weights to a network
# 4. Run face detection on a few test images

# ## 1. Imports


# Import face detection file & other facenet files
import sys
sys.path.append('../')
import facenet.src.align.detect_face as df


# Import supporting packages
import numpy as np
import tensorflow as tf
import imageio
from matplotlib import pyplot as plt


# ## 2. Load Images & Prepare


# Filepaths for weight files
det_path = '../facenet/src/align/'
pnet_path = det_path + 'det1.npy'
rnet_path = det_path + 'det2.npy'
onet_path = det_path + 'det3.npy'


test_path = '../test_data/FaceDetect/WiderDataset'
test_img_orig = imageio.imread(test_path + '/4.jpg')
plt.imshow(test_img_orig)
plt.show()
test_img = test_img_orig/255.
print(test_img.shape)


# ## 3. Loading Pretrained Face Detection Network


tf.reset_default_graph()
sess = tf.Session()

pnet, rnet, onet = df.create_mtcnn(sess, det_path)


# Not sure how to set these parameters
threshold = [0.5, 0.5, 0.3]
factor = 0.79
minsize = 10
boxes, points = df.detect_face(test_img, minsize, pnet, rnet, onet, threshold, factor)


print(boxes)
print(points)

