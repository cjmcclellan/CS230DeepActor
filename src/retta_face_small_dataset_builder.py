## Goal is to take the small dataset of Nick Offerman images and extract his face,
# saving the output faces to a different folder. After, this set will run through the align_dataset_mtcnn
# code to prepare the data for a classifier training


# Import face detection file & other facenet files
import sys
sys.path.append('../')
import facenet.contributed.face as face

# Import supporting packages
import numpy as np
import tensorflow as tf
import imageio
from matplotlib import pyplot as plt

img_path = '../train_data/FaceID/Parks_Rec/retta_raw/retta'
output_path = '../train_data/FaceID/Parks_Rec/face/retta_face/retta_face'
test_img_vec = list()
for i in range(10):  # number of images in retta_raw
    test_img_vec.append(imageio.imread(img_path + '{}.jpg'.format(i+1)))
    # Create face detector
    detector = face.Detection()
    detector.minsize = 10
    detector.face_crop_margin = 16
    faces = detector.find_faces(test_img_vec[i])
    for j, curr_face in enumerate(faces):
        op = output_path
        if len(faces) > 1:
            op = op + '_{}_'.format(j)
        imageio.imwrite(op + '{}.png'.format(i+1), curr_face.image, format='png')

