# Finds the faces and flattens them from the netflix screenshots

import os
import subprocess
import sys
sys.path.append('../')
import facenet.contributed.face as face
from imageio import imread, imwrite
from os import listdir
from os.path import isfile, join
from shutil import copy

test_dir = os.path.abspath('../test_data/ridiculous6/')
screenshot_dir = os.path.join(test_dir, 'screenshot')
raw_dir = os.path.join(test_dir, 'raw')
face_dir = os.path.join(test_dir, 'face')
flat_dir = os.path.join(test_dir, 'flattened')

# Create directories for other outputs
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

if not os.path.exists(face_dir):
    os.makedirs(face_dir)

if not os.path.exists(flat_dir):
    os.makedirs(flat_dir)

pics = [f for f in listdir(screenshot_dir) if isfile(join(screenshot_dir, f))]

for i, pic in enumerate(pics):
    file = os.path.join(screenshot_dir, pic)
    try:
        curr_img = imread(file)
    except:
        # failed to read image
        # sometimes this happens because the correct file type is not returned
        print('Image Read Failed.')
        continue

    detector = face.Detection()
    detector.minsize = 10
    detector.face_crop_margin = 16
    try:
        faces = detector.find_faces(curr_img)
    except:
        # Something in detector failed. Move on to next image
        print('Skipped', pic, ' - Something went wrong finding faces')

    if len(faces) == 0:
        print('Skipped', pic, ' - No faces found')
    else:
        copy(file, os.path.join(raw_dir, pic))
        for j, curr_face in enumerate(faces):
            imwrite(os.path.join(face_dir, '{}_{}-{}.jpg'.format(pic.split('.')[0], i + 1, j)), curr_face.image, format='jpg')

print('Flattening faces')
# Flatten faces
pwd = os.getcwd()
align_path = '../facenet/src/align/'
try:
    os.chdir(align_path)
    subprocess.check_output(
        ['python', '-W', 'ignore', 'align_dataset_mtcnn.py', face_dir, flat_dir])
    os.chdir(pwd)
except:
    os.chdir(pwd)


