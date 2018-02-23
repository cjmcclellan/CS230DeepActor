# Import face detection file & other facenet files
import sys
sys.path.append('../')
import facenet.contributed.face as face

# Import supporting packages
import numpy as np
import tensorflow as tf
import imageio
from matplotlib import pyplot as plt
import matplotlib.patches as patches

test_path = '../test_data/FaceDetect/WiderDataset'
test_img_orig = imageio.imread(test_path + '/1.jpg')
plt.imshow(test_img_orig)
plt.show()
test_img = test_img_orig
print(test_img.shape)

# Create face detector
detector = face.Detection()
detector.minsize = 10
detector.face_crop_margin = 16
faces = detector.find_faces(test_img)

fig, ax = plt.subplots(1)
ax.imshow(test_img)
for i,f in enumerate(faces):
    bb = f.bounding_box
    xybb = (bb[0], bb[1])
    width = bb[2]-bb[0]
    height = bb[3]-bb[1]
    rect = patches.Rectangle(xybb,width, height, linewidth=1, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    for j in range(5):
        circ = patches.Circle((f.points[0][j], f.points[1][j]), radius=2, facecolor='lime')
        ax.add_patch(circ)

plt.show()
