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
img_filenames = ['1', '3', '4', '6', '43', '48']
test_img_vec = list()
fig_pre = plt.figure()
figlab = 'abcdef'
detectors = list()  # List of detectors, one per image
faces_list = list()  # List of facelists from each image
for i, filename in enumerate(img_filenames):
    test_img_vec.append(imageio.imread(test_path + '/' + filename + '.jpg'))
    fig_pre.add_subplot(2, 3, i+1)
    plt.imshow(test_img_vec[i])
    plt.xlabel('({})'.format(figlab[i]))
    # Create face detector
    detectors.append(face.Detection())
    detectors[-1].minsize = 10
    detectors[-1].face_crop_margin = 16
    faces_list.append(detectors[-1].find_faces(test_img_vec[i]))

plt.tight_layout()
plt.suptitle('Example Images')
plt.subplots_adjust(top=0.90)
plt.savefig('example_input_combined.png')
plt.show()

# Separate figure for face bound box and points
fig, ax = plt.subplots(2, 3)
plt.suptitle('Example Images with Bounding Boxes and Points')
for i, test_img in enumerate(test_img_vec):
    faces = faces_list[i]
    curr_ax = ax[int(i/3)][i%3]
    curr_ax.imshow(test_img)
    curr_ax.set_xlabel('({})'.format(figlab[i]))
    for _, f in enumerate(faces):
        bb = f.bounding_box
        xybb = (bb[0], bb[1])
        width = bb[2]-bb[0]
        height = bb[3]-bb[1]
        rect = patches.Rectangle(xybb,width, height, linewidth=1, edgecolor='lime', facecolor='none')
        curr_ax.add_patch(rect)
        for j in range(5):
            circ = patches.Circle((f.points[0][j], f.points[1][j]), radius=2, facecolor='lime')
            curr_ax.add_patch(circ)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('example_output_combined.png')
plt.show()
