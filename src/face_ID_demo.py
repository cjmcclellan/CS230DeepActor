# Import face detection file & other facenet files
# Import face detection file & other facenet files
import sys
sys.path.append('../')
# import facenet.src.align
import facenet.contributed.face as face

# Import supporting packages

import numpy as np
import tensorflow as tf
import imageio
from matplotlib import pyplot as plt

im_id_path = '../test_data/FaceDetect/Parks_Rec/Parks_Rec_Rea_Ron.JPG'
image = imageio.imread(im_id_path)

#  Import and display the test image
fig_input = plt.figure()
plt.imshow(image)
plt.tight_layout()
plt.suptitle('Ritta and Nick')
plt.subplots_adjust(top=0.90)
plt.show()

# This recognizer has been trained to recognize Nick Offerman and Rhetta
recognizer = face.Recognition()

# Get Faces from the test image
detector = face.Detection()
detector.minsize = 10
detector.face_crop_margin = 16
faces = detector.find_faces(image)
print(len(faces))

# Show that there is no ID for the faces
fig_input = plt.figure()
plt.suptitle('Face ID Input')
fig_input.add_subplot(1,2,1)
plt.imshow(faces[0].image)
plt.title('Saved Name For Ron: ' + str(faces[0].name))
fig_input.add_subplot(1,2,2)
plt.imshow(faces[1].image)
plt.title('Saved Name For Retta: ' + str(faces[1].name))
plt.tight_layout()

#
plt.subplots_adjust(top=0.90)
plt.savefig('face_id_input.png')
plt.show()
nick = faces[0].image
retta = faces[1].image
# Check that there is no name information
print('Saved Name For Ron: ' + str(faces[0].name))
print('Saved Name For Retta: ' + str(faces[1].name))

fig = plt.figure()
# Try to recognize Ron
id_1 = recognizer.identify(nick)
fig.add_subplot(1,2,1)
plt.imshow(id_1[0].image)
plt.tight_layout()
plt.title('ID Output: ' + id_1[0].name)

# Try to recognize Retta
id_2 = recognizer.identify(retta)
fig.add_subplot(1,2,2)
plt.imshow(id_2[0].image)
plt.tight_layout()
plt.title('ID Output: ' + id_2[0].name)
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.suptitle('Face ID Output')
plt.savefig('face_id_output.png')
plt.show()
