### This python file will be used to convert actor faces
### to encodings in order to speed up the ID process



import tensorflow as tf
import imageio
import os
import facenet.src

## Get the path of the images that need to be encoded
IMDBpath = '/home/connor/Documents/CS230/CS230DeepActor/test_data/IMDB_Pics/'

## Get the path of the model that needs to be used
Modelpath = '/home/connor/Documents/CS230/CS230DeepActor/models/pretrained_facenet/20170512-110547/'

## Get the path of the encodings
Encodingpath = '/home/connor/Documents/CS230/CS230DeepActor/src/actor_encodings/'


## Get the file names of all the faces in IMDBpath and Encodings
IMDB_names = []
Encoding_names = []
for root, dirs, files in os.walk(Encodingpath):
    Encoding_names = files
for root, dirs, files in os.walk(IMDBpath):  ## This issue is .jpg vs the encoding file name
    # add only the files that have not been encoded already
    IMDB_names = [file if file not in Encoding_names else None for file in files]

a = 4
