### Use this py to train triplet loss

# Current thinking:
# 1. Load the pretrained model for making the embeddings
# 2. Make a new two layer FC NN that takes the embeddings and learns triplet loss

# Things to try:
# 1. Train using an online database
# 2.

import tensorflow as tf
import facenet.src.facenet as facenet
import facenet.contributed.face as face
import imageio
import pickle as pkl
import os
import numpy as np


class Encoding_Model:
    def __init__(self, model_path):
        self.sess = tf.Session()
        self.detect = face.Detection() # make a detector (using triple CNN)
        with self.sess.as_default():
            facenet.load_model(model_path)

    # either give a path to an image or the image and get out a face object
    def imagetoface(self, image = None, imagepath = None):
        if imagepath is not None:
            image = imageio.imread(imagepath)
        return self.detect.find_faces(image)

    # This function will generate all the embeddings from the images in the path and return a dictionary with the
    # image names as the keys
    # The file structure should be path -> folders/flattened_faces.jpg
    def generate_all_embeddings(self, path):
        all_embeddings = {}
        # step through all the actors
        for root, actors, _ in os.walk(path):
            for actor in actors:
                # step through all the faces
                for root2, _, faces in os.walk(root + actor):
                    for face in faces:
                        all_embeddings[face] = self.generate_embedding(self.imagetoface(root2 + face)) # get the embeddings
        # now return the embeddings
        return all_embeddings

    # This function will save the embeddings as a pickle
    def saveEmbeddings(self, pathtosave, embeddings, filename):
        pkl.dump(embeddings, open(pathtosave + filename, 'wb'))

    # This function will load the embeddings from the path
    def loadEmbeddings(self, pathtoload):
        return pkl.load(open(pathtoload, 'rb'))

    # input a face object and get the embeddings out
    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

# This class will be an NN to map the encodings to face comparrison
class Triplet_NN:
    def __init__(self):
        self.sess = tf.Session()


    def triplet_loss(self, anchor, positive, negative, alpha):
        anpos_sum = np.sum(np.square(np.subtract(vec1, vec2)))

    def choose_triplets(self, ):