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