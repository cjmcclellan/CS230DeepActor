# This program will execute the triplet loss model
# and compare possible faces with each other

#import facenet.src.train_tripletloss as trtrip
import facenet.contributed.face as face
import facenet.src.facenet as facenet
import tensorflow as tf
import imageio
import numpy as np
import os

triplet_model_path = '/home/connor/Documents/CS230/CS230DeepActor/models/triplet_loss/20180315-213418/'
Kylo_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline/Kylo_Ren--Adam_Driver/Kylo_Ren--Adam_Driver_1'
Gal_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline/Diana--Gal_Gadot/Diana--Gal_Gadot_0'
actors_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline/'


## Copy of the encode class from face.py in Facenet
class Encoder:
    def __init__(self, model_path):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(model_path)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

## My new class for triplet identifier
class TripRecognition:
    def __init__(self, model_path):
        self.detect = face.Detection()
        self.encoder = Encoder(model_path)

    # takes in an image and outputs the embeddings of that image
    def gen_embedding(self, image):
        face = self.detect.find_faces(image)
        return self.encoder.generate_embedding(face[0])


def L2_Loss(vec1, vec2):
    return np.sum(np.square(np.subtract(vec1, vec2)))

def tripLoss(vec1, vec2):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(vec1, vec2)), 1)
        loss = tf.reduce_mean(tf.maximum(pos_dist, 0.0), 0)

    return loss

# The actual program

# make the recognizer
tripletModel = TripRecognition(triplet_model_path)

# test the model
Kylo_face = imageio.imread(Kylo_path)
Kylo_embeddings = tripletModel.gen_embedding(Kylo_face)

Gal_face = imageio.imread(Gal_path)
Gal_embeddings = tripletModel.gen_embedding(Gal_face)

truth_embeddings = Gal_embeddings

# Now loop through all the known characters to compare the embeddings
results = [] # keep track of all the loss and characters
for rootdir, characters, files in os.walk(actors_path):
    for character in characters:
        for face_root, _, faces in os.walk(rootdir + '/' + character):
            new_character_image = imageio.imread(face_root + '/' + faces[0])
            new_character_embeddings = tripletModel.gen_embedding(new_character_image)
            results.append([character, L2_Loss(truth_embeddings, new_character_embeddings)])

best_fit = sorted(results, key=lambda x: x[1])
print(best_fit)
a= 5


