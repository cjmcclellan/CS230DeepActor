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
import time
import random

pretrained_model = '/home/connor/Documents/CS230/CS230DeepActor/models/pretrained_facenet/20170512-110547'
data_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline'
embeddings_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Embeddings/1521394841.7290642/Embeddings_at_1521394841.7290642'

class Encoding_Model:
    def __init__(self, model_path=None):
        if model_path == None:
            return
        self.sess = tf.Session()
        self.detect = face.Detection() # make a detector (using triple CNN)
        with self.sess.as_default():
            facenet.load_model(model_path)

    # either give a path to an image or the image and get out a face object
    def imagetoface(self, image = None, imagepath = None):
        if imagepath is not None:
            image = imageio.imread(imagepath)
        face = self.detect.find_faces(image)
        return face[0] # return the only face in the image

    # This function will generate all the embeddings from the images in the path and return a dictionary with the
    # image names as the keys
    # The file structure should be path -> folders/flattened_faces.jpg
    def generate_all_embeddings(self, path):
        all_embeddings = {} # dictionary of a dictionary
        # step through all the actors
        for root, actors, _ in os.walk(path):
            for actor in actors:
                all_embeddings[actor] = {}
                # step through all the faces
                for root2, _, faces in os.walk(root +'/' + actor):
                    for face in faces:
                        all_embeddings[actor][face] = self.generate_embedding(self.imagetoface(imagepath=root2 + '/' + face)) # get the embeddings
        # now return the embeddings
        return all_embeddings

    # This function will save the embeddings as a pickle
    def saveEmbeddings(self, pathtosave, embeddings, filename):
        if not os.path.exists(pathtosave):
            os.makedirs(pathtosave)
        pkl.dump(embeddings, open(pathtosave + '/' + filename, 'wb'))

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
    def __init__(self, input_shape, margin, learning_rate):
        self.sess = tf.Session()
        self.anchor = tf.placeholder(tf.float32, shape=[None,input_shape], name='anchor')
        self.positive = tf.placeholder(tf.float32, shape=[None, input_shape], name='positive')
        self.negative = tf.placeholder(tf.float32, shape=[None, input_shape], name='negative')

        self.anchorOut = self.NN_architecutre(self.anchor)
        self.positiveOut = self.NN_architecutre(self.positive)
        self.negativeOut = self.NN_architecutre(self.negative)
        self.loss = self.triplet_loss(self.anchorOut, self.positiveOut, self.negativeOut, margin)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=tf.Variable(batch_size))

    def global_initializer(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # This will compute the embeddings of the triplet model
    def get_all_triplet_embeddings(self, embeddings):
        triplet_embed = {} # record the embeddings
        for actor in embeddings.keys(): # look through all the actors
            triplet_embed[actor] = {}
            for face in embeddings[actor].items(): # look through the faces of each actor
                # get the embeddings of the actor's face
                triplet_embed[actor][face[0]] = self.sess.run(self.anchorOut,
                                                        feed_dict={'anchor:0': np.reshape(face[1], [1,face[1].shape[0]])})
        # return the computed embeddings
        return triplet_embed

    # compute the L2 norm
    def L2_norm(self, vec1, vec2):
        return np.sum(np.square(np.subtract(vec1, vec2)))

    # This will test the model
    def test_model(self, embeddings):
        final_results = []
        prediction_results  = []
        triplet_embed = self.get_all_triplet_embeddings(embeddings)
        for actor in triplet_embed.keys(): # look at all the actors
            actor_results = []
            actor_faces = list(triplet_embed[actor].items()) # get a list of the actors faces
            actor_face = actor_faces[0]
            for character in triplet_embed.keys(): # look at all the characters
                character_faces = list(triplet_embed[character].items()) # get the character's faces
                character_face = character_faces[0] # just take the first character face
                if character_face[0] == actor_face[0]:  # ensure that the character and actor face are not the same
                    character_face = character_faces[1] # if so, take the second face
                actor_results.append([character, self.L2_norm(character_face[1], actor_face[1])])  # record the results, be sure to only compare the embeddings
            if actor == 'Alfred--Michael_Caine':
                save_actorembed = actor_face
            # after comparing with the other characters, save the best 5
            final_results.append(sorted(actor_results, key=lambda x:x[1])[0:4])

            # check if the result was correct
            if final_results[-1][0][0] == actor:
                prediction_results.append(1)
            else:
                prediction_results.append(0)
        return (prediction_results, final_results, save_actorembed)





    # compare actor will all other actors and return i_most most likely
    #def compare_L2_norm(self, ):

    # Create the NN architecture
    def NN_architecutre(self, placeholder):
        dense1 = tf.layers.dense(inputs=placeholder, units=layersSize[0], activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(0))
        output = tf.layers.dense(inputs=dense1, units=layersSize[1], activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(0))
        return output

    # preform the triplet loss
    def triplet_loss(self, anchor, positive, negative, alpha):
        with tf.name_scope('triplet_loss'):
            pos_dis = tf.reduce_sum(tf.square(anchor - positive), 1) # compute postive distance
            neg_dis = tf.reduce_sum(tf.square(anchor - negative), 1) # compute negative distance
            loss = tf.maximum(0.0, alpha + pos_dis - neg_dis) # take the max of 0 or the distance to prevent negative loss
            loss = tf.reduce_mean(loss) # take the mean loss
            return loss

    # This function can become more complicated, but for now just choose randomly
    def choose_triplets(self, embeddings):
        i_anchor = random.randint(0, len(embeddings) - 1) # choose a random actor
        i_negative = i_anchor
        while i_negative == i_anchor: # choose a negative actor that is not the same as the anchor
            i_negative = random.randint(0, len(embeddings) - 1 )
        # Get the keys of the embeddings and the actor faces
        actors = list(embeddings.keys())
        anchor_actor = actors[i_anchor]
        negative_actor = actors[i_negative]
        anchor_faces = embeddings[anchor_actor]
        negative_faces = embeddings[negative_actor]
        # now randomly choose the faces
        i_anchor_face = random.randint(0, len(anchor_faces) - 1)
        i_positive_face = i_anchor_face
        while i_positive_face == i_anchor_face: # ensure that the positive and anchor are not the same face
            i_positive_face = random.randint(0, len(anchor_faces) - 1)
        i_negative_face = random.randint(0, len(negative_faces) - 1)
        # Now grap the face embeddings (the second element in the items)
        anchor_face = list(anchor_faces.items())[i_anchor_face][1]
        positive_face = list(anchor_faces.items())[i_anchor_face][1]
        negative_face = list(negative_faces.items())[i_negative_face][1]
        return (anchor_actor, anchor_face), (anchor_actor, positive_face), (negative_actor, negative_face)

# train the NN model
# Some Hyperparameters
layersSize = [1028, 1028]
input_layer_size = 128
margin = 1
epochs = 1000
batch_size = 20
learning_rate = 1e-5

# def main():

# first get all the embeddings and save them
# encoder = Encoding_Model(pretrained_model)  # get the encoder
# embeddings = encoder.generate_all_embeddings(data_path) # now get all the embeddings
# current_time = str(time.time())
# encoder.saveEmbeddings(embeddings_path + '/' + current_time, embeddings, 'Embeddings_at_' + current_time)

# next, load the embeddings
encoder = Encoding_Model()
embeddings = encoder.loadEmbeddings(embeddings_path)

# Now train the NN
model = Triplet_NN(input_layer_size, margin, learning_rate)  # create the NN
model.global_initializer() # initialize the variables
test_results = []
test_results.append(model.test_model(embeddings))
accuracy = float(sum(test_results[-1][0]))/float(len(test_results[-1][0]))
print('the accuracy is :')
print(accuracy)
print('')

## Notes:
## I don't think my network is right, i need to make one network and have placeholders for the loss
## I can then recompute the loss each epoch
for i_epoch in range(epochs):

    # clear the past minibatches
    anchor_minibatch = []
    positive_minibatch = []
    negative_minibatch = []

    # get the anchor, positive and negative encodings.
    for i_batch in range(batch_size):
        anchor, positive, negative = model.choose_triplets(embeddings)
        # appnend the new triplet to the minibatches (just take the embedding which is the second element in the triplet)
        anchor_minibatch.append(anchor[1])
        positive_minibatch.append(positive[1])
        negative_minibatch.append(negative[1])

    anchor_minibatch = np.array(anchor_minibatch)
    positive_minibatch = np.array(positive_minibatch)
    negative_minibatch = np.array(negative_minibatch)
    feed_dict = {'anchor:0': anchor_minibatch, 'positive:0': positive_minibatch, 'negative:0': negative_minibatch}
    # now run the model with the optimizer and the loss
    _, loss = model.sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
    print(loss)

    # do a test of the model
    if i_epoch%5 == 0:
        test_results.append(model.test_model(embeddings))
        accuracy = float(sum(test_results[-1][0]))/float(len(test_results[-1][0]))
        print('the accuracy is :')
        print(accuracy)
        print('')






