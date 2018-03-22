### Use this py to train triplet loss

# Current thinking:
# 1. Load the pretrained model for making the embeddings
# 2. Make a new two layer FC NN that takes the embeddings and learns triplet loss

# Things to try:
# 1. Train using an online database
# 2.

import tensorflow as tf
import torch
from torch.autograd import Variable
import facenet.src.facenet as facenet
import facenet.contributed.face as face
import imageio
import pickle as pkl
import os
import numpy as np
import time
import random
import matplotlib.pyplot as pyplt

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
    def __init__(self, input_shape, margin, learning_rate):#, weight_decay):
        self.model = self.NN_architecutre(input_shape)
        #self.loss_fun = torch.nn.MSELoss()
        self.loss_fun = torch.nn.TripletMarginLoss(margin=margin)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)#, weight_decay=weight_decay)

    def weight_initializer(self):
        torch.nn.init.xavier_normal(self.model[1].weight.data)
        torch.nn.init.xavier_normal(self.model[3].weight.data)
        # torch.nn.init.xavier_normal(self.model[5].weight.data)


    # This will compute the embeddings of the triplet model
    def get_all_triplet_embeddings(self, embeddings):
        triplet_embed = {} # record the embeddings
        self.model.train(False) # turn training off to run one example
        for actor in embeddings.keys(): # look through all the actors
            triplet_embed[actor] = {}
            for face in embeddings[actor].items(): # look through the faces of each actor
                # get the embeddings of the actor's face
                testing = np.reshape(face[1], [1, face[1].shape[0]])

                triplet_embed[actor][face[0]] = self.model(Variable(torch.Tensor(testing))).data.numpy()[0]

        # return the computed embeddings
        self.model.train(True) #turn training back on
        return triplet_embed

    # compute the L2 norm
    def L2_norm(self, vec1, vec2):
        return np.sum(np.square(np.subtract(vec1, vec2)))

    # This will test the model
    def test_model(self, embeddings):
        final_results = []
        prediction_results  = []
        triplet_embed = self.get_all_triplet_embeddings(embeddings) # for the anchor
        #triplet_embed_pos = self.get_all_triplet_embeddings(embeddings, 'anchor') # for the positive
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



    # Create the NN architecture
    def NN_architecutre(self, input_size):
        model_triplet = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_size),
            torch.nn.Linear(input_layer_size, layersSize[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(layersSize[0], layersSize[1]),
            torch.nn.ReLU()
        )

        return model_triplet

    # preform the triplet loss
    # def triplet_loss(self, anchor, positive, negative, alpha):
    #     with tf.name_scope('triplet_loss'):
    #         pos_dis = tf.reduce_sum(tf.square(anchor - positive), 1) # compute postive distance
    #         neg_dis = tf.reduce_sum(tf.square(anchor - negative), 1) # compute negative distance
    #         loss = tf.maximum(0.0, alpha + pos_dis - neg_dis) # take the max of 0 or the distance to prevent negative loss
    #         #loss = (alpha + pos_dis - neg_dis)  # take the max of 0 or the distance to prevent negative loss
    #         loss = tf.reduce_mean(loss) # take the mean loss
    #         return loss

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
        positive_face = list(anchor_faces.items())[i_positive_face][1]
        negative_face = list(negative_faces.items())[i_negative_face][1]
        return (anchor_actor, anchor_face), (anchor_actor, positive_face), (negative_actor, negative_face)


##########################
## Build the NN model ##
##########################


# train the NN model
# Some Hyperparameters
layersSize = [128, 256]
input_layer_size = 128
margin = 5.0
epochs = 1000
batch_size = 20
learning_rate = 1e-2
weight_decay = 0.1


# model_triplet = torch.nn.Sequential(
#     torch.nn.BatchNorm1d(input_layer_size),
#     torch.nn.Linear(input_layer_size, layersSize[0]),
#     torch.nn.ReLU(),
#     torch.nn.Linear(layersSize[0], layersSize[1]),
#     torch.nn.ReLU()
# )

# loss_fun = torch.nn.TripletMarginLoss(margin=margin)
# optimizer = torch.optim.Adam(model_triplet.parameters(), lr=learning_rate, weight_decay=weight_decay)

### Define the variables ###


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
model = Triplet_NN(input_layer_size, margin, learning_rate)#, weight_decay)  # create the NN
model.weight_initializer() # initialize the variables
# loss_fun = torch.nn.TripletMarginLoss(margin=margin)
# optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

test_results = []
loss_vec = []
#test_results.append(model.test_model(embeddings))
#accuracy = float(sum(test_results[-1][0]))/float(len(test_results[-1][0]))
# print('the accuracy is :')
# print(accuracy)
# print('')
# print(tf.all_variables())
#print(tf.get_variable('fully_connected_1/weights:0'))

## Notes:
## I don't think my network is right, i need to make one network and have placeholders for the loss
## I can then recompute the loss each epoch

for i_epoch in range(epochs):
    model.model.train()
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
        # save the first ones for comparing later
        # if i_batch == 0:
        #     save_inputan = anchor[1]
        #     save_inputpos = positive[1]
        #     save_inputneg = negative[1]

    anchor_minibatch = Variable(torch.Tensor(np.array(anchor_minibatch)))
    positive_minibatch = Variable(torch.Tensor(np.array(positive_minibatch)))
    negative_minibatch = Variable(torch.Tensor(np.array(negative_minibatch)))
    # now run the model with the optimizer and the loss

    model_output_anchor = model.model(anchor_minibatch)
    model_output_positive = model.model(positive_minibatch)
    model_output_negative = model.model(negative_minibatch)

    # get the numpy data
    # anchor1 = model_output_anchor.data.numpy()[0]
    # if i_epoch == 0:# use this for comparing the change in outputs
    #     save_anchor1 = anchor1

    # positive1 = model_output_positive.data.numpy()[0]
    # negative1 = model_output_negative.data.numpy()[0]

    loss = model.loss_fun(model_output_anchor, model_output_positive, model_output_negative)
    model.model.zero_grad()
    loss.backward()
    model.optimizer.step()
    loss_vec.append(float(loss.data))
    print('Loss: ' + str(float(loss.data)))
    #print(model_output_anchor)

    # do a test of the model
    if i_epoch%10 == 0:
        # test_results.append(model.test_model(embeddings))
        # accuracy = float(sum(test_results[-1][0]))/float(len(test_results[-1][0]))
        # print('the accuracy is :')
        # print(accuracy)
        print('')
        # print(save_anchor1 - anchor1)
        # save_anchor1 = anchor1

# print(test_results[0][-1])
# print(test_results[-1][-1])

pyplt.plot(loss_vec)
pyplt.show()
# print(np.subtract(test_results[0][-1],test_results[-1][-1]))
#



