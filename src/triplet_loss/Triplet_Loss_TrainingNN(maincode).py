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
import gender_guesser.detector as gender

# paths for saving data
#saved_data_path = '/home/connor/Dropbox/CS230/Data'

pretrained_model = '../../models/pretrained_facenet/20170512-110547'
data_path = '../../train_data/Triplet_Loss/Movie_Triplets/Baseline'

embeddings_path ='../../train_data/Triplet_Loss/Movie_Triplets/Embeddings/1521501363.9522028/Embeddings_at_1521501363.9522028'
#embeddings_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Embeddings/1521394841.7290642/Embeddings_at_1521394841.7290642'

# ridic 6 paths
test_data_path = '../../test_data/ridiculous6/r6_testset.pkl'
ridic_char_path = '../../train_data/Ridiclous_6/flattened/'
ridic_embeddings_path = '../../train_data/Ridiclous_6/embeddings/1521513956.7939253'
character_to_id = {'Chico--Terry_Crews': 0, 'Danny--Luke_Wilson': 1, 'Herm--Jorge_Garcia': 2, 'Lil_Pete--Taylor_Lautner': 3,
                   'Ramon--Rob_Schneider': 4, 'Tommy--Adam_Sandler': 5, 'Will_Patch--Will_Forte':6}


class Encoding_Model:
    def __init__(self, model_path=None):
        if model_path == None:
            return
        self.sess = tf.Session()
        self.detect = face.Detection()  # make a detector (using triple CNN)
        with self.sess.as_default():
            facenet.load_model(model_path)

    # either give a path to an image or the image and get out a face object
    def imagetoface(self, image=None, imagepath=None):
        if imagepath is not None:
            image = imageio.imread(imagepath)
        face = self.detect.find_faces(image)
        return face[0]  # return the only face in the image

    # This function will generate all the embeddings from the images in the path and return a dictionary with the
    # image names as the keys
    # The file structure should be path -> folders/flattened_faces.jpg
    def generate_all_embeddings(self, path):
        all_embeddings = {}  # dictionary of a dictionary
        # step through all the actors
        for root, actors, _ in os.walk(path):
            for actor in actors:
                all_embeddings[actor] = {}
                # step through all the faces
                for root2, _, faces in os.walk(root + '/' + actor):
                    for face in faces:
                        all_embeddings[actor][face] = self.generate_embedding(
                            self.imagetoface(imagepath=root2 + '/' + face))  # get the embeddings
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
    def __init__(self, model):
        self.model = model
        self.gender = gender.Detector()
        # self.model = self.NN_architecutre(input_shape)
        # self.loss_fun = torch.nn.MSELoss()
        # # self.loss_fun = torch.nn.TripletMarginLoss(margin=margin)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # def weight_initializer(self):
    #     torch.nn.init.xavier_normal(self.model[1].weight.data)
    #     torch.nn.init.xavier_normal(self.model[3].weight.data)
        # torch.nn.init.xavier_normal(self.model[5].weight.data)

    # This will compute the embeddings of the triplet model
    def get_all_triplet_embeddings(self, embeddings):
        triplet_embed = {}  # record the embeddings
        self.model.train(False)  # turn training off to run one example
        for actor in embeddings.keys():  # look through all the actors
            triplet_embed[actor] = {}
            for face in embeddings[actor].items():  # look through the faces of each actor
                # get the embeddings of the actor's face
                testing = np.reshape(face[1], [1, face[1].shape[0]])

                triplet_embed[actor][face[0]] = self.model(Variable(torch.Tensor(testing))).data.numpy()[0]

        # return the computed embeddings
        self.model.train(True)  # turn training back on
        return triplet_embed

    # compute the L2 norm
    def L2_norm(self, vec1, vec2):
        return np.sum(np.square(np.subtract(vec1, vec2)))

    # This will test the model
    def test_model(self, embeddings):
        final_results = []
        prediction_results = []
        triplet_embed = self.get_all_triplet_embeddings(embeddings)  # for the anchor
        # triplet_embed_pos = self.get_all_triplet_embeddings(embeddings, 'anchor') # for the positive
        for actor in triplet_embed.keys():  # look at all the actors
            actor_results = []
            actor_faces = list(triplet_embed[actor].items())  # get a list of the actors faces
            actor_face = actor_faces[0]
            for character in triplet_embed.keys():  # look at all the characters
                character_faces = list(triplet_embed[character].items())  # get the character's faces
                character_face = character_faces[0]  # just take the first character face
                if character_face[0] == actor_face[0]:  # ensure that the character and actor face are not the same
                    character_face = character_faces[1]  # if so, take the second face
                actor_results.append([character, self.L2_norm(character_face[1], actor_face[
                    1])])  # record the results, be sure to only compare the embeddings
            # if actor == 'Alfred--Michael_Caine':
            #     save_actorembed = actor_face
            # after comparing with the other characters, save the best 5
            final_results.append(sorted(actor_results, key=lambda x: x[1])[0:4])

            # check if the result was correct
            if final_results[-1][0][0] == actor:
                prediction_results.append(1)
            else:
                prediction_results.append(0)
        return (prediction_results, final_results)#, save_actorembed)


    def create_test_train_sets(self, embeddings, train_precent):
        num_embed = len(embeddings)
        test_num = num_embed*(1-train_precent)
        test_set = {}
        train_set = {}
        for items in embeddings.items():
            if len(test_set) < test_num:
                test_set[items[0]] = items[1]
            else:
                train_set[items[0]] = items[1]
        return train_set, test_set


    # This function can become more complicated, but for now just choose randomly
    def choose_triplets(self, embeddings):
        i_anchor = random.randint(0, len(embeddings) - 1)  # choose a random actor
        i_negative = i_anchor
        while i_negative == i_anchor:  # choose a negative actor that is not the same as the anchor
            i_negative = random.randint(0, len(embeddings) - 1)
        # Get the keys of the embeddings and the actor faces
        actors = list(embeddings.keys())
        anchor_actor = actors[i_anchor]
        negative_actor = actors[i_negative]
        anchor_faces = embeddings[anchor_actor]
        negative_faces = embeddings[negative_actor]
        # now randomly choose the faces
        i_anchor_face = random.randint(0, len(anchor_faces) - 1)
        i_positive_face = i_anchor_face
        while i_positive_face == i_anchor_face:  # ensure that the positive and anchor are not the same face
            i_positive_face = random.randint(0, len(anchor_faces) - 1)
        i_negative_face = random.randint(0, len(negative_faces) - 1)
        # Now grap the face embeddings (the second element in the items)
        anchor_face = list(anchor_faces.items())[i_anchor_face][1]
        positive_face = list(anchor_faces.items())[i_positive_face][1]
        negative_face = list(negative_faces.items())[i_negative_face][1]
        return (anchor_actor, anchor_face), (anchor_actor, positive_face), (negative_actor, negative_face)

    # This will return the first name of the character
    def get_first_name(self, character):
        actor_fullname = character[character.find('--') + 2:]
        firstname = actor_fullname[:actor_fullname.find('_')]
        return firstname


        # This function can become more complicated, but for now just choose randomly
    def choose_minibatches(self, embeddings, minibatch_size):
        # Get a random order of examples
        randomorder = np.arange(0, len(embeddings))
        np.random.shuffle(randomorder)
        anc_mini_batches = []
        pos_mini_batches = []
        neg_mini_batches = []
        # loop over the actors
        for i_minibatch in range(int(len(embeddings)/minibatch_size)):
            anc_current_minibatch = []
            pos_current_minibatch = []
            neg_current_minibatch = []
            for i_anchor in randomorder[i_minibatch*minibatch_size:i_minibatch*minibatch_size + minibatch_size]:
                actors = list(embeddings.keys())  # get the actors
                #i_anchor = random.randint(0, len(embeddings) - 1)  # choose a random actor
                i_negative = i_anchor
                # get the gender of the anchor
                negative_gender = 'none'
                anchor_gender = self.gender.get_gender(self.get_first_name(actors[i_anchor]))
                while i_negative == i_anchor and anchor_gender != negative_gender:  # choose a negative actor that is not the same as the anchor
                    i_negative = random.randint(0, len(embeddings) - 1)
                    negative_gender = self.gender.get_gender(self.get_first_name(actors[i_negative]))
                # Get the keys of the embeddings and the actor faces

                anchor_actor = actors[i_anchor]
                negative_actor = actors[i_negative]
                anchor_faces = embeddings[anchor_actor]
                negative_faces = embeddings[negative_actor]
                # now randomly choose the faces
                i_anchor_face = random.randint(0, len(anchor_faces) - 1)
                i_positive_face = i_anchor_face
                while i_positive_face == i_anchor_face:  # ensure that the positive and anchor are not the same face
                    i_positive_face = random.randint(0, len(anchor_faces) - 1)
                i_negative_face = random.randint(0, len(negative_faces) - 1)
                # Now grap the face embeddings (the second element in the items)
                anchor_face = list(anchor_faces.items())[i_anchor_face][1]
                positive_face = list(anchor_faces.items())[i_positive_face][1]
                negative_face = list(negative_faces.items())[i_negative_face][1]

                # add the actors to the current minibatch
                anc_current_minibatch.append(anchor_face)
                pos_current_minibatch.append(positive_face)
                neg_current_minibatch.append(negative_face)

            anc_mini_batches.append(anc_current_minibatch)
            pos_mini_batches.append(pos_current_minibatch)
            neg_mini_batches.append(neg_current_minibatch)

        return anc_mini_batches, pos_mini_batches, neg_mini_batches

    # this function will test on the ridiclous 6

    def compare_with_anchors(self, test_face, anchor_embeddings):
        best_results = []
        for anchor in list(anchor_embeddings.items()):
            L2_results = []
            for anchor_face in list(anchor[1].items()):
                L2_results.append(self.L2_norm(test_face, anchor_face[1]))
            best_results.append((anchor[0], np.mean(L2_results)))
        best = sorted(best_results, key=lambda x: x[1])  # sort by the lowest loss
        return best[0]

    def compare_with_num_anchors(self, test_face, anchor_embeddings, num_anchors):
        best_results = []
        for anchor in list(anchor_embeddings.items()):
            L2_results = []
            for i_anchor in range(num_anchors):
                anchor_face = random.choice(list(anchor[1].items()))
                L2_results.append(self.L2_norm(test_face, anchor_face[1]))
            best_results.append((anchor[0], np.mean(L2_results)))
        best = sorted(best_results, key=lambda x: x[1])  # sort by the lowest loss
        return best[0]


    def test_ridic6(self, model):
        # run the encoder on all the actors from ridic 6
        # encoder = Encoding_Model(pretrained_model) # load the encoder
        # embeddings = encoder.generate_all_embeddings(ridic_char_path) # generate all the embeddings
        # encoder.saveEmbeddings(ridic_embeddings_path, embeddings, str(time.time()))
        #
        embeddings = encoder.loadEmbeddings(ridic_embeddings_path)
        test_data_full = pkl.load(open(test_data_path, 'rb'))
        test_data = test_data_full['examples']
        prediction_results = []
        wrong_face = []
        for i_test, test_face in enumerate(test_data):
            test_embedd = test_face[0]  # get the embeddings of the face
            best_fit = self.compare_with_num_anchors(test_embedd, embeddings, 1)
            if test_face[1] == character_to_id[best_fit[0]]:
                prediction_results.append(1)
            else:
                prediction_results.append(0)
                wrong_face.append(i_test)
        accuracy = float(sum(prediction_results))/float(len(prediction_results))
        print('the ridic 6 accuracy is :')
        print(accuracy)
        print('')





##########################
## Build the NN model ##
##########################


# train the NN model
# Some Hyperparameters
layersSize = [256, 256]
input_layer_size = 128
margin = 12.0
epochs = 500
batch_size = 30
learning_rate = 10e-3
gamma = 0.99
step_size = 10
weight_decay = 1e-5


class TNet(torch.nn.Module):

    def __init__(self, input_size):
        super(TNet, self).__init__()
        self.batchNorm = torch.nn.BatchNorm1d(input_size)
        self.lin1 = torch.nn.Linear(input_size, layersSize[0])
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(layersSize[0], layersSize[1])
        self.relu2 = torch.nn.ReLU()
        # self.lin3 = torch.nn.Linear(layersSize[1], layersSize[2])
        # self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.batchNorm(x)
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        # x = self.lin3(x)
        # x = self.relu3(x)
        return x

    def init_weights(self):
        torch.nn.init.xavier_normal(self.lin1.weight.data)
        torch.nn.init.xavier_normal(self.lin2.weight.data)
        # torch.nn.init.xavier_normal(self.lin3.weight.data)

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

model = TNet(input_layer_size)
# torch.nn.init.xavier_normal(model[1].weight.data)
# torch.nn.init.xavier_normal(model[3].weight.data)
tripletTaker = Triplet_NN(model)
#model.weight_initializer()  # initialize the variables
loss_fun = torch.nn.TripletMarginLoss(margin=margin)
model.init_weights()
test_results = []
train_results = []
loss_vec = []
accuracy_vec = []
train_accuracy_vec = []
val_loss = []

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# Split the data into test and train
train_embeddings, test_embeddings = tripletTaker.create_test_train_sets(embeddings, 0.75)

# Get the first test results
# test_results.append(tripletTaker.test_model(test_embeddings))
# accuracy = float(sum(test_results[-1][0]))/float(len(test_results[-1][0]))
# accuracy_vec.append(accuracy)
# print('the test accuracy is :')
# print(accuracy)
# print('')


for i_epoch in range(epochs):
    model.train()
    #scheduler.step()
    # clear the past minibatches
    minibatch_loss = []
    # get the anchor, positive and negative encodings.
    #for i_batch in range(batch_size):
    anchor_minibatch, positive_minibatch, negative_minibatch = tripletTaker.choose_minibatches(train_embeddings, batch_size)

    for i_minibatch, anchor_batch in enumerate(anchor_minibatch):

        anchor_minibatch_input = Variable(torch.Tensor(np.array(anchor_minibatch[i_minibatch])))
        positive_minibatch_input = Variable(torch.Tensor(np.array(positive_minibatch[i_minibatch])))
        negative_minibatch_input = Variable(torch.Tensor(np.array(negative_minibatch[i_minibatch])))
        # now run the model with the optimizer and the loss
        optimizer.zero_grad()

        model_output_anchor = model(anchor_minibatch_input)
        model_output_positive = model(positive_minibatch_input)
        model_output_negative = model(negative_minibatch_input)

        loss = loss_fun(model_output_anchor, model_output_positive, model_output_negative)  # , model_output_negative)
        loss.backward()
        optimizer.step()
        minibatch_loss.append(float(loss.data))
    loss_vec.append(np.mean(minibatch_loss))

    # do a test of the model
    if i_epoch % 50 == 0 and i_epoch > 100:

        test_results.append(tripletTaker.test_model(test_embeddings))
        accuracy = float(sum(test_results[-1][0]))/float(len(test_results[-1][0]))
        accuracy_vec.append(accuracy)
        print('the test accuracy is :')
        print(accuracy)
        print('')
        print('Epoch num is: ' + str(i_epoch))
        print('')
        train_results.append(tripletTaker.test_model(train_embeddings))
        accuracy = float(sum(train_results[-1][0])) / float(len(train_results[-1][0]))
        train_accuracy_vec.append(accuracy)

    # get the valadation loss
    anchor_minibatch, positive_minibatch, negative_minibatch = tripletTaker.choose_minibatches(test_embeddings,1)
    temp_val_loss = []
    for i_minibatch, anchor_batch in enumerate(anchor_minibatch):
        model.train(False)  # turn training off to run one example

        anchor_minibatch_input = Variable(torch.Tensor(np.array(anchor_minibatch[i_minibatch])))
        positive_minibatch_input = Variable(torch.Tensor(np.array(positive_minibatch[i_minibatch])))
        negative_minibatch_input = Variable(torch.Tensor(np.array(negative_minibatch[i_minibatch])))
        model_output_anchor = model(anchor_minibatch_input)
        model_output_positive = model(positive_minibatch_input)
        model_output_negative = model(negative_minibatch_input)

        loss = loss_fun(model_output_anchor, model_output_positive, model_output_negative)  # , model_output_negative)
        temp_val_loss.append(float(loss.data))
    val_loss.append(np.mean(temp_val_loss))
    model.train(True)  # turn training off to run one example


lossfig = pyplt.plot(loss_vec)
pyplt.plot(val_loss)
pyplt.xlabel('epoch')
pyplt.ylabel('Loss')
pyplt.show()
accfig = pyplt.plot(accuracy_vec)
pyplt.plot(train_accuracy_vec)
pyplt.xlabel('epoch')
pyplt.ylabel('Accuracy')
pyplt.show()

# save the data
# currenttime = time.time()
# pkl.dump(accuracy_vec, open(saved_data_path + '/' + 'test_accuracy' + str(currenttime), 'wb'))
# pkl.dump(train_accuracy_vec, open(saved_data_path + '/' + 'train_accuracy' + str(currenttime), 'wb'))
# pkl.dump(loss_vec, open(saved_data_path + '/' + 'loss_vec' + str(currenttime), 'wb'))
# pkl.dump(val_loss, open(saved_data_path + '/' + 'val_loss_vec' + str(currenttime), 'wb'))

tripletTaker.test_ridic6(model)
# print(np.subtract(test_results[0][-1],test_results[-1][-1]))
#



