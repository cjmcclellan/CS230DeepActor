# This program will execute the triplet loss model
# and compare possible faces with each other

#import facenet.src.train_tripletloss as trtrip
import facenet.contributed.face as face
import facenet.src.facenet as facenet
import tensorflow as tf
import imageio
import numpy as np
import os

triplet_model_path = '/home/connor/Documents/CS230/CS230DeepActor/models/triplet_loss/20180316-231123/'
Kylo_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline/Kylo_Ren--Adam_Driver/Kylo_Ren--Adam_Driver_3'
Gal_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline/Diana--Gal_Gadot/Diana--Gal_Gadot_3'
Chris_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline/Bruce_Wayne--Christian_Bale/Bruce_Wayne--Christian_Bale_3'
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

# This class will be used to test models
class Testing_Class:
    def __init__(self, model_path = None, model = None, test_examples = None, input_name, model_output):
        # save the input and output names
        self.input_name = input_name
        self.model_output = model_output
        # Check if the user inputs model path or a model
        if model_path is not None:
            # add code to import model
            self.model = None
        else:
            self.model = model

        self.test_examp = test_examples

    # This function will run the model and output the embeddings
    def gen_embeddings(self, input_face):
        return self.model.sess.run(self.model_output, feed_dict={self.input_name: input_face})

## My new class for triplet identifier
class TripRecognition:
    def __init__(self, model_path):
        self.detect = face.Detection()
        self.encoder = Encoder(model_path)

    # takes in an image and outputs the embeddings of that image
    def gen_embedding(self, image):
        face = self.detect.find_faces(image)
        if len(face) == 0:
            return [None]
        return self.encoder.generate_embedding(face[0])


def L2_Loss(vec1, vec2):
    return np.sum(np.square(np.subtract(vec1, vec2)))

def tripLoss(vec1, vec2):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(vec1, vec2)), 1)
        loss = tf.reduce_mean(tf.maximum(pos_dist, 0.0), 0)

    return loss

# This will return the face with the number 'num' in the file name
def returnFace_num(faces, num):
    for face in faces:
        if num in face:
            return face
    print('Could not find the right face, just retruned the first one')
    return faces[0]


# The actual program

# make the recognizer
tripletModel = TripRecognition(triplet_model_path)

# test the model
Kylo_face = imageio.imread(Kylo_path)
Kylo_embeddings = tripletModel.gen_embedding(Kylo_face)

Gal_face = imageio.imread(Gal_path)
Gal_embeddings = tripletModel.gen_embedding(Gal_face)

Chris_face = imageio.imread(Chris_path)
Chris_embeddings = tripletModel.gen_embedding(Chris_face)

truth_embeddings = [Chris_embeddings, Gal_embeddings, Kylo_embeddings]
names = ['Chris', 'Gal', 'Kylo']

# Now loop through all the known characters to compare the embeddings
saved_embeddings = {}  # save all the embeddings of the first actor image
final_results = [] # keep track of all the loss and characters
prediction_results = []  # keep track of the weather the prediction was right or wrong
first_round = True # keep track if this was the first round in the loop
incorrect_actor = [] # keep track of the actors that were wrong
correct_actor = []  # keep track of the actors that were right
for rootdirectory, actors, files_blank in os.walk(actors_path):
    for i_actor, actor in enumerate(actors):
        if actor == 'Jack_Dawson--Leonardo_DiCaprio':
            a = 5
        actor_results = []  # keep track of all the loss and characters
        for root_actor, _, actor_faces in os.walk(rootdirectory + actor):
            if len(actor_faces) > 3:
                actor_image = imageio.imread(root_actor + '/' + returnFace_num(actor_faces, '3'))
            elif len(actor_faces) > 2:
                actor_image = imageio.imread(root_actor + '/' + returnFace_num(actor_faces, '2'))
            else:
                actor_image = imageio.imread(root_actor + '/' + returnFace_num(actor_faces, '0'))
            actor_embeddings = tripletModel.gen_embedding(actor_image)
            if actor_embeddings[0] == None:  # Check if the model found a face
                actor_results.append(['None', 100])
                print(actor + ' could not detect face')
            else:  # if there was a face, then run the testing
                for rootdir, characters, files in os.walk(actors_path):
                    for character in characters:
                        if first_round:  # compute the embeddings if the first round
                            for face_root, _, faces in os.walk(rootdir + '/' + character):
                                new_character_image = imageio.imread(face_root + '/' + returnFace_num(faces, '0'))
                                new_character_embeddings = tripletModel.gen_embedding(new_character_image)
                                if new_character_embeddings[0] == None: # check if the character has embeddings that work
                                    print(character + ' did not have any faces')
                                    actor_results.append(['None', 100])
                                    saved_embeddings[character] = new_character_embeddings
                                else:
                                    saved_embeddings[character] = new_character_embeddings
                                    actor_results.append([character, L2_Loss(actor_embeddings, new_character_embeddings)])
                        else: # use the computed embeddings if not the first round
                            # check that the character has a face
                            if saved_embeddings[character][0] == None:
                                actor_results.append(['None', 100])
                            else:
                                actor_results.append([character, L2_Loss(actor_embeddings, saved_embeddings[character])])
                first_round = False
            # else:
            #     actor_results.append(['None', 100]) # if the comparrison didn't work, add a large loss
            #     print(actor + ' does not have 4 face images')

        final_results.append(sorted(actor_results, key=lambda  x:x[1])[0:5])  # save the top five choices of the

        # check if the prediction was right
        if final_results[i_actor][0][0] == actor:
            correct_actor.append((actor, final_results[i_actor][0][1]))
            prediction_results.append(1)
        elif final_results[i_actor][0][0] == 'None':
            a = 'Done Nothing'
        else:
            incorrect_actor.append((actor, final_results[i_actor][0][0]))
            prediction_results.append(0)
print(final_results)
print(prediction_results)
print('The accuracy is: ' + str(float(sum(prediction_results))/float(len(prediction_results))))
print('')
print('The incorrect results are: ')
print( incorrect_actor)
print('The correct results are: ')
print(correct_actor)

    # best_fit = sorted(results, key=lambda x: x[1])
    # print(names[i_embed])
    # print(best_fit[0:5])
a = 5


