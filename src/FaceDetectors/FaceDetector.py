### This program will be used for taking one of the face models and creating a frame with bounding boxes



import torch
import facenet.contributed.face as face
from torch.autograd import Variable
import tensorflow as tf
from CS230DeepActor.src.triplet_loss.TNet_file import TNet
from CS230DeepActor.src.triplet_loss.triplet_Loss_TrainingNN_maincode import Encoding_Model
import cv2
import numpy as np
import os


class FaceDetector(object):

    def __init__(self, Id_model_path, embeddings_model, ref_faces_path, DoId = True, threshold = [0.6, 0.7, 0.7], minsize = 10):
        self.doID = DoId
        # check if we should load the Id model or not
        if DoId:
            self.embeddings_model = Encoding_Model(embeddings_model)
            self.actorsinVideo(ref_faces_path)
            print('imported the encoding model')

        self.id_model = torch.load(Id_model_path)
        self.detector = face.Detection()
        self.image_size = self.detector.face_crop_size
        print('imported face detect')
        self.detector.threshold = threshold
        self.detector.minsize = minsize
        self.detector.face_crop_margin = 16
        self.ref_face = None

    def addRefFace(self, face_path):
        self.ref_face = self.embeddings_model.imagetoface(imagepath=face_path)
        self.ref_face_embeddings = self.embeddings_model.generate_embedding(self.ref_face)

# this will close the encoder session to relase memeory
    def closeSess(self):
        if self.doID:
            self.embeddings_model.closeSess()
            del self.embeddings_model
        
    # give a path to actor faces in the video
    def actorsinVideo(self, pathtoactors):
        self.actor_embeddings = {}
        # look at all the actor faces
        for root, dirs, images in os.walk(pathtoactors):
            for image in images:
                # for each face, import the face image and add the embeddings
                face = self.embeddings_model.imagetoface(imagepath=root +'/'+ image)
                self.actor_embeddings[image] = self.embeddings_model.generate_embedding(face)

    # input the actors face and the frame, this will then output the bounding box
    def id_face(self, frame, threshold, wanted_faces = None):
        faces = self.detector.find_faces(frame)
        #print('detecting faces')
        ref_faces = wanted_faces.keys()
        for id_face in faces:
            # if we want to ID faces, then Id the faces
            if self.doID:

                face_id = self.bestMatch(id_face, threshold, ref_faces) # get the best match to the face

                # check that the face id is not none
                if face_id is not None:
                    for actor_name in ref_faces:
                        if actor_name.find(face_id) != -1: # check if the reference actor is in the face id'd
                            color = wanted_faces[actor_name]
                            print('############################## ' + actor_name + ' ###############################')
                            cv2.rectangle(frame, (id_face.bounding_box[0], id_face.bounding_box[1]), \
                                      (id_face.bounding_box[2], id_face.bounding_box[3]), color, 2)
                # if it is none, then draw a black box
                else:
                    print('############################## None ###############################')
                    cv2.rectangle(frame, (id_face.bounding_box[0], id_face.bounding_box[1]), \
                                  (id_face.bounding_box[2], id_face.bounding_box[3]), (0,0,0), 4)
            # else, just add the bounding box
            else:
                print('Found a face')
                cv2.rectangle(frame, (id_face.bounding_box[0], id_face.bounding_box[1]), \
                              (id_face.bounding_box[2], id_face.bounding_box[3]), (0, 255, 0), 3)
        return frame

    # this function will return the best match given the face (image) input
    def bestMatch(self, face, threshold, ref_faces, bestof = 3):
        best_match = 'nothing'  # make the inital best match 'nothing'
        best_score = threshold
        scores = {ref_face: [] for ref_face in ref_faces}  # get a dict of the scores
        face_embeddings = self.embeddings_model.generate_embedding(face)
        print('#############')
        # look through each actor embeddings
        for actor_name, actor_embeddings in self.actor_embeddings.items():
            score = self.L2Face(face_embeddings, actor_embeddings)
            name = self.removeNum(actor_name)
            if name not in list(scores.keys()):
                scores[name] = []
            scores[name].append(score)
            # if score < best_score:
            #     best_match = actor_name
            #     best_score = score
        print('#############')
        for actor_name, score in scores.items():
            score.sort()
            avg_score = sum(score[:bestof])/len(score[:bestof])
            print(actor_name)
            print(avg_score)
            print('')
            if avg_score < best_score: # mak sure the avg sorce is less than best score
                best_match = actor_name
        # now remove the file extenstion from the name
        period = best_match.find('.')
        best_match = best_match[:period - 1]
        return best_match

    # this function will remove numbers from the string
    def removeNum(self, string):
        temp = ''.join(i for i in string if not i.isdigit())
        return temp
        # period = temp.rfind('.')
        # return temp[:period - 1]


    def L2Face(self, ref_face, id_face):
        self.id_model.train(False)
        temp = []
        temp.append(ref_face)
        ref_result = self.id_model(Variable(torch.Tensor(np.array(temp))))
        temp = []
        temp.append(id_face)
        id_result = self.id_model(Variable(torch.Tensor(np.array(temp))))
        return np.sum(np.square(np.subtract(np.array(ref_result.data), np.array(id_result.data))))
