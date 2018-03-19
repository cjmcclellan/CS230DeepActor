# creates face embeddings from flattened images. Mostly modified code from face.py
# Specific for ridiculous 6 2015 movie

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import pickle
    import os
    from imageio import imread

    import facenet.contributed.face as face


r6_path = '../../train_data/FaceID/The_Ridiculous_6_2015_movie/extra/'
faces_dir = os.path.abspath(r6_path + 'flattened/')
# check faces directory
if not os.path.isdir(faces_dir):
    exit('faces_dir not a directory.')

# get list of characters
char_actors = os.listdir(faces_dir)

dataset_dict = dict()
dataset_dict['num_labels'] = len(char_actors)
dataset_dict['labels'] = char_actors

# Initialize encoder. Pretrained model hard coded in face.py
encoder = face.Encoder()

# List for input-output pairs
examples = list()

# Iterate over labels
for label in char_actors:
    # Get current, preprocessed faces
    face_files_root = os.path.join(faces_dir, label)
    face_files = os.listdir(face_files_root)
    faces = list()
    for face_file in face_files:
        curr_face = face.Face()
        curr_face.name = label
        curr_face.image = imread(os.path.join(face_files_root, face_file))
        curr_face.embedding = encoder.generate_embedding(curr_face)
        faces.append(curr_face)
        examples.append((curr_face.embedding, curr_face.name))

    dataset_dict[label + '__faces'] = faces
    dataset_dict[label + '__num_examples'] = len(faces)

dataset_dict['num_examples'] = len(examples)
dataset_dict['examples'] = examples

with open(r6_path + 'r6_dataset.pkl', 'wb') as f:
    pickle.dump(dataset_dict, f)


