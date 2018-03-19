# creates face embeddings from flattened images. Mostly modified code from face.py
# Specific for ridiculous 6 2015 movie

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import pickle
    import os
    from imageio import imread

    import facenet.contributed.face as face
    import pandas as pd


r6_path = '../test_data/ridiculous6/'
faces_dir = os.path.abspath(r6_path + 'flattened/')
# check faces directory
if not os.path.isdir(faces_dir):
    exit('faces_dir not a directory.')

# Labelled dataset dataframe
pic_label_df = pd.read_csv(os.path.join(faces_dir, 'pics_labels.csv'))
pic_label_df.head()
face_files = pic_label_df.as_matrix(columns=['photo'])
face_files = face_files.reshape(len(face_files))
labels = pic_label_df.as_matrix(columns=['label'])
labels = labels.reshape(len(labels))


dataset_dict = dict()
dataset_dict['num_labels'] = 7

# Initialize encoder. Pretrained model hard coded in face.py
encoder = face.Encoder()

# List for input-output pairs
examples = list()

# Get current, preprocessed faces
faces = list()
for face_file, label in zip(face_files, labels):
    curr_face = face.Face()
    curr_face.name = label
    curr_face.image = imread(os.path.join(faces_dir, face_file))
    curr_face.embedding = encoder.generate_embedding(curr_face)
    faces.append(curr_face)
    examples.append((curr_face.embedding, curr_face.name))

dataset_dict['faces'] = faces
dataset_dict['num_examples'] = len(examples)
dataset_dict['examples'] = examples

with open(r6_path + 'r6_testset.pkl', 'wb') as f:
    pickle.dump(dataset_dict, f)


