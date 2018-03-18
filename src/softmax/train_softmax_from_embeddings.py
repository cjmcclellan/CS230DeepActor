# From the embeddings created in 'softmax_embedding_generation.py', we train a softmax layer with the output data from
# that file

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import pickle
    import os
    from imageio import imread

    import facenet.contributed.face as face


r6_path = '../../train_data/FaceID/The_Ridiculous_6_2015_movie/extra/'
dataset_path = os.path.abspath(r6_path + 'r6_dataset.pkl')
with open(dataset_path, 'rb') as f:
    dataset_dict = pickle.load(f)


print('47')