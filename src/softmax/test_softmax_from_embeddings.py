# From the embeddings created in 'test_set_embedding_generation.py', we evaluate our softmax layer with the output data
# from that file.

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import pickle
    import os
    import numpy as np
    from imageio import imread

    import facenet.contributed.face as face

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    import torch.nn as nn
    import torch
    import torch.optim as optim
    from torch.autograd import Variable
    # from pylab import *
    import matplotlib.pyplot as plt
    import torch.optim.lr_scheduler as lr_scheduler
    import json

model_dir = os.path.abspath('../../models/softmax/ridiculous6/1915/')
model_dict = pickle.load(open(os.path.join(model_dir, 'epoch2500.pkl'), 'rb'))

model = nn.Sequential(
    nn.BatchNorm1d(128),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 7),
    )

model.load_state_dict(model_dict)

print('47')