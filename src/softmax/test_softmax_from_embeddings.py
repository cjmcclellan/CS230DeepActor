# From the embeddings created in 'test_set_embedding_generation.py', we evaluate our softmax layer with the output data
# from that file.

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    import pickle
    import os
    import numpy as np

    import torch.nn as nn
    import torch
    from torch.autograd import Variable
    import pandas as pd


########################
# Load NN
########################

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

##############################
# Load test set
##############################
test_root = os.path.abspath('../../test_data/ridiculous6/')
with open(os.path.join(test_root, 'r6_testset.pkl'), 'rb') as f:
    testset_dict = pickle.load(f)

num_examples = testset_dict['num_examples']
examples = testset_dict['examples']

# Encodings are length 128
X_test = np.zeros((len(examples), 128))
# outputs already target index (i.e. argmax(onehotvector(text_label)).
Y_test = np.zeros((len(examples), 1))
for i, example in enumerate(examples):
    X_test[i, :] = example[0]
    Y_test[i, :] = example[1]
X_test = Variable(torch.Tensor(X_test))
Y_test = Variable(torch.LongTensor(Y_test))

######################################
# Evaluate Model
######################################

Yhat = model(X_test)
temp = Yhat.data.numpy()
temp3 = torch.max(Yhat, 1)[1].data.numpy()
temp2 = Y_test.data.numpy()
comparison = (torch.max(Yhat, 1)[1].data.numpy() == Y_test.data.numpy().reshape(len(examples)))
comp_series = pd.Series(comparison)
comp_series.to_csv('test_comparison.csv')
num_correct = np.sum(comparison)
test_accuracy = num_correct/num_examples

print('Test Results:')
print('Number of test examples:', num_examples)
print('Number correct:', num_correct)
print('Accuracy:', test_accuracy)
