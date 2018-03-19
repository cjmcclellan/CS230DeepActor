# From the embeddings created in 'softmax_embedding_generation.py', we train a softmax layer with the output data from
# that file. Borrowed structure from classifier.py

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



def random_mini_batches(X, Y, minibatch_size):
    '''
    Returns a list of (X_batch, Y_batch) tuples
    X has shape (num_examples, num_feature)
    Y has shape (num_examples, num_labels)
    '''
    m = X.shape[0]
    # get a random order
    randord = np.arange(0, m)
    np.random.shuffle(randord)
    minibatches = list()
    num_minibatches = int(m / minibatch_size)
    for i in range(num_minibatches):
        X_curr = X[randord[i*minibatch_size:(i+1)*minibatch_size], :]
        Y_curr = Y[randord[i*minibatch_size:(i+1)*minibatch_size], :]
        minibatches.append((X_curr, Y_curr))
    return minibatches


r6_path = '../../train_data/FaceID/The_Ridiculous_6_2015_movie/extra/'
r6_model_path = os.path.abspath('../../models/softmax/ridiculous6/')
dataset_path = os.path.abspath(r6_path + 'r6_dataset.pkl')
with open(dataset_path, 'rb') as f:
    dataset_dict = pickle.load(f)


print('\nTotal number of examples: {}'.format(dataset_dict['num_examples']))
num_examples = dataset_dict['num_examples']

# List distribution of data
print('Dataset distribution')
# percentage of database a specific label
dataset_perc = list()
for label in dataset_dict['labels']:
    dataset_perc.append(dataset_dict[label + '__num_examples'] / num_examples)
    print(label, ':', dataset_perc[-1])

print('\nMost represented example:', max(dataset_perc), 'with label', dataset_dict['labels'][np.argmax(dataset_perc)])
print('Least represented example:', min(dataset_perc), 'with label', dataset_dict['labels'][np.argmin(dataset_perc)])

# train, dev split
train_perc = .8
dev_perc = 1-train_perc

examples = dataset_dict['examples']

np.random.shuffle(examples)
training, validation = examples[:int(num_examples*train_perc)], examples[int(num_examples*train_perc):]
print('\nTraining examples: {} (at {}%)'.format(len(training), train_perc*100))
tdict = {}
for ex in training:
    if ex[1] in tdict.keys():
        tdict[ex[1]] += 1
    else:
        tdict[ex[1]] = 1

print('Training set distribution')
train_set_perc = list()
train_keys = list()
for key, val in tdict.items():
    train_set_perc.append(val/len(training))
    train_keys.append(key)
    print(key, ':', train_set_perc[-1])

print('\nMost represented training example:', max(train_set_perc), 'with label', train_keys[np.argmax(train_set_perc)])
print('Least represented training example:', min(train_set_perc), 'with label', train_keys[np.argmin(train_set_perc)])

print('\nValidation examples: {} (at {}%)'.format(len(validation), dev_perc*100))
vdict = {}
for ex in validation:
    if ex[1] in vdict.keys():
        vdict[ex[1]] += 1
    else:
        vdict[ex[1]] = 1

print('Validation set distribution')
val_set_perc = list()
val_keys = list()
for key, val in vdict.items():
    val_set_perc.append(val/len(validation))
    val_keys.append(key)
    print(key, ':', val_set_perc[-1])

print('\nMost represented validation example:', max(val_set_perc), 'with label', train_keys[np.argmax(val_set_perc)])
print('Least represented validation example:', min(val_set_perc), 'with label', train_keys[np.argmin(val_set_perc)])

###################
# Build vectorized training dataset
###################
label_train_list = np.array([elem[1] for elem in training])

# y outputs
label_encoder = LabelEncoder()
int_encoded = label_encoder.fit_transform(label_train_list)

ohe = OneHotEncoder(sparse=False)
# ohe.fit(range(dataset_dict['num_labels']))
int_encoded = int_encoded.reshape(len(int_encoded), 1)
Y_train = torch.LongTensor(ohe.fit_transform(int_encoded))

# X
X_train = np.zeros((len(training), len(training[0][0])))
for i, elem in enumerate(training):
    X_train[i, :] = elem[0]
X_train = torch.Tensor(X_train)

#######################
# Build vectorized validation dataset
#######################
label_validation_list = np.array([elem[1] for elem in validation])

# y outputs
label_encoder = LabelEncoder()
int_encoded = label_encoder.fit_transform(label_validation_list)

ohe = OneHotEncoder(sparse=False)
# ohe.fit(range(dataset_dict['num_labels']))
int_encoded = int_encoded.reshape(len(int_encoded), 1)
Y_val = torch.LongTensor(ohe.fit_transform(int_encoded))

# X
X_val = np.zeros((len(validation), len(validation[0][0])))
for i, elem in enumerate(validation):
    X_val[i, :] = elem[0]
X_val = torch.Tensor(X_val)


#######################
# Building the model, encoding 128 vector
######################
model = nn.Sequential(
    nn.BatchNorm1d(128),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, dataset_dict['num_labels']),
    )

loss_func = nn.CrossEntropyLoss()

epochs = 5000
lr = 0.1e-3
minibatch_size = 32
num_minibatches = int(len(training)/minibatch_size)

weight_decay = 0.0
step_size = 10
gamma = 0.99
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_vec = list()
accuracy_vec = list()
vloss_vec = list()
vaccuracy_vec = list()

x = Variable(X_train)
y = Variable(Y_train)
xv = Variable(X_val)
yv = Variable(Y_val)

nn.init.xavier_normal(model[1].weight.data)
nn.init.xavier_normal(model[3].weight.data)
nn.init.xavier_normal(model[5].weight.data)

lr_vec = list()
opt_state = list()

# Gets a model
done = False
while not done:
    model_num = str(int(np.random.random()*10000))
    if model_num not in os.listdir(r6_model_path):
        os.makedirs(os.path.join(r6_model_path, model_num))
        done = True


for epoch in range(epochs):
    # Step lr scheduler
    scheduler.step()
    minibatches = random_mini_batches(x, y, minibatch_size)

    for minibatch in minibatches:
        xbatch, ybatch = minibatch
        model_out = model(xbatch)
        loss = loss_func(model_out, torch.max(ybatch, 1)[1])
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Get loss for whole batch
    yhat = model(x)
    accuracy_vec.append(np.sum((torch.max(yhat, 1)[1] == torch.max(y, 1)[1]).data.numpy())/len(training))
    loss = loss_func(yhat, torch.max(y, 1)[1])
    loss_vec.append(float(loss.data.numpy()))
    # track loss for validation set
    yvhat = model(xv)
    vaccuracy_vec.append(np.sum((torch.max(yvhat, 1)[1] == torch.max(yv, 1)[1]).data.numpy())/len(validation))
    vloss = loss_func(yvhat, torch.max(yv, 1)[1])
    vloss_vec.append(float(vloss.data.numpy()))
    # track learning rate
    lr_vec.append(optimizer.param_groups[0]['lr'])
    opt_state.append(optimizer.state_dict())

    if (epoch+1) % 100 == 0:
        with open(os.path.join(r6_model_path, '{}/epoch{}.pkl'.format(model_num, epoch+1)), 'wb') as f:
            pickle.dump(model.state_dict(), f)

perf_dict = dict()
perf_dict['scheduler_step_rate'] = step_size
perf_dict['scheduler_gamma'] = gamma
perf_dict['weight_decay'] = weight_decay
with open(os.path.join(r6_model_path, '{}/model_description.txt'.format(model_num)), 'w') as f:
    f.writelines(json.dumps(perf_dict, sort_keys=True))

perf_dict['train_loss'] = loss_vec
perf_dict['validation_loss'] = vloss_vec
perf_dict['learning_rate'] = lr_vec
perf_dict['train_accuracy'] = accuracy_vec
perf_dict['validatio_accuracy'] = vaccuracy_vec
perf_dict['optimizer'] = optimizer
with open(os.path.join(r6_model_path, '{}/model_performance.pkl'.format(model_num)), 'wb') as f:
    pickle.dump(perf_dict, f)


print('Start Loss:', loss_vec[0], 'End Loss:', loss_vec[-1])
plt.figure()
plt.semilogy(loss_vec)
plt.semilogy(vloss_vec)
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.semilogy(accuracy_vec)
plt.semilogy(vaccuracy_vec)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.semilogy(lr_vec)
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.show()



