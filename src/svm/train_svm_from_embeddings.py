# Use the same embedding used to train the softmax classifier to train a Linear SVM. While this is sort
# of a hybrid NN and ML classification (since we are using embeddings from an Inception net), it is
# still interesting to see how the final classification stage changes between an SVM and the NN made for
# softmax.


import pickle
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

r6_path = os.path.abspath('../../train_data/FaceID/The_Ridiculous_6_2015_movie/extra/')
r6_model_path = os.path.abspath('../../models/svm/ridiculous6/')
r6_formatted_dataset_path = os.path.join(r6_path, 'r6_formatted_dataset.pkl')
with open(r6_formatted_dataset_path, 'rb') as f:
    formatted_dataset_dict = pickle.load(f)

X_train = formatted_dataset_dict['X_train'].numpy()
Y_train = formatted_dataset_dict['Y_train'].numpy()
X_val = formatted_dataset_dict['X_val'].numpy()
Y_val = formatted_dataset_dict['Y_val'].numpy()

# Combine the validation set into training. Different structure needed for this part over NN training
X_train = np.append(X_train, X_val, axis=0)
Y_train = np.argmax(np.append(Y_train, Y_val, axis=0), axis=1)

classifier = SVC(kernel='linear', C=7, probability=True)
classifier.fit(X_train, Y_train)
# train_sizes, train_scores, valid_scores = learning_curve(classifier, X_train, Y_train)

# Gets a model
done = False
model_dir = ''
while not done:
    model_num = str(int(np.random.random()*10000))
    if model_num not in os.listdir(r6_model_path):
        model_dir = os.path.join(r6_model_path, model_num)
        os.makedirs(model_dir)
        done = True

model_location = os.path.join(model_dir, 'svm_model.pkl')
with open(model_location, 'wb') as f:
    pickle.dump(classifier, f)

print('SVC Model Trained! Model is stored in', model_location)
