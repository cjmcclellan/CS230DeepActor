# Tests the trained svm on the test set

import pickle
import os
import numpy as np
import pandas as pd


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

###############################
# Load Model
###############################
model_path = os.path.abspath('../../models/svm/ridiculous6/2409/svm_model.pkl')

with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

predictions = classifier.predict_proba(X_test)
class_pred = np.argmax(predictions, axis=1)
comparison = class_pred==Y_test.reshape(len(Y_test))
comp_series = pd.Series(comparison)
comp_series.to_csv('test_comparison.csv')
num_correct = np.sum(comparison)
test_accuracy = num_correct/num_examples

print('Test Results:')
print('Number of test examples:', num_examples)
print('Number correct:', num_correct)
print('Accuracy:', test_accuracy)
