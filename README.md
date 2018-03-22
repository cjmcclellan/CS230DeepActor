# CS230DeepActor
CS 230 project on using NN's for identifiying actors faces in videos

Thie project is completed and can identify actors from the Ridiculous 6 movie using SVM, softmax and a NN trained on triplet loss.

# Datasets/Models

The datasets (train & test), along with the trained classification models for SVC and Softmax can be found a the following link:
https://drive.google.com/drive/folders/1fYcIkDWaUpxP-kcmnKbTkOxsP8awkZXN?usp=sharing

# Main Code	

  ##### SVC Loss
  The SVC files can be found in /src/svm/ under 'train_svm_from_embeddings.py' and 'test_svm_from_embeddings'. Both the training and test files should run as compressed datasets and test sets will be in their respective /train_data/ and /test_data/ folders (full paths given in code). As of now, the test code has the model we used in the course paper hard-coded to load.

  ##### Softmax Loss
  The softmax files can be found in /src/softmax/ under 'train_softmax_from_embeddings.py' and 'test_softmax_from_embeddings'. Both the training and test files should run as compressed datasets and test sets will be in their respective /train_data/ and /test_data/ folders (full paths given in code). As of now, the test code has the model we used in the course paper hard-coded to load.
  
  ###### Triplet Loss
  The triplet loss training py file is in the /src/triplet_loss directory under the file name "Triplet_Loss_TrainingNN(maincode).py".  The code can run after downloading from github, but only using 3 training examples (250 examples were used for actually training, but the file size was too large for uploading to github)


# Example Code
For the project Milestone, the running code is "multi_image_face_recognition_demo.py" and "face_ID_demo.py", under the src directory.  multi_image_face_recognition_demo.py will detecte faces in 6 example images from the WIDER database and face_ID_demo.py will identify Nick Offerman and Rhetta from an image, serving as the baseline for this project


# Required for Running Code
To run the above code, a trained model is needed.  However, the model was too large to place on github.  Please downlaod the model from the google drive folder "pretrained_facenet" in this link https://drive.google.com/drive/folders/1znBZRKEqYspPqAngyOwOgKLBLENI7uAk?usp=sharing

Place the folder "pretrained_facenet" (the folder "20170512-110547" which contains 4 files) to the path "CS230DeepActor/models/pretrained_facenet/".  This shold place the model in as such "CS230DeepActor/models/pretrained_facenet/20170512-110547" with 4 files in the "20170512-110547" folder.

The libraries listed in the requirements.txt file are also needed
