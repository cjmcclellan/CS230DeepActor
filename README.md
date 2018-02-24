# CS230DeepActor
CS 230 project on using NN's for identifiying actors faces in videos

This project is currently able to detect a face in arbritary photos and idenfity the actors Nick Offerman and Rhetta (actors in the TV Show Parks and Recreation)

# Example Code
For the project Milestone, the running code is "multi_image_face_recognition_demo.py" and "face_ID_demo.py", under the src directory.  multi_image_face_recognition_demo.py will detecte faces in 6 example images from the WIDER database and face_ID_demo.py will identify Nick Offerman and Rhetta from an image, serving as the baseline for this project


# Required for Running Code
To run the above code, a trained model is needed.  However, the model was too large to place on github.  Please downlaod the model from the google drive folder "asdfsd" and add to the path "CS230DeepActor/models/pretrained_facenet/".  This shold place the model in as such "CS230DeepActor/models/pretrained_facenet/20170512-110547" with 4 files in the "20170512-110547" folder.

The libraries listed in the requirements.txt file are also needed
