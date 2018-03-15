# This script will be used to generate training data for the
# actor triplet loss.  Training examples will be generated using
# actors the same Movie/TV Show

import src.build_face_database
import os



# Get the path of the file for movies to use
movie_list_path = input('Give the path to the list of Movies for training: ')

with open(movie_list_path):

