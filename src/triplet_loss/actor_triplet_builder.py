# This script will be used to generate training data for the
# actor triplet loss.  Training examples will be generated using
# actors the same Movie/TV Show

import src.build_face_database_main as bfd
import os
import random
import shutil



# Get the path of the file for movies to use
#movie_list_path = input('Give the path to the list of Movies for training: ')
movie_list_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/movie_list'

# Read in the movie list
with open(movie_list_path) as movie_list_file:
    movie_list_str = movie_list_file.read()
    movie_list_dirty = movie_list_str.split('\n')
    # Remove any blank spaces
    movie_list = [movie for movie in movie_list_dirty if movie != '']
    del movie_list_dirty
    del movie_list_str

# Now run through each movie and collect faces

# Some "hyperparameters"
movie_selection = 1 # always choose the first choice
num_actors = 4  #
num_images = 9 #
#print(os.path.exists('/home/connor/Documents/CS230/CS230DeepActor/train_data/FaceID/The_Dark_Knight_2008_movie/downloaded/Bruce Wayne The Dark Knight Christian Bale 2008'))
for movie in movie_list:
    # Run the build face database
    #try:
    bfd.main(movie, movie_selection, num_actors, num_images)
    #except:
     #   a = 10

# Now that the faces have been collected, sort them into triplets
faceID_path = '/home/connor/Documents/CS230/CS230DeepActor/train_data/FaceID/'
triplet_examples = '/home/connor/Documents/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/Baseline/'


# Look at all the movies
for root, movies, files_blank in os.walk(faceID_path):
    for movie in movies:
        if 'movie' in movie:
            # Now look at all the characters
            for root2, characters, files in os.walk(root + movie + '/flattened/'):

                # First attempt to just get triplets from the same movie (not considering gender)
                for i_char, character in enumerate(characters):
                    triplet_example_path = triplet_examples + character + '/'
                    for root3, dir_none, images in os.walk(root2 + character + '/'):
                        if len(images) > 1:  # make sure there are at least two faces

                            if not os.path.exists(triplet_example_path):
                                os.makedirs(triplet_example_path)
                            images.sort()  # sort the images to ensure the examples use the most likely examples
                            for i_face, face in enumerate(images):
                                # Now create the triplet training example
                                shutil.copy2(root3 + face,
                                             triplet_example_path + character + '_' + str(i_face))







