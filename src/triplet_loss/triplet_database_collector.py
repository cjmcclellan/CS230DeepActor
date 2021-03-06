# This script will be used to generate training data for the
# actor triplet loss.  Training examples will be generated using
# actors the same Movie/TV Show

import src.build_face_database_main as bfd
import os
import random
import shutil



# Get the path of the file for movies to use
#movie_list_path = input('Give the path to the list of Movies for training: ')
movie_list_path = '/home/connor/Documents/Stanford_Classes/CS230/CS230DeepActor/train_data/Triplet_Loss/movie_list'

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
num_actors = 8  #
num_images = 20 #

for movie in movie_list:
    # Run the build face database
    bfd.main(movie, movie_selection, num_actors, num_images)

# Now that the faces have been collected, sort them into triplets
faceID_path = '/home/connor/Documents/Stanford_Classes/CS230/CS230DeepActor/train_data/FaceID/'
triplet_examples = '/home/connor/Documents/Stanford_Classes/CS230/CS230DeepActor/train_data/Triplet_Loss/Movie_Triplets/'

# Some "hyperparameters"
num_of_examples_per_character = 2

# Look at all the movies
for root, movies, files_blank in os.walk(faceID_path):
    for movie in movies:
        i_training_example = 0
        if 'movie' in movie:
            # Now look at all the characters
            for root2, characters, files in os.walk(root + movie + '/flattened/'):

                # First attempt to just get triplets from the same movie (not considering gender)
                for i_char, character in enumerate(characters):
                    other_characters = [ch for ch in characters if ch != character] # collect the other characters
                    for root3, dir_none, images in os.walk(root2 + character + '/'):
                        images.sort()  # sort the images to ensure the examples use the most likely examples
                        for i_image in range(num_of_examples_per_character):
                            i_anchor = i_image * 2
                            i_positive = i_anchor + 1
                            if i_positive < len(images) and len(other_characters) > 0: # Make sure there are enough faces
                                # randomly pick another character from this movie as the negative
                                if len(other_characters) == 0:

                                    a = 5
                                negative_ch = random.choice(other_characters)
                                for root4, dir_blank, neg_images in os.walk(root2 + negative_ch + '/'): # get a negative image
                                    if len(neg_images) > 0:
                                        negative_image_name = random.choice(neg_images)

                                        negative_image = root4 + negative_image_name

                                # Now create the triplet training example if there enough neg images
                                if len(neg_images) > 0:
                                    triplet_example_path = triplet_examples + movie + '/example' + str(i_training_example) + '/'
                                    if not os.path.exists(triplet_example_path):
                                        i_training_example += 1 # increase the training example number
                                        os.makedirs(triplet_example_path)
                                        shutil.copy2(root3 + images[i_anchor],
                                                     triplet_example_path + 'anchor_' + images[i_anchor])
                                        shutil.copy2(root3 + images[i_positive],
                                                     triplet_example_path + 'positive_' + images[i_positive])
                                        shutil.copy2(negative_image,
                                                     triplet_example_path + 'negative_' + negative_image_name)







