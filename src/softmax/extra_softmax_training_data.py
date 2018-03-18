# Very manual way of making more data for softmax training

from imdb import IMDb
import os
import subprocess
import sys

sys.path.append('../')
import facenet.contributed.face as face
from imageio import imread, imwrite
from os import listdir
from os.path import isfile, join
from shutil import copy

num_images = 100


output_root = '../../train_data/FaceID/'
movie_dir = output_root + 'The_Ridiculous_6_2015_movie/'
downloads_dir = os.path.abspath(movie_dir + 'extra/downloaded/')
raw_dir = os.path.abspath(movie_dir + 'extra/raw/')
face_dir = os.path.abspath(movie_dir + 'extra/face/')
flat_dir = os.path.abspath(movie_dir + 'extra/flattened/')

actors = ['Adam Sandler', 'Terry Crews', 'Luke Wilson', 'Jorge Garcia', 'Taylor Lautner', 'Rob Schneider', 'Will Forte']
characters = ['Tommy', 'Chico', 'Danny', 'Herm', 'Lil Pete', 'Ramon', 'Will Patch']

# print('\nDownloading {} images for {}'.format(str(num_images), actor))
# # subprocess.check_output(
# #     ['googleimagesdownload', '-k', '{}'.format(query), '-l', str(num_images), '-o', downloads_dir, '-f',
# #      'jpg', '-s', 'medium'])
# subprocess.check_output(
#     ['googleimagesdownload', '-k', '{}'.format(query), '-l', str(num_images), '-o', downloads_dir, '-f', 'jpg', '-s', 'medium'])
# print('Downloads finished')
# print('Processing images and finding faces')

for i in range(1, len(actors)):

    actor = actors[i]
    character = characters[i]

    char_actor = '/' + '--'.join(['_'.join(character.split()), '_'.join(actor.split())]) + '/'

    # Create directories for other outputs
    if not os.path.exists(raw_dir + char_actor):
        os.makedirs(raw_dir + char_actor)

    if not os.path.exists(face_dir + char_actor):
        os.makedirs(face_dir + char_actor)

    if not os.path.exists(flat_dir + char_actor):
        os.makedirs(flat_dir + char_actor)

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)

    query = actor
    query = ''.join(query.split(','))
    query_dir = downloads_dir + '/' + query

    # Get a list of the pics for the current actor
    qlist = list()
    try:
        qlist = listdir(query_dir)
    except:
        query_dir = '\"' + query_dir + '\"'
        qlist = listdir(query_dir)
    pics = [f for f in qlist if isfile(join(query_dir, f))]

    for j, pic in enumerate(pics):
        file = query_dir + '/' + pic
        try:
            curr_img = imread(file)
        except:
            # failed to read image
            # sometimes this happens because the correct file type is not returned
            print('Image Read Failed.')
            continue

        detector = face.Detection()
        detector.minsize = 10
        detector.face_crop_margin = 16
        try:
            faces = detector.find_faces(curr_img)
        except:
            # Something in detector failed. Move on to next image
            print('Skipped', pic, ' - Something went wrong finding faces')

        if len(faces) > 1:
            print('Skipped', pic, ' - More than one face')
        elif len(faces) == 0:
            print('Skipped', pic, ' - No faces found')
        else:
            copy(file, raw_dir + char_actor + pic)
            imwrite(face_dir + char_actor + '_'.join(character.split()) +
                    '{}.jpg'.format(j + 1), faces[0].image, format='jpg')

    print('Flattening faces')
    # Flatten faces
    pwd = os.getcwd()
    align_path = '../../facenet/src/align/'
    try:
        os.chdir(align_path)
        subprocess.check_output(
            ['python', '-W', 'ignore', 'align_dataset_mtcnn.py', face_dir, flat_dir])
        os.chdir(pwd)
    except:
        os.chdir(pwd)