# Script to help build a database of images for characters. Asks the user how they want it to be built.
# Required to run from this folder

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
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


# Not my code, response to a question on stack exchange
# https://codereview.stackexchange.com/questions/25417/is-there-a-better-way-to-make-a-function-silent-on-need
# Simply suppresses the output for some functions
class NoStdStreams(object):
    def __init__(self,stdout = None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()



# Get IMDB instance
ia = IMDb()

# Get movie search

movie_name = input('Search for the movie the database will be based on. (enter title): ')
with NoStdStreams():
    results = ia.search_movie(movie_name)

# Print the movie title search results

print('\nSearch Results:\n')
print('Title, Year, Kind')
print('-----------------')
for i, result in enumerate(results):
    if i > 9:
        break
    print(str(i+1) + '. ' + result.data['title'] + ', (' + str(result.data['year']) + '), ' + result.data['kind'])

val = input('\nEnter a title number (non-matching characters will exit program): ')

# Select a title

try:
    val = int(val)
    if val in range(1, min(len(results), 10)):
        movie = ia.get_movie(results[val-1].movieID)
    else:
        print('Please enter a value between 1 and {}'.format(min(len(results), 10)))
        raise Exception

except:
    quit('Quit signal given. Program Exited')

#
# Select number of actors from the selected title
#

print('\nTop 30 Billed Actors:\n')
print('Cast, Character')
print('---------------')
for i, castmember in enumerate(movie['cast']):
    if i > 29:
        break
    print(str(i+1) + '. ' + castmember['name'] + ', ' + castmember.notes)

num_actors = input('\nHow many characters do you want to include? (Taken from top to bottom): ')

try:
    num_actors = int(num_actors)
    if num_actors not in range(1, min(len(movie['cast']), 30)):
        print('Please enter a value between 1 and {}'.format(min(movie['cast'], 30)))
        raise Exception

except:
    quit('Quit signal given. Program Exited')

# Select number of images to download
num_images = input('\nHow many images per actors? (1-100): ')

try:
    num_images = int(num_images)
    if num_images not in range(1, 100):
        print('Please enter a value between 1 and 100')
        raise Exception

except:
    quit('Quit signal given. Program Exited')


#
# Start Building the database
#

output_root = '../train_data/FaceID/'
movie_dir = output_root + '_'.join(movie['title'].split() + [str(movie['year'])] + [str(movie['kind'])]) + '/'
downloads_dir = movie_dir + 'downloaded/'
raw_dir = movie_dir + 'raw/'
face_dir = movie_dir + 'face/'
flat_dir = movie_dir + 'flattened/'

for i in range(num_actors):

    character = movie['cast'][i].notes.split('/')[0]
    char_actor = '--'.join(['_'.join(character.split()), '_'.join(movie['cast'][i]['name'].split())]) + '/'

    # Create directories for other outputs
    if not os.path.exists(raw_dir + char_actor):
        os.makedirs(raw_dir + char_actor)

    if not os.path.exists(face_dir + char_actor):
        os.makedirs(face_dir + char_actor)

    if not os.path.exists(flat_dir + char_actor):
        os.makedirs(flat_dir + char_actor)

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)

    query = ' '.join([character, movie['title'], movie['cast'][i]['name'], str(movie['year'])])
    query_dir = downloads_dir + '\"{}\"'.format(query)

    print('\nDownloading {} images for {}'.format(str(num_images), movie['cast'][i]['name']))
    subprocess.check_output(
        ['googleimagesdownload', '-k', '\"{}\"'.format(query), '-l', str(num_images), '-o', downloads_dir, '-f',
         'jpg', '-s','medium'])
    print('Downloads finished')
    print('Processing images and finding faces')

    # Get a list of the pics for the current actor
    pics = [f for f in listdir(query_dir) if isfile(join(query_dir, f))]

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
        faces = detector.find_faces(curr_img)
        if len(faces) > 1:
            print('Skipped', pic, ' - More than one face')
        elif len(faces) == 0:
            print('Skipped', pic, ' - No faces found')
        else:
            copy(file, raw_dir + char_actor + pic)
            imwrite(face_dir + char_actor + '_'.join(character.split()) +
                    '{}.jpg'.format(j+1), faces[0].image, format='jpg')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        print('Flattening faces')
        # Flatten faces
        pwd = os.getcwd()
        align_path = '../facenet/src/align/'
        try:
            os.chdir(align_path)
            subprocess.check_output(['python', 'align_dataset_mtcnn.py', '../../' + face_dir, '../../' + flat_dir])
            os.chdir(pwd)
        except:
            os.chdir(pwd)

print('Database creation complete!')
