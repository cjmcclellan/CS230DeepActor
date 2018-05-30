import skvideo.io
import src.helper.helperfuncs as helper
import os
import numpy as np
import cv2

# this program will simply import videos with the same name but interative numbers and merge them

movie_path = '/home/connor/Downloads/'
movie_title = 'Solo_ AStarWarsStoryOfficialTrailer_5000_[0.2,0.3,0.3]'
movie_type = '.avi'



def main():
    movies = {}
    # first get all the movies with the movie title in it
    for root, dirs, files in os.walk(movie_path):
        for file in files:
            if movie_title in file:
                # get the increment number and add that to the movie list
                try:  # try the find Incr.  If it doesnt work, then you found the full movie
                    movie_i = helper.findIncr(file)
                    movies[movie_i] = file
                except:
                    pass

    max_num = len(movies.keys())
    if max_num == 0:
        print('Couldnt find any movies with :' + movie_title)

    # get the first video
    video = cv2.VideoCapture(os.path.join(movie_path, movies[0]))

    # setup the video writer
    full_movie_path = os.path.join(movie_path, movie_title + movie_type)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = video.get(cv2.cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(full_movie_path, fourcc, fps, (frame_width, frame_height))

    # now loop through all the movies and combine them
    for i in range(0, max_num):
        video = cv2.VideoCapture(os.path.join(movie_path, movies[i]))
        success, frame = video.read()
        while success:
            out.write(np.array(frame))
            success, frame = video.read()



        # new_part = skvideo.io.vread(os.path.join(movie_path, movies[i]))
        # full_movie = np.append(full_movie, new_part, axis=0)

    # now with the full movie, save it
    # skvideo.io.vwrite(os.path.join(movie_path, movie_title + movie_type), full_movie)

main()