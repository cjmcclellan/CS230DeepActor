#### This program will be used for creating bounding boxes around characters in a full video



import cv2
import numpy as np
import src.FaceDetectors.FaceDetector as faceDetector
from CS230DeepActor.src.triplet_loss.TNet_file import *
import skvideo.io
import time


save_model_path = '../../triplet_loss_data/triplet_model/tip_mod_1.pt'
pretrained_model = '../../models/pretrained_facenet/20170512-110547'

solo_face_path = '/home/connor/Documents/Stanford_Classes/CS230/CS230DeepActor/train_data/FaceID/Solo:_A_Star_Wars_Story_2018_movie/face/Han_Solo--Alden_Ehrenreich/Han_Solo12.jpg'
solo_actors = '/home/connor/Documents/Stanford_Classes/CS230/CS230DeepActor/train_data/FaceID/Solo:_A_Star_Wars_Story_2018_movie/face/all_actors'
solo_trailer = '/home/connor/Downloads/Solo_ AStarWarsStoryOfficialTrailer.mp4'


class video_encoder:

    def __init__(self, pathtovideo, FaceDetectModel):
        self.FaceDetectModel = FaceDetectModel
        self.pathtovideo = pathtovideo
        self.video = cv2.VideoCapture(self.pathtovideo)


    # this function will complie random samples of images from the video as negative face examples
    def randomVideoSampling(self, num_samples, save_path):
        # get the image size from the face detection model
        sample_name = save_path
        image_size = self.FaceDetectModel.image_size
        num_frames = self.video.get(cv2.cv2.CAP_PROP_FRAME_COUNT)
        frame_width = self.video.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.video.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT)
        # now randomly choose the indexes of the frames to sample from
        i_frames = list(np.random.randint(0, high=num_frames, size=(num_samples,)))
        i_frames.sort()
        success = True
        count = 0
        # keep on going with new frames and while there are still frames to sample
        while success and len(i_frames) > 0:
            success, frame = self.video.read() # read in the frame
            if count == i_frames[0]:
                # now randomly choose a x and y coordinate
                x = np.random.randint(0, high=(frame_width - image_size))
                y = np.random.randint(0, high=(frame_height - image_size))
                # take the sample
                sample = frame[y:y+image_size, x:x+image_size]
                sample_name = self.__incrementFile(sample_name)
                cv2.imwrite(sample_name, sample)
                i_frames.pop(0)
            count += 1 # increment the count


    # this will be used to incrment a file path name by increaing the number before the file type
    def __incrementFile(self, path):
        period = path.find('.')
        number = int(path[period-1:period])
        return path[:period - 1] + str(number + 1) + path[period:]


    def runDetector(self, frame_freq, savepath, wanted_faces, threshold = 10e8, print_freq = 50):
        frames = []
        count = 1
        success = True
        # get sample frame from video
        # frame_width = int(self.video.get(3))
        # frame_height = int(self.video.get(4))
        # fps = self.video.get(cv2.cv2.CAP_PROP_FPS)
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # self.out = cv2.VideoWriter(savepath, fourcc, fps, (frame_width, frame_height))
        print('about to start looping through the frames')
        # now loop through every frame
        timezero = time.time() # get the inital time
        while success and count < 2000:
            success, frame = self.video.read()
            # if the frame is of the frame frequency, run the detector
            if count%frame_freq == 0:

                # run the face through the detection model, giving the bounding box frame output
                frame = self.FaceDetectModel.id_face(frame, threshold,\
                                        wanted_faces= wanted_faces)

                #skvideo.io.vwrite(savepath, np.array(frames))
                #self.out.write(np.array(frames))
            count += 1
            frames.append(frame)
            if count%print_freq == 0:
                print('the count is ' + str(count))
                print('it took ' + str(time.time() - timezero))
                timezero = time.time()

        # now save the video
        skvideo.io.vwrite(savepath, np.array(frames))


####### Controls over the NEt ###########
DoId = True  # if set to true, the model will just detect the face (not Id the actor) to speed up the run


####### some hyperparameters ############

# for the multi-task CNN
threshold_base = np.array([0.2, 0.3, 0.3]).reshape((1,3))  # three steps's threshold (default is 0.6, 0.7, 0.7)
minsize = 5  # minimum size of face (default is 20)
# create a bunch of thresholds
ranged = np.linspace(1, 3, 10).reshape(10,1)
thresholds = list(np.multiply(threshold_base, ranged))
thresholds = list(threshold_base)  #jsut do one for now

#########################################


################# Testing code #############
#movie = mvpy.VideoFileClip(solo_trailer).subclip(50,60)
a = 5



# pick which faces and the corredsponding color
wanted_faces = {'Han_Solo.jpg': (0, 255, 0), 'Lando_Calrissian.jpg': (255, 0, 0) \
            , 'Qira.jpg': (0, 0, 255), 'Beckett.jpg': (255, 255, 255)}
for threshold in thresholds:
    # make threshold a list
    threshold = list(threshold)
    # create the face detector model
    face_detector_model = faceDetector.FaceDetector(save_model_path, pretrained_model, ref_faces_path=solo_actors\
                                                    , DoId = DoId, threshold = threshold, minsize = minsize)
    print('created face detect model')
    # create the video_encoder model
    Solo = video_encoder('/home/connor/Downloads/Solo_ AStarWarsStoryOfficialTrailer.mp4', face_detector_model)
    print('created video encoder')

    ############# create random samples ##########
    #Solo.randomVideoSampling(40, solo_actors + '/random_sample0.jpg' )

    # now run the detector
    Solo.runDetector(10, '/home/connor/Downloads/Solo_ AStarWarsStoryOfficialTrailer_'\
                  + str(minsize) + '_'  + str(threshold) + '_.mp4', wanted_faces, 11000)
