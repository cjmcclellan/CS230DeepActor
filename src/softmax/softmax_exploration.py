# Messing around with softmax classifier

import sys
sys.path.append('../')
import argparse
import facenet.src.train_softmax as sm
import os

# gets default args
args = sm.parse_arguments([])
# prints the default args for the softmax
print(args)

facenet_model_checkpoint = os.path.abspath("../../models/pretrained_facenet/20170512-110547")
models_base_dir = os.path.abspath('../../models/softmax/ridiculous6/')
logs_base_dir = os.path.abspath('../../logs/softmax/ridiculous6/')
data_dir = os.path.abspath('../../train_data/FaceID/The_Ridiculous_6_2015_movie/flattened/')

args.logs_base_dir = logs_base_dir
args.models_base_dir = models_base_dir
args.pretrained_model = facenet_model_checkpoint
args.data_dir = data_dir
args.model_def = 'facenet.src.models.inception_resnet_v1'

sm.main(args)


