import sys
sys.path.append('../facenet/')
import numpy as np
det_path = '../facenet/src/align/'
model = np.load(det_path + 'det1.npy', encoding='latin1')
a = 4