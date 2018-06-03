import cv2 as cv
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

def create_orb_features(files, dir_basis, dir_images, masks=None, dir_masks=None):
    for i, file in enumerate(files):

        filename = dir_images + file
        maskname = dir_masks + masks[i] if dir_masks is not None else None
        
        if os.path.exists(filename[:-3] + 'pckl'):
            continue
        
        mask = (255 - cv.imread(maskname)) if maskname is not None else 1

        img = mask * cv.imread(filename)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        orb = cv.xfeatures2d.SIFT_create()

        kp, descriptors = orb.detectAndCompute(gray, None)

        temp = [(point.pt, point.size, point.angle, point.response, point.octave, 
            point.class_id, desc) for point, desc in zip(kp, descriptors)]

        with open(filename[:-3] + 'pckl', 'wb') as file:
            pickle.dump(temp, file)

def deserialize_keypoint(point):
    return cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], 
                        _response=point[3], _octave=point[4], _class_id=point[5]), point[6]


def get_keypoints(image_file):
    kp1, desc1 = [], []
    with open(image_file[:-3] + 'pckl', 'rb') as file:
        for entry in pickle.load(file):
            k, d = deserialize_keypoint(entry)
            kp1.append(k)
            desc1.append(d)       
    return kp1, np.array(desc1)