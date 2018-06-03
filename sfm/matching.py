import cv2 as cv
import numpy as np
from sfm.features import get_keypoints

def find_correspondences(files, K):
    
    FLANN_INDEX_LSH = 6
    index_params= dict(
        algorithm = FLANN_INDEX_LSH,
        table_number = 6, # 12
        key_size = 12,     # 20
        multi_probe_level = 1
    ) #2
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    fname0 = files[0]
    kp0, desc0 = get_keypoints(fname0)
    pts_correspondence = {}

    for i in range(len(files))[1:]:

        fname1 = files[i]
        kp1, desc1 = get_keypoints(fname1)

        matches = flann.knnMatch(desc0,desc1,k=2)

        matchesMask = [[0,0] for i in range(len(matches))]

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for j, p in enumerate(matches):
            try:
                m, n = p
            except:
                continue
            if m.distance < 0.8 * n.distance:
                matchesMask[j] = [1, 0]
                good.append(m)
                pts2.append(kp1[m.trainIdx])
                pts1.append(kp0[m.queryIdx])

        pts1_coord = np.int32([kp.pt for kp in pts1])
        pts2_coord = np.int32([kp.pt for kp in pts2])

        F, mask = cv.findFundamentalMat(pts1_coord, pts2_coord, cv.FM_RANSAC)
        
        pts1 = np.array(pts1)[mask.ravel() == 1]
        pts2 = np.array(pts2)[mask.ravel() == 1]
        
        pts1_coord = pts1_coord[mask.ravel() == 1]
        pts2_coord = pts2_coord[mask.ravel() == 1]
        
        E, mask = cv.findEssentialMat(pts1_coord, pts2_coord, K)
        T = np.eye(4)
        _, R, t, _ = cv.recoverPose(E, pts1_coord, pts2_coord, K)
        T[:3, :3] = R
        T[:-1, -1] = t.reshape(-1)
        
        pts_correspondence[(i-1, i)] = (pts1.copy(), pts2.copy(), T.copy())

        kp0, desc0 = kp1, desc1
        
    return pts_correspondence


