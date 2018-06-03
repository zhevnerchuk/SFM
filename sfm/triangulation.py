import cv2 as cv
import numpy as np

def triangulate_points(files, tracks, dir_proj):

    succesfull_triangulations = {}

    for i, track in enumerate(tracks):
        
        if len(track[-1]) < 3:
            continue
        
        image_coords = np.array([kp.pt for kp in tracks[i][-1]])
        image_indices = np.arange(tracks[i][1], tracks[i][1] + len(tracks[i][-1]))
        
        triangulations = []
        
        for j in range(len(image_indices) - 1):
            
            P = np.loadtxt(dir_proj + files[image_indices[j]].split('.')[0] + '.txt', skiprows=1) 
            P_dash = np.loadtxt(dir_proj + files[image_indices[j + 1]].split('.')[0] + '.txt', skiprows=1)
            
            world_point = cv.triangulatePoints(
                P, P_dash,
                image_coords[j].reshape(2, -1), image_coords[j + 1].reshape(2, -1)
            )
            
            world_point = cv.convertPointsFromHomogeneous(world_point.reshape(-1, 4))
            triangulations.append(world_point)
            
        triangulation = np.array(triangulations).mean(axis=0)
        world_point = cv.convertPointsToHomogeneous(triangulation).reshape(-1)
        
        for j in range(len(image_indices)):
            
            P = np.loadtxt(dir_proj + files[image_indices[j]].split('.')[0] + '.txt', skiprows=1)
            
            rep = cv.convertPointsFromHomogeneous(
                (P @ world_point).reshape(-1, 3)
            ).reshape(-1)
            
            if np.linalg.norm(rep - image_coords[j]) > 10:
                break
        else:
            succesfull_triangulations[i] = triangulation.reshape(-1)

    return succesfull_triangulations