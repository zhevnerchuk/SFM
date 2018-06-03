import cv2 as cv
import numpy as np

def build_tracks(files, pts_correspondence):
	
	tracks = [[True, 0, [pt1, pt2]] for pt1, pt2 in zip(*pts_correspondence[(0, 1)][:-1])]

	for i in range(1, len(files) - 1):
    
	    if i % 50 == 0:
	        print(i)
	        print(len(tracks))
	        
	    taken_points = np.zeros_like(pts_correspondence[(i, i + 1)][0], dtype=bool)
	    
	    for track in tracks:
	        
	        has_match = False
	        kp = track[-1][-1]
	        
	        if track[0] == False:
	            continue

	        for j, kp1 in enumerate(pts_correspondence[(i, i + 1)][0]):
	            if np.linalg.norm(np.array(kp.pt) - np.array(kp1.pt)) < kp.size:
	                taken_points[j] = True
	                if has_match:
	                    continue
	                has_match = True
	                track[-1].append(pts_correspondence[(i, i + 1)][1][j])
	        
	        track[0] = has_match
	    
	    for kp_index in np.where(taken_points == False)[0]:
	        tracks.append([True, i, [
	            pts_correspondence[(i, i + 1)][0][kp_index],
	            pts_correspondence[(i, i + 1)][1][kp_index]
	        ]])

	return tracks