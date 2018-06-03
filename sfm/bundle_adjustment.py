from scipy.sparse import lil_matrix
import numpy as np
import time
from scipy.optimize import least_squares
import os
import sys

def prepare_data(triangulations, tracks, dir_proj):
	
	tracks_successfull = [tracks[i] for i in triangulations]

	points_2d = []
	points_ind = []
	camera_ind = []
	sigmas = []

	for i, track in enumerate(tracks_successfull):
	    for j, kp in enumerate(track[-1]):
	        points_ind.append(i)
	        camera_ind.append(track[1] + j)
	        points_2d.append(kp.pt)
	        sigmas.append(kp.size)
	        
	points_2d = np.array(points_2d)
	point_indices = np.array(points_ind, dtype=np.uint32)
	camera_indices = np.array(camera_ind, dtype=np.uint32)
	sigmas = np.array(sigmas)

	points_3d = np.array([triangulations[i] for i in triangulations])
	n_points = len(points_3d)

	projectors = []
	for name in sorted(os.listdir(dir_proj)):
	    projectors.append(np.loadtxt(dir_proj + name, skiprows=1))
	projectors = np.array(projectors)
	n_cameras = len(projectors)

	return points_2d, point_indices, projectors, camera_indices, sigmas, points_3d, n_points, n_cameras


def project(points, projectors, camera_indices, point_indices):
    projected = np.matmul(projectors[camera_indices], 
                          np.hstack((points[point_indices], 
                                     np.ones(len(point_indices)).reshape(-1, 1)))[:, :, np.newaxis])
    projected = np.squeeze(projected)
    projected = projected[:, :2] / projected[:, -1][:, None]
    return projected


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, projectors, sigmas):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    points_3d = params.reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], projectors[camera_indices], camera_indices, point_indices)
    return ((points_proj - points_2d) / np.sqrt(sigmas)[:, None]).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)

    for s in range(3):
        A[2 * i, point_indices * 3 + s] = 1
        A[2 * i + 1, point_indices * 3 + s] = 1
    
    return A


def optimize(
	x0,
	n_cameras,
	n_points,
	camera_indices,
	point_indices,
	points_2d,
	projectors,
	sigmas,
	jac_sparsity
	):
	
	t0 = time.time()
	res = least_squares(fun, x0, jac_sparsity=jac_sparsity, verbose=2, x_scale='jac', ftol=1e-8, method='trf',
                    	args=(n_cameras, n_points, camera_indices, point_indices, points_2d, projectors, sigmas))
	t1 = time.time()
	return res