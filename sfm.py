# projector is given by 6 points:
# rotation angle, two angles which gives rotation axis, three points of translation
# point is given by 3 coordinates

from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np

class SFM(object):
    '''Class solving reconstruction optimization problem

        Args:
            tracks: points tracks
            focal_distance: focal distance of the camera

    '''

    def __init__(self, tracks, focal_distance, num_cameras):
        self.tracks = tracks
        self.num_points = len(tracks)
        self.focal_distance = focal_distance
        self.num_cameras = num_cameras
        self.type = 'ml'
        
    def _vec_to_coords(self, X):
        points_coords = X[:3 * self.num_points].reshape(self.num_points, 3)
        cameras_coords = X[3 * self.num_points:].reshape(self.num_cameras, 6)
        return points_coords, cameras_coords
    
    def _coords_to_vec(self, points_coords, cameras_coords):
        vec = np.append(points_coords.flatten(), cameras_coords.flatten())
        return vec
    
    def _project(self, coords, projectors):
        '''Performing projection of a point to the given shots

            Args:
                coords: 3D coordinates of a point
                projectors: N x 6 matrix, each row corresponds to a projector.
                    Every projector is given by 6 numbers: first is rotation angle, second and third
                    represents spherical coordinates of a unit vector around which rotation is performed.
                    Coordinates 4 to 6 represents translation

        '''

        translation = projectors[:, 3:]
        xi, theta, phi = projectors[:, 0], projectors[:, 1], projectors[:, 2]
        e = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)]).T
        rotated = (np.inner(e, coords) * (1 - np.cos(xi)))[:, None] * e + \
                  np.cross(e, coords) * np.sin(xi[:, None]) + coords * np.cos(xi[:, None])
        translated = rotated + translation
        projected = self.focal_distance * translated[:, :2] / translated[:, -1][:, None]
        return projected
    
    def _projection_log_likelihood(self, observed, projected, sigma):
        logpdf = np.sum(norm.logpdf(projected, 
                                    loc=observed, 
                                    scale=np.repeat(sigma.reshape(-1, 1), 2, axis=1)))
        return logpdf
        
    def neg_log_likelihood(self, X):
        points_coords, cameras_coords = self._vec_to_coords(X)
        log_likelihood = 0
        for i, track in enumerate(self.tracks):
            projected = self._project(points_coords[i], cameras_coords[track[1]:track[1] + len(track[-1])])
            if self.type == 'ml':
                sigma = np.array([pt.size for pt in track[-1]])
            elif self.type == 'geometry':
                sigma = np.ones(len(track[-1]))
            else: 
                raise Exception('No such reconstruction type is avaliable')
            observed = np.array([pt.pt for pt in track[-1]])
            log_likelihood += self._projection_log_likelihood(observed, projected, sigma)
                
        return -log_likelihood
    
    def reconstruct(self, points_coords0, cameras_coords0, reconstruction_type='ml', kwargs={}):
        self.type = reconstruction_type
        vec0 = self._coords_to_vec(points_coords0, cameras_coords0)
        minimize(self.neg_log_likelihood, vec0, **kwargs)
        points_coords, cameras_coords = self._vec_to_coords(res.x)
        return points_coords, cameras_coords, res
