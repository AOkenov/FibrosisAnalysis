import pandas as pd
import numpy as np
from scipy import spatial


class Spatial:
    def __init__(self):
        pass

    @staticmethod
    def distance(XA, XB, data=None):
        if isinstance(data, pd.DataFrame):
            XA = data[XA].to_numpy()
        return spatial.distance.cdist(XA, np.array([XB])).flatten()

    @staticmethod
    def angle(coords, centroid, data=None):
        ''' Returns angle [0, 2pi] between x-axis and radius vector from
            centroid to coords.
        '''
        if isinstance(data, pd.DataFrame):
            coords = data[coords].to_numpy()

        y = coords[:, 0] - centroid[0]
        x = coords[:, 1] - centroid[1]
        distance = Spatial.distance(coords, centroid)
        angle = np.sign(y) * np.arccos(x / distance)

        angle = np.where(angle < 0, 2*np.pi + angle, angle)
        angle = np.where((y == 0) & (x < 0), np.pi, angle)
        return angle, distance

    @staticmethod
    def orientation(orientation, data=None):
        '''Returns orientation angle between x-axis and major axis of object'''
        if isinstance(data, pd.DataFrame):
            orientation = data[orientation].to_numpy()
        return 0.5 * np.pi - orientation

    @staticmethod
    def align_orientation(data):
        angle = data['angle']
        orientation = data['orientation']
        orientation = np.where((angle < np.pi) & (orientation < angle) |
                               (angle > np.pi), np.pi + orientation, orientation)
        orientation = np.where((angle >= np.pi) & (orientation < angle),
                               np.pi + orientation, orientation) - angle
        orientation = np.degrees(np.where(data['angle groups'] > 0,
                                 orientation + angle, orientation))
        return orientation
