import numpy as np
from scipy import spatial
from skimage import filters


class Distance:
    """Distance class providing methods for computing distances between
    coordinates.

    This class includes methods for finding the shortest distance between
    points using a KD-tree and calculating Euclidean distances between pairs
    of coordinates.

    Methods
    -------
    shortest(coord_1, coord_2):
        Finds the shortest distance and corresponding indices between
        two sets of coordinates using a KD-tree.

    between(coord_1, coord_2):
        Computes Euclidean distances between pairs of coordinates.

    Attributes
    ----------
    None
    """

    def __init__(self):
        """Initialize an instance of the Distance class.
        """
        pass

    @staticmethod
    def closest_neighbour(coord_1, coord_2):
        """Finds the closest neighbour between two sets of coordinates.
        
        Parameters
        ----------
            coord_1 : np.ndarray
                Array of coordinates.
            coord_2 : np.ndarray
                Array of coordinates for which closest point in coord_1
                is found.
        """
        _, inds = Distance.shortest(coord_1, coord_2)
        return coord_1[inds]

    @staticmethod
    def shortest(coord_1, coord_2, distance_upper_bound=np.inf):
        """Finds the shortest distance and corresponding indices between
        two sets of coordinates.

        Uses a KD-tree for efficient nearest neighbor search.

        Parameters
        ----------
        coord_1 : np.ndarray
            Array of coordinates for the KD-tree.
        coord_2 : np.ndarray
            Array of coordinates for which distances are computed.

        Returns
        -------
        tuple
            Tuple containing the shortest distances and corresponding
            indices for each coord_2.
        """
        tree = spatial.KDTree(coord_1)
        distance, inds = tree.query(coord_2, workers=4, 
                                    distance_upper_bound=distance_upper_bound)
        return distance, inds

    @staticmethod
    def between(coord_1, coord_2):
        """Computes Euclidean distances between pairs of coordinates.

        Parameters
        ----------
        coord_1 : np.ndarray
            Array of coordinates.
        coord_2 : np.ndarray
            Array of coordinates.

        Returns
        -------
        np.ndarray
            Array containing the Euclidean distances between pairs of
            coordinates.
        """
        return np.sqrt(np.sum((coord_1 - coord_2) ** 2, axis=1))

    @staticmethod
    def distance_matrix(coords):
        """Compute sparse distance matrix

        Parameters
        ----------
        coords : np.ndarray
            Array if coordiantes

        Returns
        -------
        coo_matrix
            Sparse coordinate matrix
        """
        tree = spatial.KDTree(coords)
        res = tree.sparse_distance_matrix(coords, output_type='coo_matrix')
        return res
