import numpy as np
from fibrosisanalysis.plots.point_density import PointDensity


class PolarPlots:
    """
    Class representing structural anisotropy analysis.

    Attributes:
    ----------
    distribution_ellipses : DistributionEllipses
        Instance of DistributionEllipses class.
    """

    def __init__(self):
        pass

    @staticmethod
    def polar_to_cartesian(radius, theta):
        """
        Converts polar coordinates to Cartesian coordinates.

        Parameters:
        ----------
        radius : float 
            The radius or distance from the origin.
        theta : float 
            The angle in radians.

        Returns:
        -------
        tuple
            A tuple containing the Cartesian coordinates (x, y).
        """
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

    @staticmethod
    def cartesian_to_polar(x, y):
        """
        Convert Cartesian coordinates to polar coordinates.

        Parameters:
        ----------
        x : float
            The x-coordinate.
        y : float
            The y-coordinate.

        Returns:
        -------
            tuple: A tuple containing the radius and theta (angle) in radians.
        """
        radius = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return radius, theta

    @staticmethod
    def sort_by_density(r, theta):
        """
        Calculate the point density. All points are duplicated and the angle
        is shifted by pi radians to cover 360 degrees.

        Parameters:
        ----------
        r : array
            The radius or distance from the origin.
        theta : array
            The angle in radians.

        Returns:
        -------
            tuple: A tuple containing the radius, theta (angle) in radians,
                and density.
        """
        r, theta, density = PolarPlots.point_density(r, theta)
        idx = np.argsort(density)
        return r[idx], theta[idx], density[idx]

    @staticmethod
    def point_density(r, theta):
        """
        Calculate the point density. All points are duplicated and the angle
        is shifted by pi radians to cover 360 degrees.

        Parameters:
        ----------
        r : array
            The radius or distance from the origin.
        theta : array
            The angle in radians.

        Returns:
        -------
            tuple: A tuple containing the radius, theta (angle) in radians,
                and density.
        """
        r = np.concatenate((r, r))
        theta = np.concatenate((theta, theta + np.pi))
        x, y = PolarPlots.polar_to_cartesian(r, theta)
        density = PointDensity.gaussian_kde(x, y)
        return r, theta, density

    @staticmethod
    def sorting_idx(r, theta):
        """Sorting index.

        Parameters
        ----------
        r : array
            The radius or distance from the origin.
        theta : array
            The angle in radians.

        Returns
        -------
        array
            Sorting index."""
        _, _, density = PolarPlots.point_density(r, theta)
        idx = np.argsort(density)
        idx = idx[idx < len(r)]
        return idx, density[idx]

    @staticmethod
    def rotated_ellipse(a, b, alpha):
        """Rotated ellipse function.

        Parameters
        ----------
        a : float
            Major axis.
        b : float
            Minor axis.
        theta : float
            Angle of rotation.
        alpha : float
            Angle of ellipse."""
        theta = np.linspace(0, 2 * np.pi, 100)
        r = (a * b) / np.sqrt((b * np.cos(theta - alpha))**2
                              + (a * np.sin(theta - alpha))**2)
        return r, theta
