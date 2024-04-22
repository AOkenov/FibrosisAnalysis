import numpy as np
from scipy import stats
from matplotlib import patches
from fibrosisanalysis.plots.polar_plots import PolarPlots


class DistributionEllipse:
    def __init__(self):
        self.type_name = None
        self.width = None
        self.height = None
        self.orientation = None

    @property
    def anisotropy(self):
        return self.width / self.height


class DistributionEllipseBuilder:
    def __init__(self):
        self.distribution_ellipse = DistributionEllipse()

    def build(self, objects_props, n_std=2., ellipse_type='error'):
        """Build a distribution ellipse from objects properties.

        Parameters
        ----------
        objects_props : ObjectsProperties
            Objects properties.
        n_std : float, optional
            Number of standard deviations.
        ellipse_type : str, optional
            Type of the ellipse ('error' or 'confidence').
        """
        r = np.concatenate((objects_props.axis_ratio,
                            objects_props.axis_ratio))
        theta = np.concatenate((objects_props.orientation,
                                objects_props.orientation + np.pi))

        if len(r) < 10:
            self.distribution_ellipse = DistributionEllipse()
            self.distribution_ellipse.type_name = ellipse_type
            self.distribution_ellipse.width = np.nan
            self.distribution_ellipse.height = np.nan
            self.distribution_ellipse.orientation = np.nan
            self.distribution_ellipse.full_radius = np.nan
            self.distribution_ellipse.full_theta = np.nan
            return self.distribution_ellipse

        x, y = PolarPlots.polar_to_cartesian(r, theta)
        cov = self.covariance(x, y)
        eig_vals, eig_vec = self.sorted_eigs(cov)

        if ellipse_type.lower() == 'error':
            width, height = self.error_ellipse(eig_vals, n_std)

        if ellipse_type.lower() == 'confidence':
            width, height = self.confidence_ellipse(eig_vals, n_std)

        theta = np.arctan2(* eig_vec[::-1, 0])
        # theta = np.arctan2(* eig_vec[:, 0])

        self.distribution_ellipse = DistributionEllipse()
        self.distribution_ellipse.type_name = ellipse_type
        self.distribution_ellipse.width = width
        self.distribution_ellipse.height = height
        self.distribution_ellipse.orientation = theta

        res = PolarPlots.rotated_ellipse(0.5 * width, 0.5 * height, theta)

        self.distribution_ellipse.full_radius = res[0]
        self.distribution_ellipse.full_theta = res[1]
        return self.distribution_ellipse

    def error_ellipse(self, eig_vals, n_std):
        """Calculate the width and height of an error ellipse.

        Parameters
        ----------
        eig_vals : np.ndarray
            Eigenvalues of the covariance matrix.
        n_std : float
            Number of standard deviations.

        Returns
        -------
        tuple
            Width and height of the error ellipse.
        """
        width, height = 2 * n_std * np.sqrt(eig_vals)
        return width, height

    def confidence_ellipse(self, eig_vals, n_std):
        """Calculate the width and height of a confidence ellipse.

        Parameters
        ----------
        eig_vals : np.ndarray
            Eigenvalues of the covariance matrix.
        n_std : float
            Number of standard deviations.

        Returns
        -------
        tuple
            Width and height of the confidence ellipse.
        """
        # Confidence level
        q = 2 * stats.norm.cdf(n_std) - 1
        r2 = stats.chi2.ppf(q, 2)
        width, height = 2 * np.sqrt(eig_vals * r2)
        return width, height

    def covariance(self, x, y):
        """Calculate the covariance matrix between two variables.

        Parameters
        ----------
        x : np.ndarray
            Input data for x-coordinate.
        y : np.ndarray
            Input data for y-coordinate.

        Returns
        -------
        np.ndarray
            Covariance matrix between x and y.
        """
        return np.cov(np.vstack([x, y]), rowvar=True)

    def sorted_eigs(self, cov):
        """Sort eigenvalues and eigenvectors in descending order.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix.

        Returns
        -------
        tuple
            Sorted eigenvalues and eigenvectors.
        """
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def patch(self, x, y, width, height, angle, **kwargs):
        """Generate a matplotlib patch representing an ellipse.

        Parameters
        ----------
        x : np.ndarray
            Input data for x-coordinate.
        y : np.ndarray
            Input data for y-coordinate.
        width : float
            Width of the ellipse.
        height : float
            Height of the ellipse.
        angle : float
            Rotation angle of the ellipse.
        **kwargs
            Additional keyword arguments for matplotlib patches.

        Returns
        -------
        patches.Ellipse
            Matplotlib patch representing the ellipse.
        """
        xy = np.vstack([x, y]).mean(axis=1)
        patch = patches.Ellipse(xy=xy, width=width, height=height,
                                angle=np.degrees(angle), **kwargs)
        return patch
