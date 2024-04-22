import numpy as np
from skimage import measure
from scipy import interpolate


class SplineEdge:
    """Edge class for representing a spline-defined edge.

    Attributes
    ----------
    full_nodes: np.ndarray
        Nodes of the spline drawn on the image
    nodes: np.ndarray
        Nodes of the spline that defines the edge.
    ordered_nodes: np.ndarray
        Ordered nodes of the spline that defines the edge.
    """

    def __init__(self, nodes=None):
        """Initialize the Edge instance.

        Parameters
        ----------
        nodes : np.ndarray
            An array containing the nodes of the spline.
        """
        self.nodes = nodes
        self.ordered_nodes = None
        self.full_nodes = None

    def load(self, path):
        """Load spline nodes from a file.

        Parameters
        ----------
        path : str
            The file path from which to load the spline nodes.
        """
        self.nodes = np.load(path)

    def sample_nodes(self, number_of_nodes):
        """Sample nodes from spline

        Parameters
        ----------
        number_of_nodes : int
            Number of desired nodes

        Returns
        -------
        np.ndarray
            Sampled nodes
        """
        nodes = self.compute(number_of_nodes)
        return nodes

    def compute(self, number_of_nodes=100_000):
        """Compute spline points along the edge.

        Parameters
        ----------
        number_of_nodes: int, optional
            The number of nodes to compute on the spline. Defaults to 100000.

        Returns
        -------
        np.ndarray
            Array containing computed spline points along the edge.
        """
        # Duplicate the first node to close the spline
        x, y = np.append(self.nodes, [self.nodes[0]], axis=0).T

        # Perform spline interpolation
        (t, c, k), u = interpolate.splprep([x, y], s=0)
        c = np.asarray(c).T
        bspline = interpolate.BSpline(t, c, k)

        # Evaluate the spline at specified points
        u_new = np.linspace(0, 1, number_of_nodes + 1)
        nodes = bspline(u_new).astype(int)[:-1]
        # Exclude the duplicated nodes
        _, unique_index = np.unique(nodes, axis=0, return_index=True)

        nodes = nodes[np.sort(unique_index)]
        return nodes

    def clip_boundaries(self, nodes, shape):
        """Clip coordinates to stay within mask boundaries.

        Parameters
        ----------
        nodes: np.ndarray
            Edge nodes
        shape: list
            Shape of the image.

        Returns
        -------
        np.ndarray
            Clipped coordinates.
        """
        nodes[:, 0][nodes[:, 0] >= shape[0] - 2] = shape[0] - 2
        nodes[:, 0][nodes[:, 0] <= 1] = 2
        nodes[:, 1][nodes[:, 1] >= shape[1] - 2] = shape[1] - 2
        nodes[:, 1][nodes[:, 1] <= 1] = 2
        return nodes

    def check_connectivity(self, nodes, shape):
        """Check connectivity of the edge.

        Parameters
        ----------
        nodes: np.ndarray
            Edge nodes.
        shape: list
            Shape of the image.

        Returns
        -------
        bool
            True if connectivity is maintained.
        """
        image = np.ones(shape, dtype=int)
        image[tuple(nodes.T)] = 0
        _, num = measure.label(image, connectivity=1, return_num=True)
        return num == 2

    @property
    def direction(self):
        """Find direction vector

        Returns
        -------
        np.ndarray
            Vector arrays
        """
        direction = (np.roll(self.ordered_nodes, -1, axis=0)
                     - self.ordered_nodes)

        return direction
