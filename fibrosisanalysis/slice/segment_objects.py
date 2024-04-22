from fibrosisanalysis.plots.point_density import PointDensity
from fibrosisanalysis.slice.objects_properties import ObjectsProperties


class SegmentObjects:
    def __init__(self):
        self.objects_props = None
        self.orientation = None
        self.structural_anisotrophy = None
        self.distribution_ellipses = None
        self.sorted_objects = None


class DistributionEllipse:
    def __init__(self, width=None, height=None, orientation=None):
        self.width = width
        self.height = height
        self.orientation = orientation


class SegmentObjectsBuilder:
    def __init__(self):
        self.segment_objects = SegmentObjects()

    def build(self, objects_props):
        res = StruturalAnisotrophy.point_density(objects_props.axis_ratio,
                                                 objects_props.orientation)
        axis_ratio, orientation, density = res
        
        structural_anisotrophy = StruturalAnisotrophy(axis_ratio, orientation,
                                                     density)
        self.segment_objects.structural_anisotrophy = structural_anisotrophy
        res = StruturalAnisotrophy.major_minor_axes(axis_ratio, orientation,
                                                    n_std)
        width, height, orientation = res

        self.segment_objects.distribution_ellipses = DistributionEllipse(
            width, height, orientation)

        res = PointDensity.rotated_ellipse(0.5 * width, 0.5 * height,
                                           orientation)
        radius_range, theta_range = res
        self.segment_objects.distribution_ellipse.radius_range = radius_range
        self.segment_objects.distribution_ellipse.theta_range = theta_range
