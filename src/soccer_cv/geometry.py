import cv2
import numpy as np

class ViewTransformer:
    """
    Uses homography to transform 2D points from one plane to another
        :param source: coordinates in original image (detected key points)
        :param target: coordinates in the destination view (2D layout of field)
    """
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)   # float32 required by OpenCV for homography
        target = target.astype(np.float32)
        self.matrix, _ = cv2.findHomography(source, target)  # computes homography matrix `m`; linear transformation from source plane to target plane

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.matrix)   # apply homography matrix to the input points; transform
        return points.reshape(-1, 2).astype(np.float32)          # reurn as flat array of (x, y) coordinates 
