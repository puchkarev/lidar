import math
import numpy
import unittest

from sklearn.cluster import DBSCAN

from slam_geometry import normalize_angle, line_segment_intersection
from slam_lines import line_angle
from slam_segments import segment_angle

def detect_corners_from_lines(lines, angle_threshold = math.pi/6):
  """ Detect corners from a list of lines based on the angle between them. """
  corners = []
  for i in range(len(lines)):
    j = (i + 1) % len(lines)
    angle = numpy.abs(normalize_angle(line_angle(lines[i]) - line_angle(lines[j])))
    if angle > angle_threshold:
      x = (lines[j][1] - lines[i][1]) / (lines[i][0] - lines[j][0])
      y = lines[i][0] * x + lines[i][1]
      corners.append((x, y))
  return corners

def detect_corners_from_segments(segments, angle_threshold = math.pi/6):
  """ Detect corners from a list of segments based on the angle between them. """
  corners = []
  for i in range(len(segments)):
    j = (i + 1) % len(segments)
    angle = numpy.abs(normalize_angle(segment_angle(segments[i]) - segment_angle(segments[j])))
    if angle > angle_threshold:
      corners.append(line_segment_intersection(segments[i], segments[j]))
  return corners

def cluster_corners(corners, eps=0.5, min_samples=2):
  """
  Cluster corner points using DBSCAN.

  Parameters:
  - corners: List of corner points (x, y).
  - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
  - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

  Returns:
  - clusters: A list of clustered corner coordinates.
  """

  # Extract corner points
  corner_points = numpy.array([(x, y) for (x, y) in corners])
  if (len(corner_points) == 0):
    return []

  # DBSCAN clustering
  clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(corner_points)
  labels = clustering.labels_

  # Group corners by cluster label
  clustered_corners = []
  for label in numpy.unique(labels):
    if label != -1:  # Ignore noise points
      cluster_members = corner_points[labels == label]
      cluster_center = cluster_members.mean(axis=0)
      clustered_corners.append(tuple(cluster_center))

  return clustered_corners

class TestBasicMethods(unittest.TestCase):
  def test_detect_corners_from_lines(self):
    numpy.testing.assert_allclose(numpy.array(detect_corners_from_lines( \
      lines = [(1, 0), (0, 10), (-1, 20)], \
      angle_threshold = math.pi / 6)),
      numpy.array([(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]))

  def test_detect_corners_from_segments(self):
    numpy.testing.assert_allclose(numpy.array(detect_corners_from_segments( \
      segments = [((0, 0), (1, 1)), ((0, 10), (10, 10)), ((0, 20), (10, 10))], \
      angle_threshold = math.pi / 6)),
      numpy.array([(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]))

  def test_cluster_corners(self):
    numpy.testing.assert_allclose(numpy.array(cluster_corners( \
      corners = [(1.0, 2.0), (1.1, 2.1), (5.0, 5.0), (10.0, 10.0), (1.2, 2.2), (5.1, 5.1)], \
      eps = 0.5, min_samples = 1)),
      numpy.array([(1.0999999999999999, 2.1), (5.05, 5.05), (10.0, 10.0)]))

if __name__ == '__main__':
  unittest.main()
