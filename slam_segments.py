import math
import numpy
import unittest

from slam_geometry import point_to_line_distance

def extract_segments(cartesian_points, threshold = 1.0, min_points = 2):
  """
  Extracts line segments formed by the cartesian_points, the cartesian_points are expected to be ordered,
  as generated from a lidar scan.

  Parameters:
  - cartesian_points: list of points (x, y) in a sequential list
  - threshold: how close each point needs to be to the line segment
  - min_points: minimum number of points to consider a segment

  Returns:
  - List of tuples which express line segments as a tuple of two end points
  """
  segments = []

  last = len(cartesian_points) - 1
  si = -1
  ei = 0
  while ei <= last:
    # measure the maximum distance to the intermediate points
    distance = 0
    for ix in range(si + 1, ei):
      distance = max(distance, point_to_line_distance(cartesian_points[ix], \
                                                      cartesian_points[si], cartesian_points[ei]))
      if distance >= threshold:
        break

    # See if we can keep this point, or start a segment with it.
    if distance < threshold:
      if si < 0:
        si = ei
        points = 0
      ei += 1
      continue

    # segment must have at least two distinct points to be valid.
    if ei - si >= min_points:
      segments.append((cartesian_points[si], cartesian_points[ei - 1]))

    # the start of next segment can start and the end of the previous one.
    si = ei - 1

  # if we finished without terminating the segment, consider terminating it
  if si >= 0:
    if last + 1 - si >= min_points:
      segments.append((cartesian_points[si], cartesian_points[last]))
  return segments

def segment_angle(segment):
  return numpy.arctan2(segment[1][1] - segment[0][1], segment[1][0] - segment[0][0])

class TestBasicMethods(unittest.TestCase):
  def test_extract_segments(self):
    numpy.testing.assert_allclose(numpy.array(extract_segments( \
      cartesian_points = [(10, 10), (10, 11), (10, 12), (10, 13), (9, 11), (8, 9)], \
      threshold = 1.0)),
      numpy.array([((10.0, 10.0), (10, 13)), ((10, 13), (8, 9))]))

  def test_segment_angle(self):
    self.assertEqual(segment_angle(((0.0, 5.0), (1.0, 5.0))), 0.0)
    self.assertEqual(segment_angle(((0.0, 10.0), (1.0, 10.0))), 0.0)

    self.assertEqual(segment_angle(((0.0, 5.0), (0.0, 6.0))), math.pi / 2)
    self.assertEqual(segment_angle(((1.0, 5.0), (1.0, 6.0))), math.pi / 2)

    self.assertEqual(segment_angle(((0.0, 0.0), (1.0, 1.0))), math.pi / 4)
    self.assertEqual(segment_angle(((0.0, 0.0), (1.0, -1.0))), -math.pi / 4)

if __name__ == '__main__':
  unittest.main()

