import math
import numpy
import unittest

from slam_geometry import normalize_angle, line_line_intersection
from slam_segments import segment_angle

def detect_corners_from_segments(segments, angle_threshold = math.pi/6):
  """ Detect corners from a list of segments based on the angle between them. """
  corners = []
  for i in range(len(segments)):
    j = (i + 1) % len(segments)
    angle = numpy.abs(normalize_angle(segment_angle(segments[i]) - segment_angle(segments[j])))
    if angle > angle_threshold:
      corners.append(line_line_intersection(segments[i], segments[j]))
  return corners

class TestBasicMethods(unittest.TestCase):
  def test_detect_corners_from_segments(self):
    numpy.testing.assert_allclose(numpy.array(detect_corners_from_segments( \
      segments = [((0, 0), (1, 1)), ((0, 10), (10, 10)), ((0, 20), (10, 10))], \
      angle_threshold = math.pi / 6)),
      numpy.array([(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]))

if __name__ == '__main__':
  unittest.main()
