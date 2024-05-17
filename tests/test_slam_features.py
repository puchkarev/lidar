import math
import numpy
import unittest

from slam.slam_features import *

class TestBasicMethods(unittest.TestCase):
  def test_extract_segments(self):
    numpy.testing.assert_allclose(numpy.array(extract_segments( \
      cartesian_points = [(10, 10), (10, 11), (10, 12), (10, 13), (9, 11), (8, 9)], \
      threshold = 1.0)),
      numpy.array([((10.0, 10.0), (10, 13)), ((10, 13), (8, 9))]))

  def test_detect_corners_from_segments(self):
    numpy.testing.assert_allclose(numpy.array(detect_corners_from_segments( \
      segments = [((0, 0), (1, 1)), ((0, 10), (10, 10)), ((0, 20), (10, 10))], \
      angle_threshold = math.pi / 6)),
      numpy.array([(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]))

if __name__ == '__main__':
  unittest.main()

