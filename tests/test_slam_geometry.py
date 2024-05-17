import math
import numpy
import unittest

from slam.slam_geometry import *

class TestBasicMethods(unittest.TestCase):
  def test_normalize_angle(self):
    self.assertEqual(normalize_angle(0.0), 0.0)
    self.assertEqual(normalize_angle(math.pi), math.pi)
    self.assertEqual(normalize_angle(-math.pi), math.pi)

    self.assertEqual(normalize_angle(math.pi / 2), math.pi / 2)
    self.assertEqual(normalize_angle(-math.pi / 2), -math.pi / 2)

    self.assertEqual(normalize_angle(math.pi * 2), 0.0)
    self.assertEqual(normalize_angle(-math.pi * 2), 0.0)
    self.assertEqual(normalize_angle(-math.pi * 4), 0.0)
    self.assertEqual(normalize_angle(math.pi * 4), 0.0)
    self.assertEqual(normalize_angle(-math.pi * 6), 0.0)
    self.assertEqual(normalize_angle(math.pi * 6), 0.0)
    self.assertEqual(normalize_angle(-math.pi * 8), 0.0)
    self.assertEqual(normalize_angle(math.pi * 8), 0.0)

    self.assertEqual(normalize_angle(-math.pi * 3), math.pi)
    self.assertEqual(normalize_angle(math.pi * 3), math.pi)
    self.assertEqual(normalize_angle(-math.pi * 5), math.pi)
    self.assertEqual(normalize_angle(math.pi * 5), math.pi)
    self.assertEqual(normalize_angle(-math.pi * 7), math.pi)
    self.assertEqual(normalize_angle(math.pi * 7), math.pi)
    self.assertEqual(normalize_angle(-math.pi * 9), math.pi)
    self.assertEqual(normalize_angle(math.pi * 9), math.pi)

  def test_point_to_line_distance(self):
    self.assertEqual(point_to_line_distance((0, 0), ((1, 1), (1, 1))), 0.0)
    self.assertEqual(point_to_line_distance((0, 0), ((1, 1), (2, 1))), 1.0)
    self.assertEqual(point_to_line_distance((0, 0), ((1, 1), (1, 2))), 1.0)
    self.assertEqual(point_to_line_distance((2, 0), ((1, 1), (1, 2))), 1.0)
    self.assertEqual(point_to_line_distance((0, 2), ((1, 1), (2, 1))), 1.0)
    self.assertEqual(point_to_line_distance((3, 0), ((1, 1), (1, 2))), 2.0)
    self.assertEqual(point_to_line_distance((0, 3), ((1, 1), (2, 1))), 2.0)

  def test_point_to_segment_distance(self):
    self.assertAlmostEqual(point_to_segment_distance((0, 0), ((1, 1), (1, 1))), 1.4142135623730951)
    self.assertAlmostEqual(point_to_segment_distance((0, 0), ((1, 1), (2, 1))), 1.4142135623730951)
    self.assertAlmostEqual(point_to_segment_distance((0, 0), ((1, 1), (1, 2))), 1.4142135623730951)
    self.assertAlmostEqual(point_to_segment_distance((2, 0), ((1, 1), (1, 2))), 1.4142135623730951)
    self.assertAlmostEqual(point_to_segment_distance((0, 2), ((1, 1), (2, 1))), 1.4142135623730951)
    self.assertAlmostEqual(point_to_segment_distance((3, 0), ((1, 1), (1, 2))), 2.23606797749979)
    self.assertAlmostEqual(point_to_segment_distance((0, 3), ((1, 1), (2, 1))), 2.23606797749979)

  def test_segment_to_segment_distance(self):
    self.assertAlmostEqual(segment_to_segment_distance(((0, 0), (0, 0)), ((1, 1), (1, 1))), 1.4142135623730951)
    self.assertAlmostEqual(segment_to_segment_distance(((0, 0), (1, 0)), ((1, 1), (1, 1))), 1.0)
    self.assertAlmostEqual(segment_to_segment_distance(((0, 0), (1, 0)), ((1, 1), (2, 1))), 1.0)
    self.assertAlmostEqual(segment_to_segment_distance(((0, 0), (1, 0)), ((1, 1), (2, 0))), 0.7071067811865475)
    self.assertAlmostEqual(segment_to_segment_distance(((0, 0), (1, 0)), ((1, 0), (2, 0))), 0.0)
    self.assertAlmostEqual(segment_to_segment_distance(((0, 0), (1, 1)), ((1, 0), (0, 1))), 0.0)

  def test_line_line_intersection(self):
    self.assertEqual(line_line_intersection(line1 = ((0, 0), (1, 0)), \
                                            line2 = ((1, 1), (1, 0))), \
                     (1, 0))

    self.assertEqual(line_line_intersection(line1 = ((0, 0), (1, 0)), \
                                            line2 = ((1, 2), (1, 1))), \
                     (1, 0))

  def test_polar_to_cartesian(self):
    numpy.testing.assert_allclose(numpy.array(polar_to_cartesian( \
      lidar_data = [(1, 0, 10), (1, 90, 20), (1, 180, 30)], \
      pose = (1.0, 2.0, math.pi / 2))), \
      numpy.array([(1.0, 12.0), (-19.0, 2.0), (1.0, -28.0)]))

  def test_transform_point(self):
    numpy.testing.assert_allclose(numpy.array(transform_point(point = (10.0, 15.0), \
      original_pose = (3.0, 1.0, math.pi / 2), \
      new_pose = (2.0, 4.0, math.pi))), \
      numpy.array((-12.0, 11.0)))

  def test_transform_segment(self):
    numpy.testing.assert_allclose(numpy.array(transform_segment( \
      segment = ((10.0, 15.0), (15.0, 20.0)), \
      original_pose = (3.0, 1.0, math.pi / 2), \
      new_pose = (2.0, 4.0, math.pi))), \
      numpy.array([[-12.0, 11.0], [-17.0, 16.0]]))

  def test_score_distance(self):
    self.assertAlmostEqual(score_distance(0.0, sensor_noise = 1.0), 1.0)
    self.assertAlmostEqual(score_distance(1.0, sensor_noise = 1.0), 0.6065306597126334)
    self.assertAlmostEqual(score_distance(2.0, sensor_noise = 1.0), 0.13533528323661267)
    self.assertAlmostEqual(score_distance(3.0, sensor_noise = 1.0), 0.011108996538242308)

  def test_segment_angle(self):
    self.assertEqual(segment_angle(((0.0, 5.0), (1.0, 5.0))), 0.0)
    self.assertEqual(segment_angle(((0.0, 10.0), (1.0, 10.0))), 0.0)

    self.assertEqual(segment_angle(((0.0, 5.0), (0.0, 6.0))), math.pi / 2)
    self.assertEqual(segment_angle(((1.0, 5.0), (1.0, 6.0))), math.pi / 2)

    self.assertEqual(segment_angle(((0.0, 0.0), (1.0, 1.0))), math.pi / 4)
    self.assertEqual(segment_angle(((0.0, 0.0), (1.0, -1.0))), -math.pi / 4)

if __name__ == '__main__':
  unittest.main()
