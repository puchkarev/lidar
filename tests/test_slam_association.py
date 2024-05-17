import math
import unittest

from slam.slam_association import *

class TestBasicMethods(unittest.TestCase):
  def test_associations(self):
    def corner_distance(new_corner, map_corner):
      return math.hypot(new_corner[1] - map_corner[1], new_corner[0] - map_corner[0])

    associations, unassociated_new, unassociated_map = \
      associate_features(new_features = [(1.10, 2.1), (5.05, 5.05)], \
      map_features = [(1.10, 2.1), (10.05, 5.05)], \
      scoring_function = corner_distance, \
      threshold = 1.0)

    self.assertEqual(associations, [((1.1, 2.1), (1.1, 2.1))])
    self.assertEqual(unassociated_new, [(5.05, 5.05)])
    self.assertEqual(unassociated_map, [(10.05, 5.05)])

if __name__ == '__main__':
  unittest.main()
