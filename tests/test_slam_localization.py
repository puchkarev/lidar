import numpy
import unittest

from slam.slam_localization import *

class TestBasicMethods(unittest.TestCase):
  def test_initialize_particles(self):
    poses, weights = initialize_particles(num_particles = 1000, initial_pose=(-2.0, -4.0, -1.0), \
                                          pose_noise_cov=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    mean, covariance = compute_mean_and_covariance(poses, weights)
    numpy.testing.assert_allclose(numpy.array(mean), \
                                  numpy.array([-2.0, -4.0, -1.0]), \
                                  rtol=0, atol=1.0)
    numpy.testing.assert_allclose(numpy.array(covariance), \
                                  numpy.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]), \
                                  rtol=0, atol=1.0)

if __name__ == '__main__':
  unittest.main()
