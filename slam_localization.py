import numpy
import math
import unittest

from slam_geometry import normalize_angle, transform_point, segment_to_segment_distance, point_to_segment_distance
from slam_segments import segment_angle

def initialize_particles(num_particles, initial_pose, \
                         pose_noise_cov = numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])):
  """
  Initialize particles with noise around an initial pose.

  Parameters:
  - num_particles: Number of particles to initialize.
  - initial_pose: Tuple (x, y, theta) representing the estimated initial pose.
  - pose_noise_cov: covariance matrix for noise.

  Returns:
  - List of poses and weights
  """
  poses = []
  weights = []
  rng = numpy.random.default_rng()
  for _ in range(num_particles):
    noisy_pose = rng.multivariate_normal(initial_pose, pose_noise_cov)
    poses.append(noisy_pose)
    weights.append(1.0 / num_particles)
  return poses, weights

def compute_mean_and_covariance(poses, weights):
  """Computes the mean position and covariance based on the poses and weights"""

  weighted_mean = [0.0, 0.0, 0.0]
  for pose, weight in zip(poses, weights):
    weighted_mean[0] += pose[0] * weight
    weighted_mean[1] += pose[1] * weight
    weighted_mean[2] += pose[2] * weight

  weighted_cov = numpy.zeros((3, 3))
  for pose, weight in zip(poses, weights):
    vec = pose - weighted_mean
    weighted_cov += weight * numpy.outer(vec, vec)

  return weighted_mean, weighted_cov

def score_pose(pose, reference_pose, feature_associations, transform_function, scoring_function, sensor_noise):
  """
  Computes the score/weight of a given position

  Parameters:
  - pose: a position that is to be scored
  - reference_pose: pose from which all the new features were believe to be observed
  - feature_associations: associations between new and mapped features
  - transform_function: function that consumes a new feature in reference pose and previews it in pose
  - scoring_function: function that provides a distance measure between the the features
  - sensor_noise: amount of sensor noise assumed to be present

  Returns:
  computed score/weight of the position
  """
  weight = 1.0
  for observed_feature, map_feature in feature_associations:
    corrected_feature = transform_function(observed_feature, reference_pose, pose)
    distance = scoring_function(corrected_feature, map_feature)
    likelihood = numpy.exp(-0.5 * ((distance ** 2) / sensor_noise**2))
    weight *= likelihood
  return weight

def update_localization(poses, weights, resample=True):
  """
  Update the localization based on poses and weights.

  Parameters:
  - poses (list of pose): Current set of poses representing the pose distribution.
  - weights (list of weights): Current set of weihts representing the pose distribution.
  - resample: if resample is set to true then new sample points are drawn

  Returns:
  - Updated list of poses, and updated list of weights.
  """
  total_weight = sum(weights)

  if total_weight <= 0.0:
    # Rescale the particle weights, and resample
    for i, _ in enumerate(weights):
      weights[i] = 1.0 / len(weights)

  # Normalize weights
  total_weight = sum(weights)
  while numpy.abs(total_weight - 1.0) > 1e-7:
    for i, w in enumerate(weights):
      weights[i] = w / total_weight
    total_weight = sum(weights)

  # See if we wanted to resample the poses
  if resample:
    mean, covariance = compute_mean_and_covariance(poses, weights)
    return initialize_particles(num_particles = len(poses), initial_pose=mean, pose_noise_cov=covariance)

  # Otherwise duplicate existing samples
  return numpy.random.choice(poses, size=len(poses), p=weights), weights

class TestBasicMethods(unittest.TestCase):
  def test_initialize_particles(self):
    poses, weights = initialize_particles(num_particles = 1000, initial_pose=(-2.0, -4.0, -6.0), \
                                          pose_noise_cov=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    mean, covariance = compute_mean_and_covariance(poses, weights)
    numpy.testing.assert_allclose(numpy.array(mean), \
                                  numpy.array([-2.0, -4.0, -6.0]), \
                                  rtol=0, atol=1.0)
    numpy.testing.assert_allclose(numpy.array(covariance), \
                                  numpy.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]), \
                                  rtol=0, atol=1.0)

if __name__ == '__main__':
  unittest.main()
