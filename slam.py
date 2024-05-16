import math
import numpy

from slam_association import associate_features
from slam_geometry import polar_to_cartesian, transform_segment, transform_point, \
                          point_to_point_distance, segment_endpoint_distance
from slam_localization import initialize_particles, compute_mean_and_covariance, update_localization, \
                              score_pose
from slam_segments import extract_segments
from slam_corners import detect_corners_from_segments, cluster_corners

class MappingEnvironment:
  def __init__(self, initial_position, segments):
    # Initialize the map segments
    self.map_segments = {}
    for segment in segments:
      self.map_segments[segment] = {}

    # Extract the map corners
    corners = detect_corners_from_segments(list(self.map_segments.keys()), angle_threshold = math.pi / 6)
    corners = cluster_corners(corners, eps=0.5, min_samples=1)
    self.map_corners = {}
    for corner in corners:
      self.map_corners[corner] = {}

    # Initialize the initial guess at the position
    self.robot_mean = initial_position
    self.robot_covariance = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, math.pi / 200.0]]
    self.poses, self.weights = initialize_particles(num_particles = 100, initial_pose = self.robot_mean, \
                                                    pose_noise_cov = self.robot_covariance)

  def lidar_update(self, lidar_data):
    # We first draw new samples based on the distribution
    self.poses, self.weights = update_localization(self.poses, self.weights, resample = True)

    # This is the pose from which lidar points are provided.
    reference_pose = self.robot_mean

    # convert the lidar data into cartesian data
    self.cartesian_points = polar_to_cartesian(lidar_data, reference_pose)

    # Extract the features
    seen_segments = extract_segments(self.cartesian_points, threshold = 10.0, min_points = 5)
    seen_corners = detect_corners_from_segments(seen_segments, angle_threshold = math.pi / 6)

    clustered_corners = cluster_corners(seen_corners, eps=0.5, min_samples=1)

    # Here we attempt to find a pose from which association seems more reasonable
    association_pose = self.robot_mean
    for pose in self.poses:
      association_pose = pose

      # Perform assocation to the map features
      self.segment_associations, self.new_segments, _ = associate_features( \
        new_features = [transform_segment(s, reference_pose, association_pose) for s in seen_segments], \
        map_features = self.map_segments, \
        scoring_function = segment_endpoint_distance, \
        threshold = 60.0)
      self.corner_associations, self.new_corners, _ = associate_features( \
        new_features = [transform_point(p, reference_pose, association_pose) for p in clustered_corners], \
        map_features = self.map_corners, \
        scoring_function = point_to_point_distance, \
        threshold = 20.0)

      # We need at least 3 features
      if len(self.segment_associations) + len(self.corner_associations) >= 3:
        break

    # Score each of the candidates
    for i, pose in enumerate(self.poses):
      self.weights[i] = 1.0
      self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                    feature_associations = self.segment_associations, \
                                    transform_function = transform_segment, \
                                    scoring_function = segment_endpoint_distance, \
                                    sensor_noise = 50.0)
      self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                    feature_associations = self.corner_associations, \
                                    transform_function = transform_point, \
                                    scoring_function = point_to_point_distance, \
                                    sensor_noise = 50.0)

    # Normalize the weights
    total_sum = sum(self.weights)
    if total_sum <= 0.0:
      for i, _ in enumerate(self.weights):
        self.weights[i] = 1 / len(self.weights)
        total_sum = sum(self.weights)
    self.weights = [w / total_sum for w in self.weights]

    # estimate best robot position from particles
    self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)

  def move_robot(self, move_distance, rotate_angle, distance_error, rotation_error):
    rng = numpy.random.default_rng()
    best_guess = (move_distance, rotate_angle)
    covariance = [[distance_error, 0.0], [0.0, rotation_error]]
    for i, p in enumerate(self.poses):
      noisy_guess = rng.multivariate_normal(best_guess, covariance)
      # noise to rotation is applied before and after move to model error accumulated during the move
      self.poses[i][2] += noisy_guess[1]
      self.poses[1][0] += noisy_guess[0] * numpy.cos(self.poses[i][2])
      self.poses[1][1] += noisy_guess[0] * numpy.sin(self.poses[i][2])
      self.poses[i][2] += noisy_guess[1]

    # estimate best robot position from particles
    self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)

if __name__ == '__main__':
  print("done")
