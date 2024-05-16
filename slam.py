import math
import numpy

from slam_association import associate_features
from slam_geometry import polar_to_cartesian, transform_segment, transform_point, \
                          point_to_point_distance, segment_endpoint_distance, normalize_angle
from slam_localization import initialize_particles, compute_mean_and_covariance, \
                              score_pose, normalize_weights
from slam_segments import extract_segments
from slam_corners import detect_corners_from_segments, cluster_corners

class MappingEnvironment:
  def __init__(self, initial_position, robot_covariance, num_points, segments):
    # Configuration options
    self.segment_distance_threshold = 10.0 # minimum distance between lidar points and the line segment
    self.segment_minimum_points = 5 # minimum points for segment to be considered valid
    self.corner_angle_threshold = numpy.deg2rad(30.0) # above this angle we look for corners

    self.corner_cluster_eps = 0.5 # eps for corner clustering
    self.corner_cluster_samples = 1 # minimum number of samples for corner clustering

    self.min_features = 3 # minimum number of features needed to localize

    self.segment_association_threshold = 50.0 # threshold for associating segments
    self.corner_association_threshold = 10.0 # threshold for corner association

    self.increase_pose_variance = 2.0 # if we did not localize grow position variance by this amount
    self.increase_angle_variance = numpy.deg2rad(2.0) # if we did nto localize grow angle variance by this amount

    self.scoring_sensor_noise = 50.0 # the amount of sensor noise we are expecting for scoring

    self.localized_distance_threshold = 15.0
    self.localized_angle_threshold = numpy.deg2rad(5.0)

    # Initialize the map segments
    self.map_segments = {}
    for segment in segments:
      self.map_segments[segment] = {}

    # Extract the map corners
    corners = detect_corners_from_segments(list(self.map_segments.keys()), \
                                           angle_threshold = self.corner_angle_threshold)
    corners = cluster_corners(corners, eps = self.corner_cluster_eps, min_samples = self.corner_cluster_samples)
    self.map_corners = {}
    for corner in corners:
      self.map_corners[corner] = {}

    # Initialize the initial guess at the position
    self.robot_mean = initial_position
    self.robot_covariance = robot_covariance
    self.poses, self.weights = initialize_particles(num_particles = num_points, initial_pose = self.robot_mean, \
                                                    pose_noise_cov = self.robot_covariance)

    # indicates whether we are localized
    self.localized = self.is_localized_from_covariance(self.robot_covariance)

  def is_localized_from_covariance(self, robot_covariance):
    return robot_covariance[0][0] < self.localized_distance_threshold and \
           robot_covariance[1][1] < self.localized_distance_threshold and \
           robot_covariance[2][2] < self.localized_angle_threshold

  def lidar_update(self, lidar_data):
    # We first draw new samples based on the distribution
    self.poses, self.weights = initialize_particles(num_particles = len(self.poses), \
                                                    initial_pose = self.robot_mean, \
                                                    pose_noise_cov = self.robot_covariance)

    # This is the pose from which lidar points are provided.
    reference_pose = self.robot_mean

    # convert the lidar data into cartesian data
    self.cartesian_points = polar_to_cartesian(lidar_data, reference_pose)

    # Extract the features
    seen_segments = extract_segments(self.cartesian_points, threshold = self.segment_distance_threshold, \
                                     min_points = self.segment_minimum_points)
    seen_corners = detect_corners_from_segments(seen_segments, angle_threshold = self.corner_angle_threshold)

    clustered_corners = cluster_corners(seen_corners, eps = self.corner_cluster_eps,
                                        min_samples = self.corner_cluster_samples)

    # Here we attempt to find a pose from which association seems more reasonable
    association_pose = self.robot_mean
    self.segment_associations = []
    self.new_segments = seen_segments
    self.corner_associations = []
    self.new_corners = clustered_corners
    if len(self.map_segments) + len(self.map_corners) >= self.min_features:
      for pose in [self.robot_mean] + self.poses + [self.robot_mean]:
        association_pose = pose

        # Perform assocation to the map features
        self.segment_associations, self.new_segments, _ = associate_features( \
          new_features = [transform_segment(s, reference_pose, association_pose) for s in seen_segments], \
          map_features = self.map_segments, \
          scoring_function = segment_endpoint_distance, \
          threshold = self.segment_association_threshold)
        self.corner_associations, self.new_corners, _ = associate_features( \
          new_features = [transform_point(p, reference_pose, association_pose) for p in clustered_corners], \
          map_features = self.map_corners, \
          scoring_function = point_to_point_distance, \
          threshold = self.corner_association_threshold)

        # We need some minimum feature set
        if len(self.segment_associations) + len(self.corner_associations) >= self.min_features:
          break

    # if we did not match enough features we can't score the candidates
    if len(self.segment_associations) + len(self.corner_associations) < self.min_features:
      # if the map has enough features, then we are likely in a bad position so increase the variance
      if len(self.map_segments) + len(self.map_corners) >= self.min_features:
        self.robot_covariance[0][0] += self.increase_pose_variance
        self.robot_covariance[1][1] += self.increase_pose_variance
        self.robot_covariance[2][2] += self.increase_angle_variance
        self.localized = False
      return

    # Score each of the candidates
    for i, pose in enumerate(self.poses):
      self.weights[i] = 1.0
      self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                    feature_associations = self.segment_associations, \
                                    transform_function = transform_segment, \
                                    scoring_function = segment_endpoint_distance, \
                                    sensor_noise = self.scoring_sensor_noise)
      self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                    feature_associations = self.corner_associations, \
                                    transform_function = transform_point, \
                                    scoring_function = point_to_point_distance, \
                                    sensor_noise = self.scoring_sensor_noise)

    # Normalize the weights
    self.weights = normalize_weights(self.weights)

    # estimate best robot position from particles
    self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)

    # update whether we are localized based on the covariance
    self.localized = self.is_localized_from_covariance(self.robot_covariance)

  def move_robot(self, move_distance, rotate_angle, distance_error, rotation_error):
    best_guess = [move_distance, rotate_angle]
    for i, p in enumerate(self.poses):
      move_error = numpy.random.normal(0, distance_error)
      rotation_error1 = numpy.random.normal(0, rotation_error)
      rotation_error2 = numpy.random.normal(0, rotation_error)

      new_pose = [p[0] + (move_distance + move_error) * numpy.cos(p[2] + rotate_angle + rotation_error1), \
                  p[1] + (move_distance + move_error) * numpy.sin(p[2] + rotate_angle + rotation_error1), \
                  normalize_angle(p[2] + rotate_angle + rotation_error2)]

      self.poses[i] = new_pose

    # estimate best robot position from particles
    self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)

    # update whether we are localized based on the covariance
    self.localized = self.is_localized_from_covariance(self.robot_covariance)

if __name__ == '__main__':
  print("done")
