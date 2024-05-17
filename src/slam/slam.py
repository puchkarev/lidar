import math
import numpy
import json

from slam.slam_association import *
from slam.slam_geometry import *
from slam.slam_features import *
from slam.slam_localization import *

class Slam:
  def __init__(self, initial_position, robot_covariance, num_points, segments):
    # Configuration options
    self.segment_distance_threshold = 10.0 # minimum distance between lidar points and the line segment
    self.segment_minimum_points = 5 # minimum points for segment to be considered valid
    self.corner_angle_threshold = numpy.deg2rad(30.0) # above this angle we look for corners

    self.min_features = 3 # minimum number of features needed to localize

    self.segment_association_threshold = 50.0 # threshold for associating segments
    self.corner_association_threshold = 10.0 # threshold for corner association

    self.increase_pose_variance = 2.0 # if we did not localize grow position variance by this amount
    self.increase_angle_variance = numpy.deg2rad(5.0) # if we did nto localize grow angle variance by this amount

    self.scoring_sensor_noise = 50.0 # the amount of sensor noise we are expecting for scoring

    self.localized_distance_threshold = 15.0
    self.localized_angle_threshold = numpy.deg2rad(5.0)

    # Initialize the map segments
    self.map_segments = segments

    # Extract the map corners
    corners = detect_corners_from_segments(list(self.map_segments), \
                                           angle_threshold = self.corner_angle_threshold)
    self.map_corners = corners

    # Initialize the initial guess at the position
    self.robot_mean = numpy.array(initial_position)
    self.robot_covariance = numpy.array(robot_covariance)
    self.reinitialize_particles(num_points)

    # indicates whether we are localized
    self.localized = self.is_localized_from_covariance(self.robot_covariance)

    self.cartesian_points = []
    self.segment_associations = []
    self.new_segments = []
    self.corner_associations = []
    self.new_corners = []

  # Determines if we are localized purely based on the covariance matrix
  def is_localized_from_covariance(self, robot_covariance):
    return robot_covariance[0][0] < self.localized_distance_threshold and \
           robot_covariance[1][1] < self.localized_distance_threshold and \
           robot_covariance[2][2] < self.localized_angle_threshold

  # Reinitializes the particles and may change the number of particles used.
  def reinitialize_particles(self, num_particles):
    self.poses, self.weights = initialize_particles(num_particles = num_particles, \
                                                    initial_pose = self.robot_mean, \
                                                    pose_noise_cov = self.robot_covariance)

  def add_noise(self, x_error, y_error, rotation_error):
    for i, p in enumerate(self.poses):
      new_pose = [p[0] + numpy.random.normal(0, x_error), \
                  p[1] + numpy.random.normal(0, y_error), \
                  p[2] + numpy.random.normal(0, rotation_error)]
      self.poses[i] = new_pose

    # estimate best robot position from particles
    self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)

    # update whether we are localized based on the covariance
    self.localized = self.is_localized_from_covariance(self.robot_covariance)

  def lidar_update(self, lidar_data):
    # We first draw new samples based on the distribution
    self.reinitialize_particles(len(self.poses))

    # This is the pose from which lidar points are provided.
    reference_pose = self.robot_mean

    # convert the lidar data into cartesian data
    self.cartesian_points = polar_to_cartesian(lidar_data, reference_pose)

    # Extract the features
    seen_segments = extract_segments(self.cartesian_points, threshold = self.segment_distance_threshold, \
                                     min_points = self.segment_minimum_points)
    seen_corners = detect_corners_from_segments(seen_segments, angle_threshold = self.corner_angle_threshold)

    # Here we attempt to find a pose from which association seems more reasonable
    association_pose = self.robot_mean
    self.segment_associations = []
    self.new_segments = seen_segments
    self.corner_associations = []
    self.new_corners = seen_corners
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
          new_features = [transform_point(p, reference_pose, association_pose) for p in seen_corners], \
          map_features = self.map_corners, \
          scoring_function = point_to_point_distance, \
          threshold = self.corner_association_threshold)

        # We need some minimum feature set
        if len(self.segment_associations) + len(self.corner_associations) >= self.min_features:
          break

    # if we did not match enough features we can't score the candidates
    if len(self.segment_associations) + len(self.corner_associations) < self.min_features:
      # if the map has enough features, then we are likely in a bad position so increase the variance
      self.add_noise(self.increase_pose_variance, self.increase_pose_variance, self.increase_angle_variance)
      self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)
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

    if max(self.weights) == 0.0:
      # The scoring of all the features resulted in all candidates being elminated, so need to increase variance
      self.add_noise(self.increase_pose_variance, self.increase_pose_variance, self.increase_angle_variance)
      self.weights = normalize_weights(self.weights)
      self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)
      self.localized = False
      return

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
    self.localized = self.localized and self.is_localized_from_covariance(self.robot_covariance)

  def to_json(self):
    return json.dumps({ \
      "localized": str(self.localized), \
      "robot_mean": self.robot_mean.tolist(), \
      "robot_covariance": self.robot_covariance.tolist(), \
      "segment_associations": list(self.segment_associations), \
      "new_segments": list(self.new_segments), \
      "corner_associations": list(self.corner_associations), \
      "new_corners": list(self.new_corners), \
      "map_segments": list(self.map_segments), \
      "map_corners": list(self.map_corners), \
    })

if __name__ == '__main__':
  print("done")
