import json
import math
import numpy
import types

from slam.slam_association import *
from slam.slam_geometry import *
from slam.slam_features import *
from slam.slam_localization import *

def DefaultSlamConfig():
  return { \
    "SegmentDistanceThreshold": 10.0, \
    "SegmentMinimumPoints": 5, \
    "CornerAngleThreshold": numpy.deg2rad(30.0), \
    "MinFeaturesToLocalize": 3, \
    "SegmentAssociationThreshold": 50.0, \
    "CornerAssociationThreshold": 10.0, \
    "IncreasePoseVariance": 2.0, \
    "IncreaseAngleVariance": numpy.deg2rad(5.0), \
    "ScoringSensorNoise": 50.0, \
    "LocalizedDistanceThreshold": 15.0, \
    "LocalizedAngleThreshold": numpy.deg2rad(5.0), \
    "Localizing": 1, \
  }

class Slam:
  def __init__(self, initial_position, robot_covariance, num_points, segments, config = DefaultSlamConfig()):
    # Initial state of the system
    self.set_config(config)
    self.set_map(segments)
    self.set_position(initial_position, robot_covariance, num_points)

    # Things which will be computed and populated later
    self.lidar_points = []

    self.reference_pose = self.robot_mean
    self.cartesian_points = []
    self.seen_segments = []
    self.seen_corners = []

    self.association_pose = self.robot_mean
    self.segment_associations = []
    self.new_segments = []
    self.corner_associations = []
    self.new_corners = []

  def set_config(self, config):
    """Sets the new configuration of the system. Returns a non empty string of errors if any"""
    default_config = DefaultSlamConfig()

    errors = []
    for key in default_config.keys():
      if key not in config:
        errors.append(str(key) + " missing from config")
      elif type(config[key]) is not type(default_config[key]):
        error.append(str(key) + " should be of type " + str(type(default_config[key])) + \
                     " but is of type " + str(type(config[key])))

    for key in config.keys():
      if key not in default_config:
        errors.append(str(key) + " should not be in the config")

    if len(errors) > 0:
      return ', '.join(errors)

    self.config = config
    return ""

  def set_map(self, segments):
    """Defines a new map"""
    self.map_segments = segments
    self.map_corners = detect_corners_from_segments(self.map_segments, \
                                                    angle_threshold = self.config["CornerAngleThreshold"])

  def set_position(self, initial_position, robot_covariance, num_points):
    """Overwrites the current position"""
    self.robot_mean = numpy.array(initial_position)
    self.robot_covariance = numpy.array(robot_covariance)
    self.localized = self.is_localized_from_covariance(self.robot_covariance)
    self.reinitialize_particles(num_points)

  def is_localized_from_covariance(self, robot_covariance):
    """Determines if we are localized purely based on the covariance matrix"""
    return robot_covariance[0][0] < self.config["LocalizedDistanceThreshold"] and \
           robot_covariance[1][1] < self.config["LocalizedDistanceThreshold"] and \
           robot_covariance[2][2] < self.config["LocalizedAngleThreshold"]

  def reinitialize_particles(self, num_particles):
    """Reinitializes the particles and may change the number of particles used."""
    self.poses, self.weights = initialize_particles(num_particles = num_particles, \
                                                    initial_pose = self.robot_mean, \
                                                    pose_noise_cov = self.robot_covariance)

  def add_noise(self, x_error, y_error, rotation_error):
    """Adds specific amount of noise to the pose estimation"""
    for i, p in enumerate(self.poses):
      new_pose = [p[0] + numpy.random.normal(0, x_error), \
                  p[1] + numpy.random.normal(0, y_error), \
                  p[2] + numpy.random.normal(0, rotation_error)]
      self.poses[i] = new_pose

    # estimate best robot position from particles
    self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)

    # update whether we are localized based on the covariance
    self.localized = self.is_localized_from_covariance(self.robot_covariance)

  def extract_features(self, lidar_data, reference_pose):
    """Extracts the features from the lidar data using the reference pose"""

    # This is the pose from which lidar points are provided.
    self.reference_pose = reference_pose

    # convert the lidar data into cartesian data
    self.cartesian_points = polar_to_cartesian(lidar_data, reference_pose)

    # Extract the features
    self.seen_segments = extract_segments(self.cartesian_points, \
                                          threshold = self.config["SegmentDistanceThreshold"], \
                                          min_points = self.config["SegmentMinimumPoints"])
    self.seen_corners = detect_corners_from_segments(self.seen_segments, \
                                                     angle_threshold = self.config["CornerAngleThreshold"])

  def associate_features(self, reference_pose, seen_segments, seen_corners):
    """Associate the features to the reference map"""

    # We first assume that no association is possibe to handle the case with
    # an incomplete map.
    self.association_pose = reference_pose
    self.segment_associations = []
    self.new_segments = seen_segments
    self.corner_associations = []
    self.new_corners = seen_corners

    if len(self.map_segments) + len(self.map_corners) < self.config["MinFeaturesToLocalize"]:
      return

    # sort the poses by weight (and add the mean as the best possible candidate)
    scored_poses = [(self.robot_mean, 1.0)] + [a for a in zip(self.poses, self.weights)]
    scored_poses.sort(key = lambda x: x[1], reverse = True)

    for pose, weight in scored_poses:
      # we are not interested in any candidates which have been previously eliminated
      if weight <= 0.0:
        continue

      # Perform assocation to the map features
      self.association_pose = numpy.array(pose)
      self.segment_associations, self.new_segments, _ = associate_features( \
        new_features = [transform_segment(s, reference_pose, pose) for s in seen_segments], \
        map_features = self.map_segments, \
        scoring_function = segment_endpoint_distance, \
        threshold = self.config["SegmentAssociationThreshold"])
      self.corner_associations, self.new_corners, _ = associate_features( \
        new_features = [transform_point(p, reference_pose, pose) for p in seen_corners], \
        map_features = self.map_corners, \
        scoring_function = point_to_point_distance, \
         threshold = self.config["CornerAssociationThreshold"])

      # Once we match enough features we can exit
      if len(self.segment_associations) + len(self.corner_associations) >= self.config["MinFeaturesToLocalize"]:
        break

  def localize(self, association_pose, segment_associations, corner_associations):
    # if we did not match enough features we can't score the candidates
    if len(segment_associations) + len(corner_associations) < self.config["MinFeaturesToLocalize"]:
      # if the map has enough features, then we are likely in a bad position so increase the variance
      if len(self.map_segments) + len(self.map_corners) >= self.config["MinFeaturesToLocalize"]:
        self.add_noise(self.config["IncreasePoseVariance"], self.config["IncreasePoseVariance"], \
                       self.config["IncreaseAngleVariance"])
        self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)
        self.localized = False
      return

    # Draw new samples based on the distribution (maybe should keep the best previous candidates,
    # or possibly just resample the candidates which scored 0's.
    self.reinitialize_particles(len(self.poses))

    # Score each of the candidates
    for i, pose in enumerate(self.poses):
      self.weights[i] = 1.0
      self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                    feature_associations = segment_associations, \
                                    transform_function = transform_segment, \
                                    scoring_function = segment_endpoint_distance, \
                                    sensor_noise = self.config["ScoringSensorNoise"])
      self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                    feature_associations = corner_associations, \
                                    transform_function = transform_point, \
                                    scoring_function = point_to_point_distance, \
                                    sensor_noise = self.config["ScoringSensorNoise"])

    # If all the candidates are bad, we increase the variance.
    if max(self.weights) == 0.0:
      self.add_noise(self.config["IncreasePoseVariance"], self.config["IncreasePoseVariance"], \
                     self.config["IncreaseAngleVariance"])
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

  def lidar_update(self, lidar_data):
    """Informs slam of a more recent lidar update"""
    # Extract the features using mean pose as the reference pose
    self.lidar_points = lidar_data
    self.extract_features(lidar_data, self.robot_mean)

    # Associate the features to the map
    self.associate_features(self.reference_pose, self.seen_segments, self.seen_corners)

    # Perform localization
    if self.config["Localizing"] == 1:
      self.localize(self.association_pose, self.segment_associations, self.corner_associations)

  def move_robot(self, move_distance, rotate_angle, distance_error, rotation_error):
    """Informs slam of the robot movement"""
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
    """Convers the current system measurements to json"""
    return json.dumps({ \
      "localized": str(self.localized), \
      "robot_mean": self.robot_mean.tolist(), \
      "robot_covariance": self.robot_covariance.tolist(), \
      "lidar_points": list(self.lidar_points), \
      "cartesian_points": list(self.cartesian_points), \
      "reference_pose": self.reference_pose.tolist(), \
      "seen_segments": list(self.seen_segments), \
      "seen_corners": list(self.seen_segments), \
      "association_pose": self.association_pose.tolist(), \
      "segment_associations": list(self.segment_associations), \
      "new_segments": list(self.new_segments), \
      "corner_associations": list(self.corner_associations), \
      "new_corners": list(self.new_corners), \
      "map_segments": list(self.map_segments), \
      "map_corners": list(self.map_corners), \
      "config": self.config, \
    })

if __name__ == '__main__':
  print("done")
