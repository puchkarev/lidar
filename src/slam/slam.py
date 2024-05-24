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
    "SegmentAssociationThreshold": 20.0, \
    "CornerAssociationThreshold": 20.0, \
    "IncreasePoseVariance": 1.0, \
    "IncreaseAngleVariance": numpy.deg2rad(1.0), \
    "ScoringSensorNoise": 50.0, \
    "LocalizedDistanceThreshold": 15.0, \
    "LocalizedAngleThreshold": numpy.deg2rad(5.0), \
    "Localizing": 1, \
  }

def segment_to_segment_comparator(observed_segment, map_segment):
  """Provides a distance measure between an observed segment and a mapped segment"""
  # This measure is used to compare the segments for association and localization
  # Note that map segments are generally expected to be longer and observed segments
  # may be broken down in smaller chunks, so may represent sub-sections of wall segments

  # The actual value returned does not matter, as long as we tune the segment association
  # threshold to it. But the cost landscape needs to be such that as segments become more rotated
  # or shifted, the larger the distance measure should be.

  # This function just measures distances between endpoints of segments to the other segment
  # it heavily penalizes the disparity between the mapped vs observed segment lengths
  # 0.697s /54640 calls = 12.75622 us / call
  # return segment_endpoint_distance(observed_segment, map_segment)

  # This function just measures distances between the endpoints projected onto segments (if they
  # project, and the endpoints themselves). It does not penalize short segments and does ok.
  # 1.574s / 155,624 calls = 10.11412 us / call
  # return projected_segment_endpoint_distance(observed_segment, map_segment)

  # The implementation below specifically measures the properties we want:
  # rotation - measured as how much to shift the endpoint so that the segment lines up with the map segment
  # shift - how far to move the observed segment to line it up with the map segment
  # gap - how much gap remains between the observed and mapped segments
  # 2.346 s / 155,036 calls = 15.13196 us / call

  o_ang = numpy.abs(normalize_angle(segment_angle(observed_segment)))
  m_ang = numpy.abs(normalize_angle(segment_angle(map_segment)))

  o_len = point_to_point_distance(observed_segment[0], observed_segment[1])
  m_len = point_to_point_distance(map_segment[0], map_segment[1])

  o_mid = interpolate_point_on_segment(observed_segment, 0.5)
  m_mid = interpolate_point_on_segment(map_segment, 0.5)

  # how much the endpoint moves to fix the orientation around the midpoint, note that for the same rotation
  # this penalizes long segments more than short ones
  alignment_shift = numpy.sin(numpy.abs(normalize_angle(o_ang - m_ang))) * o_len * 0.5

  o_mid_projected = interpolate_point_on_segment(map_segment, projected_point_on_segment(o_mid, map_segment))

  # how much we need to shift the observed segment after rotation to have it lie on the same line as map segment
  shift_distance = point_to_point_distance(o_mid, o_mid_projected)

  # the remaining gap between the map segment and observed segment after rotation and shifting.
  gap_distance = max(point_to_point_distance(o_mid_projected, m_mid) - o_len * 0.5 - m_len * 0.5, 0.0)

  return math.sqrt(alignment_shift ** 2 + shift_distance ** 2 + gap_distance ** 2)

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

  def reinitialize_particles(self, num_particles, threshold = None):
    """Reinitializes the particles and may change the number of particles used."""
    if not threshold:
      self.poses, self.weights = initialize_particles(num_particles = num_particles, \
                                                      initial_pose = self.robot_mean, \
                                                      pose_noise_cov = self.robot_covariance)
      return

    # Remove all the elements outside the threshold
    scored_poses = [(p, w) for p, w in zip(self.poses, self.weights) if w >= threshold]

    # Sort the list so that we go from highest score to lowest
    scored_poses.sort(key = lambda x: x[1], reverse = True)

    # If we have too many elements, then remove the lowest scoring ones.
    if len(scored_poses) > num_particles:
      scored_poses = scored_poses[:num_particles]

    if len(scored_poses) == 0:
      # we have emptied the list, just repopulate
      self.poses, self.weights = initialize_particles(num_particles = num_particles, \
                                                      initial_pose = self.robot_mean, \
                                                      pose_noise_cov = self.robot_covariance)
      return

    # unpack the lists
    self.poses = [p for p, w in scored_poses]
    self.weights = [w for p, w in scored_poses]

    # if the list is now long enough, then just return it
    need = num_particles - len(self.poses)
    if need == 0:
      return

    # we need to grow the list, so grab new candidates
    new_poses, _ = initialize_particles(num_particles = need, \
                                        initial_pose = self.robot_mean, \
                                        pose_noise_cov = self.robot_covariance)
    self.poses = [s for l in [self.poses, new_poses] for s in l]

    # we have not scored the new candidates, so take the remainder of weight and distribute it evenly.
    remaining_weight = max(1.0 - sum(self.weights), 0.0)
    self.weights = [w for l in [self.weights, [remaining_weight / need] * need] for w in l]

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

    # Perform assocation to the map features
    if self.config["SegmentAssociationThreshold"] > 0.0:
      self.segment_associations, self.new_segments, _ = associate_features( \
        new_features = seen_segments, \
        map_features = self.map_segments, \
        scoring_function = segment_to_segment_comparator, \
        threshold = self.config["SegmentAssociationThreshold"])
    if self.config["CornerAssociationThreshold"] > 0.0:
      self.corner_associations, self.new_corners, _ = associate_features( \
        new_features = seen_corners, \
        map_features = self.map_corners, \
        scoring_function = point_to_point_distance, \
        threshold = self.config["CornerAssociationThreshold"])

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

    # Draw new samples based on the distribution
    resample_threshold = 1.0 / len(self.poses)
    self.reinitialize_particles(num_particles = len(self.poses), threshold = resample_threshold)

    # Score each of the candidates
    for i, pose in enumerate(self.poses):
      self.weights[i] = 1.0

      if len(segment_associations) > 0:
        previewed_segments = [(p, s[1]) for p, s in zip( \
          transform_segments([s[0] for s in segment_associations], association_pose, pose), \
          segment_associations)]
        self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                      feature_associations = previewed_segments, \
                                      transform_function = None, \
                                      scoring_function = segment_to_segment_comparator, \
                                      sensor_noise = self.config["ScoringSensorNoise"])

      if len(corner_associations) > 0:
        previewed_corners = [(p, s[1]) for p, s in zip( \
          transform_points([s[0] for s in corner_associations], association_pose, pose), \
          corner_associations)]
        self.weights[i] *= score_pose(pose = pose, reference_pose = association_pose, \
                                      feature_associations = previewed_corners, \
                                      transform_function = None, \
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

    # Validate the input
    if len(lidar_data) < 3:
      return "Not enough points"
    for l in lidar_data:
      if len(l) != 3:
        return "Invalid points"

    # Extract the features using mean pose as the reference pose
    self.lidar_points = lidar_data
    self.extract_features(lidar_data, self.robot_mean)

    # Associate the features to the map
    self.associate_features(self.reference_pose, self.seen_segments, self.seen_corners)

    # Perform localization
    if self.config["Localizing"] == 1:
      self.localize(self.association_pose, self.segment_associations, self.corner_associations)

  def move_robot(self, move_distance, distance_error):
    """Informs slam of the robot movement"""
    for i, p in enumerate(self.poses):
      noisy_distance = move_distance + numpy.random.normal(0, distance_error)
      new_pose = [p[0] + noisy_distance * numpy.cos(p[2]), \
                  p[1] + noisy_distance * numpy.sin(p[2]), \
                  p[2]]
      self.poses[i] = new_pose

    # estimate best robot position from particles
    self.robot_mean, self.robot_covariance = compute_mean_and_covariance(self.poses, self.weights)

    # update whether we are localized based on the covariance
    self.localized = self.localized and self.is_localized_from_covariance(self.robot_covariance)

  def rotate_robot(self, rotate_angle, rotation_error):
    """Informs slam of the robot movement"""
    for i, p in enumerate(self.poses):
      noisy_angle = rotate_angle + numpy.random.normal(0, rotation_error)
      new_pose = [p[0], p[1], normalize_angle(p[2] + noisy_angle)]
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
