# Simulated Robot
import numpy

from slam.slam_geometry import *
from slam.slam_kinematics import *

class SimulatedRobot:
  def __init__(self, initial_position, segments, \
               noise_std_dev = 1.0, angle_noise_std_dev = 0.5, \
               sensor_noise = 0.1, sensor_angle_std_dev = 0.1, \
               num_points = 100, field_of_view = numpy.deg2rad(360.0), max_distance = 100, \
               lidar_offset = [0.0, 0.0, numpy.deg2rad(0.0)], \
               wheel_base = 50.0, robot_contour = [(100.0, 75.0), (100.0, -75.0), (-100.0, -75.0), (-100.0, 75.0)]):
    self.position = numpy.array(initial_position) # Actual position of the robot
    self.segments = segments # The representation of the world as a series of segmnets
    self.noise_std_dev = noise_std_dev  # Noise standard deviation for movement
    self.angle_noise_std_dev = angle_noise_std_dev  # Noise standard deviation for rotation
    self.sensor_noise = sensor_noise # Noise standard deviation for lidar distance
    self.sensor_angle_std_dev = sensor_angle_std_dev # Noise standard deviation for lidar angle
    self.num_points = num_points  # Number of LIDAR points
    self.field_of_view = field_of_view  # Field of view for LIDAR in degrees
    self.max_distance = max_distance  # Maximum sensing distance for LIDAR
    self.lidar_offset = lidar_offset # Position of the lidar relative to center of the robot
    self.wheel_base = wheel_base # Distance between robot wheels
    self.contour = robot_contour # Contour of the robot

    # starting motor speeds
    self.left = 0.0
    self.right = 0.0

  def set_speed(self, left, right):
    """
    Sets the left and right motor speeds for a differential drive robot
    """
    self.left = left
    self.right = right

  def advance_time(self, dt):
    """
    Advances the time for a differential drive robot
    """
    turn1, dist2, turn3 = get_turn_move_turn_from_differential_drive(vL = self.left, vR = self.right, base = self.wheel_base, dt = dt)
    if turn1 != 0.0:
      self.rotate(turn1)
    if dist2 != 0.0:
      self.move(dist2)
    if turn3 != 0.0:
      self.rotate(turn3)

    return turn1, dist2, turn3

  def move(self, distance):
    """
    Move the robot a certain distance with noise.
    """
    noisy_distance = distance + numpy.random.normal(0, self.noise_std_dev)
    self.position[0] += noisy_distance * numpy.cos(self.position[2])
    self.position[1] += noisy_distance * numpy.sin(self.position[2])

  def rotate(self, rotation):
    """
    Rotate the robot a certain angle with noise.
    """
    noisy_rotation = rotation + numpy.random.normal(0, self.angle_noise_std_dev)
    self.position[2] = normalize_angle(self.position[2] + noisy_rotation)

  def sense_environment(self):
    """
    Simulate sensing the environment using integrated LIDAR settings.
    """
    sensed_data = []
    lidar_position = numpy.array(to_world_from_ref(self.lidar_offset, self.position))
    for angle in numpy.linspace(0, numpy.rad2deg(self.field_of_view), self.num_points):
      distance = self.sense(lidar_position, numpy.deg2rad(angle) + lidar_position[2], self.max_distance)
      if distance < self.max_distance:
        sensed_data.append((1, angle + numpy.random.normal(0, self.sensor_angle_std_dev), \
                            distance + numpy.random.normal(0, self.sensor_noise)))
    return sensed_data

  def sense(self, origin, angle_rad, max_distance):
    """
    Casts a ray from origin in the direction of angle_rad and returns first intersection
    """
    min_distance = max_distance
    x_end = origin[0] + min_distance * numpy.cos(angle_rad)
    y_end = origin[1] + min_distance * numpy.sin(angle_rad)
    ray_segment = (origin, (x_end, y_end))
    for wall_segment in self.segments:
      intersection = segment_segment_intersection(ray_segment, wall_segment)
      if intersection:
        distance = point_to_point_distance(origin, intersection)
        if distance < min_distance:
          min_distance = distance
    return min_distance

