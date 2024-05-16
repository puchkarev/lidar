# Simulated Robot Environment
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from slam_geometry import point_to_point_distance, segments_intersect, line_segment_intersection
from slam_localization import compute_mean_and_covariance
from slam import MappingEnvironment

class SimulatedRobot:
  def __init__(self, initial_position, segments, noise_std_dev=1.0, angle_noise_std_dev=0.5, num_points=100, field_of_view=360, max_distance=100):
    self.position = np.array(initial_position)
    self.segments = segments
    self.noise_std_dev = noise_std_dev  # Noise standard deviation for movement
    self.angle_noise_std_dev = angle_noise_std_dev  # Noise standard deviation for rotation
    self.num_points = num_points  # Number of LIDAR points
    self.field_of_view = field_of_view  # Field of view for LIDAR in degrees
    self.max_distance = max_distance  # Maximum sensing distance for LIDAR

  def move(self, distance, rotation):
    """
    Move the robot a certain distance and rotation, with noise affecting both.
    Rotation should be in radians. Rotation is applied first.
    """
    # Introduce noise to rotation and convert to radians
    noisy_rotation = rotation + np.random.normal(0, self.angle_noise_std_dev)
    self.position[2] += noisy_rotation

    # Calculate movement with added noise
    noisy_distance = distance + np.random.normal(0, self.noise_std_dev)
    dx = noisy_distance * np.cos(self.position[2])
    dy = noisy_distance * np.sin(self.position[2])

    # Update position
    self.position += np.array([dx, dy, 0.0])

  def sense_environment(self):
    """
    Simulate sensing the environment using integrated LIDAR settings.
    """
    sensed_data = []
    for angle in np.linspace(0, self.field_of_view, self.num_points):
      distance = self.sense(self.position, np.deg2rad(angle) + self.position[2], self.max_distance)
      if distance < self.max_distance:
        sensed_data.append((1, angle, distance))
    return sensed_data

  def sense(self, origin, angle_rad, max_distance):
    """
    Casts a ray from origin in the direction of angle_rad and returns first intersection
    """
    min_distance = max_distance
    x_end = origin[0] + min_distance * np.cos(angle_rad)
    y_end = origin[1] + min_distance * np.sin(angle_rad)
    ray_segment = (origin, (x_end, y_end))
    for wall_segment in self.segments:
      intersection = line_segment_intersection(ray_segment, wall_segment)
      if intersection and segments_intersect(ray_segment, wall_segment):
        distance = point_to_point_distance(origin, intersection)
        if distance < min_distance:
          min_distance = distance
    return min_distance

if __name__ == '__main__':
  # Define the environment
  segments = [
    ((100.0, 100.0), (200.0, 100.0)),
    ((200.0, 100.0), (200.0, 150.0)),
    ((200.0, 150.0), (300.0, 150.0)),
    ((300.0, 150.0), (300.0, 100.0)),
    ((300.0, 100.0), (500.0, 100.0)),
    ((500.0, 100.0), (500.0, 500.0)),
    ((100.0, 100.0), (100.0, 500.0)),
    ((100.0, 500.0), (500.0, 500.0))
  ]

  initial_position = [400.0, 250.0, np.deg2rad(35.0)]

  # type of noise we expect from movement
  move_error = 2.0
  turn_error = np.deg2rad(1.0)

  # Initialize the robot
  robot = SimulatedRobot(initial_position = initial_position, \
                         segments = segments, \
                         noise_std_dev = move_error, \
                         angle_noise_std_dev = turn_error, \
                         max_distance = 1000.0)

  mapping_environment = MappingEnvironment(initial_position=initial_position, segments = segments)

  # Set up the plotting
  fig, ax = plt.subplots()
  plt.xlim(0, 600)
  plt.ylim(0, 600)
  ax.set_aspect('equal')

  def update(frame):
    ax.cla()
    plt.xlim(0, 600)
    plt.ylim(0, 600)
    ax.set_aspect('equal')

    # move the robot randomly about the map
    move_distance = np.random.uniform(0.0, 5.0)
    turn_angle = np.random.uniform(np.deg2rad(-1.0), np.deg2rad(5.0))

    # inform the mapping environment that we have moved
    if move_distance != 0.0 or turn_angle != 0.0:
      robot.move(distance = move_distance, rotation = turn_angle)
      mapping_environment.move_robot(move_distance = move_distance, rotate_angle = turn_angle, \
                                     distance_error = move_error, rotation_error = turn_error)

    # get the sensor data
    lidar_data = robot.sense_environment()

    # update the mapping invironment based on lidar data
    mapping_environment.lidar_update(lidar_data)

    # get the best pose estimate
    robot_mean, robot_covariance = compute_mean_and_covariance(mapping_environment.poses, \
                                                               mapping_environment.weights)
    print("distribution", robot_mean, robot_covariance[0][0], robot_covariance[1][1], robot_covariance[2][2])

    # show the real returns based on robot actual position.
    #if lidar_data:
    #  robot_pose = robot.position
    #  ax.plot([robot_pose[0] + d * np.cos(np.deg2rad(a) + robot_pose[2]) for i, a, d in lidar_data], \
    #          [robot_pose[1] + d * np.sin(np.deg2rad(a) + robot_pose[2]) for i, a, d in lidar_data], \
    #          'bo', markersize=3)

    # show the real segments from the environment.
    for segment in segments:
      ax.plot([segment[0][0], segment[1][0]], \
              [segment[0][1], segment[1][1]], 'k-')

    # show the actual robot position
    ax.plot([robot.position[0]], [robot.position[1]], 'bo', markersize=10)

    # show the robot positions
    ax.plot([p[0] for p in mapping_environment.poses], \
            [p[1] for p in mapping_environment.poses], 'ro', markersize=1)

    # show the lidar returns as understood by the mapping
    ax.plot([p[0] for p in mapping_environment.cartesian_points], \
            [p[1] for p in mapping_environment.cartesian_points], 'bo')

    # show the segments as understood by the mapping
    for segment_association in mapping_environment.segment_associations:
      segment = segment_association[0]
      ax.plot([segment[0][0], segment[1][0]],
              [segment[0][1], segment[1][1]], 'g-')

    for segment in mapping_environment.new_segments:
      ax.plot([segment[0][0], segment[1][0]],
              [segment[0][1], segment[1][1]], 'r-')

    # show the corners as understood by the mapping
    for corner_association in mapping_environment.corner_associations:
      corner = corner_association[0]
      ax.plot([corner[0]], [corner[1]], 'go', markersize = 5)

    for corner in mapping_environment.new_corners:
      ax.plot([corner[0]], [corner[1]], 'ro', markersize = 5)

  ani = animation.FuncAnimation(fig, update, frames=100, interval=10)
  plt.show()
