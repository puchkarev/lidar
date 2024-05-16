# Simulated Robot
import numpy

from slam_geometry import point_to_point_distance, segments_intersect, line_segment_intersection, normalize_angle, \
                          segment_to_segment_distance, point_to_segment_distance

class SimulatedRobot:
  def __init__(self, initial_position, segments, \
               noise_std_dev = 1.0, angle_noise_std_dev = 0.5, \
               sensor_noise = 0.1, sensor_angle_std_dev = 0.1, \
               num_points = 100, field_of_view = numpy.deg2rad(360.0), max_distance = 100):
    self.position = numpy.array(initial_position) # Actual position of the robot
    self.segments = segments # The representation of the world as a series of segmnets
    self.noise_std_dev = noise_std_dev  # Noise standard deviation for movement
    self.angle_noise_std_dev = angle_noise_std_dev  # Noise standard deviation for rotation
    self.sensor_noise = sensor_noise # Noise standard deviation for lidar distance
    self.sensor_angle_std_dev = sensor_angle_std_dev # Noise standard deviation for lidar angle
    self.num_points = num_points  # Number of LIDAR points
    self.field_of_view = field_of_view  # Field of view for LIDAR in degrees
    self.max_distance = max_distance  # Maximum sensing distance for LIDAR

  def move(self, distance, rotation):
    """
    Move the robot a certain distance and rotation, with noise affecting both.
    Rotation should be in radians. Rotation is applied first.
    """
    # Introduce noise to rotation and convert to radians
    noisy_rotation = rotation + numpy.random.normal(0, self.angle_noise_std_dev)
    self.position[2] = normalize_angle(self.position[2] + noisy_rotation)

    # Calculate movement with added noise
    noisy_distance = distance + numpy.random.normal(0, self.noise_std_dev)
    dx = noisy_distance * numpy.cos(self.position[2])
    dy = noisy_distance * numpy.sin(self.position[2])

    new_position = self.position + numpy.array([dx, dy, 0.0])
    # check for collisions
    has_collision = False
    pt_old = [self.position[0], self.position[1]]
    pt_new = [new_position[0], new_position[1]]
    for segment in self.segments:
      path_dist = segment_to_segment_distance((pt_old, pt_new), segment)
      if path_dist < 10.0:
        if point_to_segment_distance(pt_old, segment) > path_dist:
          has_collision = True
          break

    # If we have collided inform the caller
    if has_collision:
      return False

    self.position = new_position
    return True

  def sense_environment(self):
    """
    Simulate sensing the environment using integrated LIDAR settings.
    """
    sensed_data = []
    for angle in numpy.linspace(0, numpy.rad2deg(self.field_of_view), self.num_points):
      distance = self.sense(self.position, numpy.deg2rad(angle) + self.position[2], self.max_distance)
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
      intersection = line_segment_intersection(ray_segment, wall_segment)
      if intersection and segments_intersect(ray_segment, wall_segment):
        distance = point_to_point_distance(origin, intersection)
        if distance < min_distance:
          min_distance = distance
    return min_distance

