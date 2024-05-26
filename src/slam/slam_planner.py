import numpy

from slam.slam_geometry import *

def plan_speeds_from_lidar(lidar_data, localized):
  if not localized:
    return 0.0, 0.0

  fov = numpy.deg2rad(45.0)
  want_distance = 600.0
  stop_distance = 300.0
  base_speed = 100.0

  min_distance = want_distance
  for i, a, d in lidar_data:
    if numpy.abs(normalize_angle(numpy.deg2rad(a) - 0.0)) < fov:
      if (d < min_distance):
        min_distance = d

  return clamped_linear(x0 = stop_distance, y0 = -base_speed, x1 = want_distance, y1 = base_speed, x = min_distance), base_speed

def plan_motion_from_lidar(lidar_data, localized):
  if not localized:
    return 0.0, 0.0

  fov = numpy.deg2rad(45.0)
  want_distance = 200.0

  min_distance = want_distance
  for i, a, d in lidar_data:
    if numpy.abs(normalize_angle(numpy.deg2rad(a) - 0.0)) < fov:
      if (d < min_distance):
        min_distance = d

  move_distance = max(0.0, (min_distance - 20.0) * 0.1)
  turn_angle = (1.0 - min_distance / want_distance) * numpy.deg2rad(10.0)

  return move_distance, turn_angle
