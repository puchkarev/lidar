import numpy

from slam_geometry import normalize_angle

def plan_motion_from_lidar(lidar_data):
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
