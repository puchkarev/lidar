import numpy

from slam.slam_geometry import *

def plan_speeds_from_lidar(lidar_data, localized):
  if not localized:
    return 0.0, 0.0

  base_speed = 100.0
  steer_speed = 50.0
  turn_speed = 10.0

  wall_distance = 400.0
  fov = numpy.deg2rad(20.0)

  front_wall = float('inf')
  left_wall = float('inf')
  back_wall = float('inf')
  right_wall = float('inf')

  for i, a, d in lidar_data:
    if numpy.abs(normalize_angle(numpy.deg2rad(a) - numpy.deg2rad(0.0))) < fov:
      front_wall = min(front_wall, d)
    if numpy.abs(normalize_angle(numpy.deg2rad(a) - numpy.deg2rad(90.0))) < fov:
      left_wall = min(front_wall, d)
    if numpy.abs(normalize_angle(numpy.deg2rad(a) - numpy.deg2rad(180.0))) < fov:
      back_wall = min(front_wall, d)
    if numpy.abs(normalize_angle(numpy.deg2rad(a) - numpy.deg2rad(270.0))) < fov:
      right_wall = min(front_wall, d)

  if front_wall < wall_distance:
    # turn left (wall ahead)
    return -turn_speed, turn_speed
  elif right_wall < wall_distance:
    # go straight (wall on right)
    return base_speed, base_speed
  elif front_wall <= left_wall and front_wall <= right_wall and front_wall <= back_wall:
    # go straight (closest wall ahead)
    return base_speed, base_speed
  elif right_wall <= front_wall and right_wall <= left_wall and right_wall <= back_wall:
    # steer right (closest wall on right)
    return base_speed, steer_speed
  elif left_wall <= front_wall and left_wall <= right_wall and left_wall <= back_wall:
    # steer right (closest wall on left)
    return base_speed, steer_speed
  elif back_wall <= front_wall and back_wall <= right_wall and back_wall <= left_wall:
    # steer right (closest wall behind)
    return base_speed, steer_speed
  else:
    # this is weird. So stop
    return 0.0, 0.0

  # return clamped_linear(x0 = stop_distance, y0 = -base_speed, x1 = want_distance, y1 = base_speed, x = min_distance), base_speed

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
