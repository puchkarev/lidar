import math
import numpy

def normalize_angle(angle):
  """Normalizes the angle to be in range (-pi, pi]"""
  angle = ( ( angle % (math.pi * 2) ) + math.pi * 2) % (math.pi * 2)
  if (angle > math.pi):
    angle -= math.pi * 2;
  return angle

def point_to_point_distance(p0, p1):
  """Returns distance between two points"""
  return math.hypot(p0[1] - p1[1], p0[0] - p1[0])

def segment_endpoint_distance(segment1, segment2):
  """Returns the distance from segment endpoints to the segments"""
  return point_to_segment_distance(segment1[0], segment2) + \
         point_to_segment_distance(segment1[1], segment2) + \
         point_to_segment_distance(segment2[0], segment1) + \
         point_to_segment_distance(segment2[1], segment1)

def point_to_line_distance(point, line):
  """Returns the distance between the point and the line"""
  # Length of the line segment
  segment_length = math.hypot(line[1][1] - line[0][1], line[1][0] - line[0][0])
  if segment_length <= 0:
    # if segment length is 0, then the line is arebitrary so we can always draw a line
    # to point that will cross it.
    return 0.0
  # Double the area of a triangle formed by points point and endpoints of line
  double_triangle_area = numpy.abs((line[1][0] - line[0][0]) * (point[1] - line[0][1]) - \
                                   (point[0] - line[0][0]) * (line[1][1] - line[0][1]))
  # in this case the height of triangle is (distance to point from line) and we have A = (1/2) * b * h
  # so h = 2 * A / b
  return double_triangle_area / segment_length

def projected_point_on_segment(point, segment):
  """Returns the projection of a point onto the segment"""
  # Make v1 (as the segment) and v2 as the segment[0] to point
  v1 = (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
  v2 = (point[0] - segment[0][0], point[1] - segment[0][1])

  v1_len = math.hypot(v1[0], v1[1])
  if v1_len <= 0.0:
    return math.inf

  # Project v2 onto v1 normalized.
  v1_norm = (v1[0] / v1_len, v1[1] / v1_len)
  dot = v1_norm[0] * v2[0] + v1_norm[1] * v2[1]
  return dot / v1_len

def point_to_segment_distance(point, segment):
  """Returns the distance between the point and the segment"""
  dot = projected_point_on_segment(point, segment)

  # return distances to edges of segment if the point does not project onto line,
  # and if it does measure the distance from the projected point.
  if dot <= 0.0:
    return math.hypot(point[0] - segment[0][0], point[1] - segment[0][1])
  elif dot >= 1.0:
    return math.hypot(point[0] - segment[1][0], point[1] - segment[1][1])
  else:
    return point_to_line_distance(point, segment)

def segments_intersect(segment1, segment2):
  """Returns true if segmnets intersect"""
  if segment_segment_intersection(segment1, segment2):
    return True
  return False

def segment_segment_intersection(segment1, segment2):
  """Returns the intersection point between two segments"""
  intersection = line_line_intersection(segment1, segment2)
  if not intersection:
    return None
  p1 = projected_point_on_segment(intersection, segment1)
  if p1 < 0.0 or p1 > 1.0:
    return None
  p2 = projected_point_on_segment(intersection, segment2)
  if p2 < 0.0 or p2 > 1.0:
    return None
  return intersection

def segment_to_segment_distance(segment1, segment2):
  """Returns the minimum distance between two segments"""
  if segments_intersect(segment1, segment2):
    return 0.0
  distances = [point_to_segment_distance(segment1[0], segment2), \
               point_to_segment_distance(segment1[1], segment2), \
               point_to_segment_distance(segment2[0], segment1), \
               point_to_segment_distance(segment2[1], segment1)]
  return min(distances)

def lines_intersect(line1, line2):
  """Returns true if two lines expressed by pairs of points, intersect"""
  if line_line_intersection(line1, line2):
    return True
  return False

def line_line_intersection(line1, line2):
  """Returns the intersection point formed by two lines defined by pairs of points"""
  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

  def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

  div = det(xdiff, ydiff)
  if div == 0:
    return None

  d = (det(*line1), det(*line2))
  x = det(d, xdiff) / div
  y = det(d, ydiff) / div
  return (x, y)

def polar_to_cartesian(lidar_data, pose):
  """
  Convert polar LIDAR data to Cartesian coordinates based on the robot's current pose.

  Parameters:
  - lidar_data: List of tuples, each containing (intensity, angle, distance) relative to the robot's local frame.
  - pose: The robot's current pose (x, y, theta) in the global frame.

  Returns:
  - List of tuples representing the points in global Cartesian coordinates.
  """
  x_pos, y_pos, theta = pose
  cartesian_points = []
  for intensity, angle, distance in lidar_data:
    # Convert relative angle (from LIDAR data) to global angle
    global_angle = theta + numpy.deg2rad(angle)

    # Calculate global coordinates
    x = x_pos + distance * numpy.cos(global_angle)
    y = y_pos + distance * numpy.sin(global_angle)
    cartesian_points.append((x, y))

  return cartesian_points

def transform_point(point, original_pose, new_pose):
  """Transforms a point as seen from some original pose, to a point seen from the new pose"""
  distance = math.hypot(point[1] - original_pose[1], point[0] - original_pose[0])
  angle = numpy.arctan2(point[1] - original_pose[1], point[0] - original_pose[0]) - original_pose[2]
  global_angle = angle + new_pose[2]
  return (new_pose[0] + distance * numpy.cos(global_angle), new_pose[1] + distance * numpy.sin(global_angle))

def transform_segment(segment, original_pose, new_pose):
  """Transforms a segment as seen from some original pose, to a point seen from the new pose"""
  return ( \
    transform_point(point = segment[0], original_pose = original_pose, new_pose = new_pose), \
    transform_point(point = segment[1], original_pose = original_pose, new_pose = new_pose)
  )

def segment_angle(segment):
  return numpy.arctan2(segment[1][1] - segment[0][1], segment[1][0] - segment[0][0])

def score_distance(distance, sensor_noise):
  """Function that converts a distance between measurements and noise value to a likelihood"""
  return numpy.exp(-0.5 * ( distance ** 2 / sensor_noise ** 2) )

