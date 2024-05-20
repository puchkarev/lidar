import math
import numpy

def normalize_angle(angle: float):
  """Normalizes the angle to be in range (-pi, pi]"""
  angle = ( ( angle % (math.pi * 2) ) + math.pi * 2) % (math.pi * 2)
  if (angle > math.pi):
    angle -= math.pi * 2;
  return angle

def point_to_point_distance(p0: list[float], \
                            p1: list[float]):
  """Returns distance between two points"""
  return math.hypot(p0[1] - p1[1], p0[0] - p1[0])

def segment_endpoint_distance(segment1: list[list[float], list[float]], \
                              segment2: list[list[float], list[float]]):
  """Returns the distance from segment endpoints to the segments"""
  return point_to_segment_distance(segment1[0], segment2) + \
         point_to_segment_distance(segment1[1], segment2) + \
         point_to_segment_distance(segment2[0], segment1) + \
         point_to_segment_distance(segment2[1], segment1)

def point_to_line_distance(point: list[float], \
                           line: list[list[float], list[float]]):
  """Returns the distance between the point and the line"""
  # Length of the line segment
  segment_length = point_to_point_distance(line[0], line[1])
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

def projected_point_on_segment(point: list[float], \
                               segment: list[list[float], list[float]], \
                               segment_len: float = None):
  """Returns the projection of a point onto the segment"""
  # Make v1 (as the segment) and v2 as the segment[0] to point
  v1 = (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
  v2 = (point[0] - segment[0][0], point[1] - segment[0][1])

  if not segment_len:
    segment_len = point_to_point_distance((0, 0), (v1[0], v1[1]))
  if segment_len <= 0.0:
    return math.inf

  # compute the division only once
  segment_len_inv = 1.0 / segment_len

  # Project v2 onto v1 normalized.
  v1_norm = (v1[0] * segment_len_inv, v1[1] * segment_len_inv)
  dot = v1_norm[0] * v2[0] + v1_norm[1] * v2[1]
  return dot * segment_len_inv

def line_fit(points: list[list[float]]):
  x = [p[0] for p in points]
  y = [p[1] for p in points]

  # Center of the distribution
  x_mean = numpy.mean(x)
  y_mean = numpy.mean(y)

  # Place the points around the origin
  x_centered = x - x_mean
  y_centered = y - y_mean

  # Compute covariance matrix
  data = numpy.vstack([x_centered, y_centered])
  cov_matrix = numpy.cov(data)

  # Eigendecomposition
  eigenvalues, eigenvectors = numpy.linalg.eig(cov_matrix)

  # Principal eigenvector
  principal_eigenvector = eigenvectors[:, numpy.argmax(eigenvalues)]

  a, b = principal_eigenvector
  c = -(a * x_mean + b * y_mean)

  # a * x + b * y + c = 0
  # the direction vector is (a, b) intersects (x_mean, y_mean)
  return a, b, c, x_mean, y_mean

def point_to_segment_distance(point: list[float], \
                              segment: list[list[float], list[float]]):
  """Returns the distance between the point and the segment"""
  dot = projected_point_on_segment(point, segment)

  # return distances to edges of segment if the point does not project onto line,
  # and if it does measure the distance from the projected point.
  if dot <= 0.0:
    return point_to_point_distance(point, segment[0])
  elif dot >= 1.0:
    return point_to_point_distance(point, segment[1])
  else:
    return point_to_line_distance(point, segment)

def segments_intersect(segment1: list[list[float], list[float]], \
                       segment2: list[list[float], list[float]]):
  """Returns true if segmnets intersect"""
  if segment_segment_intersection(segment1, segment2):
    return True
  return False

def segment_segment_intersection(segment1: list[list[float], list[float]], \
                                 segment2: list[list[float], list[float]]):
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

def segment_to_segment_distance(segment1: list[list[float], list[float]], \
                                segment2: list[list[float], list[float]]):
  """Returns the minimum distance between two segments"""
  if segments_intersect(segment1, segment2):
    return 0.0
  distances = [point_to_segment_distance(segment1[0], segment2), \
               point_to_segment_distance(segment1[1], segment2), \
               point_to_segment_distance(segment2[0], segment1), \
               point_to_segment_distance(segment2[1], segment1)]
  return min(distances)

def lines_intersect(line1: list[list[float], list[float]], \
                    line2: list[list[float], list[float]]):
  """Returns true if two lines expressed by pairs of points, intersect"""
  if line_line_intersection(line1, line2):
    return True
  return False

def line_line_intersection(line1: list[list[float], list[float]], \
                           line2: list[list[float], list[float]]):
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

def polar_to_cartesian(lidar_data: list[list[float]], \
                       pose: list[float]):
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

def transform_point(point: list[float], \
                    original_pose: list[float], \
                    new_pose: list[float]):
  """Transforms a point as seen from some original pose, to a point seen from the new pose"""
  distance = point_to_point_distance(point, original_pose)
  angle = numpy.arctan2(point[1] - original_pose[1], point[0] - original_pose[0]) - original_pose[2]
  global_angle = angle + new_pose[2]
  return (new_pose[0] + distance * numpy.cos(global_angle), new_pose[1] + distance * numpy.sin(global_angle))

def transform_segment(segment: list[list[float], list[float]], \
                      original_pose: list[float], \
                      new_pose: list[float]):
  """Transforms a segment as seen from some original pose, to a point seen from the new pose"""
  return ( \
    transform_point(point = segment[0], original_pose = original_pose, new_pose = new_pose), \
    transform_point(point = segment[1], original_pose = original_pose, new_pose = new_pose)
  )

def segment_angle(segment: list[list[float], list[float]]):
  return numpy.arctan2(segment[1][1] - segment[0][1], segment[1][0] - segment[0][0])

