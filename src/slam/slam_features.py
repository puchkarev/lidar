import math
import numpy

from slam.slam_geometry import *

def extract_segments(cartesian_points, threshold = 1.0, min_points = 2):
  """
  Extracts line segments formed by the cartesian_points, the cartesian_points are expected to be ordered,
  as generated from a lidar scan.

  Parameters:
  - cartesian_points: list of points (x, y) in a sequential list
  - threshold: how close each point needs to be to the line segment
  - min_points: minimum number of points to consider a segment

  Returns:
  - List of tuples which express line segments as a tuple of two end points
  """

  point_count = len(cartesian_points)

  def max_distance(start, end):
    """
    computes the maximum of distance between all intermediate points and a line
    formed by first and last points
    """
    ix = (start + 1) % point_count
    distance = 0
    while ix != (end % point_count):
      distance = max(distance, point_to_line_distance(cartesian_points[ix % point_count], \
                                                      (cartesian_points[start % point_count], \
                                                       cartesian_points[end % point_count])))
      ix = (ix + 1) % point_count
    return distance

  def get_point_quantity(start, end):
    """returns the number of points between the start end end index inclusive"""
    start_rem = start % point_count
    end_rem = end % point_count
    if end_rem > start_rem:
      return end_rem - start_rem + 1
    elif end_rem == start_rem:
      return 0
    else:
      return (point_count - start_rem) + end_rem

  def extendable(start, end):
    """returns if the segment can be extended to include the point at end index"""
    points = get_point_quantity(start, end)

    # for short segments we check the distance all all intermediate points.
    # complexity grows at the rate of number of points O(N), but since we
    # call this for every points which we add to the segment the outher
    # complexity is also O(N) so combined this is O(N^2) so should be avoided.
    if points < 5:
      return max_distance(start, end) < threshold

    # for a line which is long enough only check the end point as compared to a line to the midpoint.
    # complexity of this is O(1).
    mid = int(start + points / 2)
    return point_to_line_distance(cartesian_points[end % point_count], \
                                  (cartesian_points[start % point_count], \
                                   cartesian_points[mid % point_count])) < threshold

  si = 0
  ei = 2
  segment_ix = []
  while si < point_count:
    # this is true if the end index is equal to the point after the start of the first segment
    beyond_the_start_of_first_segment = \
      len(segment_ix) > 0 and \
      (ei % point_count) == ((segment_ix[0][0] + 1) % point_count)

    # If possible to extend the segment we extend it
    # - must not go beyond the start of the first segment (if one exists)
    # - must not loop around
    # - all intermediate points must be within threshold distance
    if (not beyond_the_start_of_first_segment) and get_point_quantity(si, ei) < point_count and extendable(si, ei):
      ei += 1
      continue

    # segment must have at the minium number of points to be valid.
    if (ei - si) >= min_points:
      start_ix = si % point_count
      end_ix = (ei - 1) % point_count
      segment_ix.append((start_ix, end_ix))

    # the start of next segment can start and the end of the previous one,
    # but we should not extend past the end of the first segment (if one exists)
    si = ei - 1
    if len(segment_ix) > 0 and (ei % point_count) == (segment_ix[0][0] % point_count):
      break
    ei += 1

  # check if we can combine the first and last segments
  if len(segment_ix) >= 2 and (segment_ix[-1][1] % point_count) == (segment_ix[0][0] % point_count):
    if max_distance((segment_ix[-1][0] % point_count), (segment_ix[0][1] % point_count)) < threshold:
      segment_ix[0] = (segment_ix[-1][0] % point_count, segment_ix[0][1] % point_count)
      segment_ix.pop()

  segments = []
  for seg_ix in segment_ix:
    # grab all the points on the segment
    pts = []
    start_ix = seg_ix[0] % point_count
    end_ix = seg_ix[1] % point_count
    if start_ix < end_ix:
      pts = cartesian_points[start_ix:end_ix + 1]
    else:
      pts = cartesian_points[start_ix:] + cartesian_points[:end_ix+1]

    # find the best fit line (intersects x_mean, y_mean)
    # a * x + b * y + c = 0
    a, b, c, x_mean, y_mean = line_fit(pts)

    # project the start and end points onto the line
    seg = ((x_mean, y_mean), (x_mean + a, y_mean + b))
    f1 = projected_point_on_segment(cartesian_points[seg_ix[0]], seg)
    p1 = (seg[0][0] + a * f1, seg[0][1] + b * f1)
    f2 = projected_point_on_segment(cartesian_points[seg_ix[1]], seg)
    p2 = (seg[0][0] + a * f2, seg[0][1] + b * f2)

    segments.append((p1, p2))
  return segments

def detect_corners_from_segments(segments, angle_threshold = math.pi/6):
  """ Detect corners from a list of segments based on the angle between them. """
  corners = []
  for i in range(len(segments)):
    j = (i + 1) % len(segments)
    angle = numpy.abs(normalize_angle(segment_angle(segments[i]) - segment_angle(segments[j])))
    if angle > angle_threshold:
      corners.append(line_line_intersection(segments[i], segments[j]))
  return corners
