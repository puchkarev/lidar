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

  count = len(cartesian_points)

  def max_distance(start, end):
    ix = (start + 1) % count
    distance = 0
    while ix != (end % count):
      distance = max(distance, point_to_line_distance(cartesian_points[ix % count], \
                                                      (cartesian_points[start % count], \
                                                       cartesian_points[end % count])))
      ix = (ix + 1) % count
    return distance

  def extendable(start, end):
    if (end - start + 1) < 5:
      return max_distance(start, end) < threshold

    # for a line which is long enough only check the end point as compared to a line to the midpoint
    mid = int((start + end) / 2)
    return point_to_line_distance(cartesian_points[end % count], \
                                  (cartesian_points[start % count], \
                                   cartesian_points[mid % count])) < threshold

  si = 0
  ei = 2
  segment_ix = []
  while si < count:
    # If possible to extend the segment we extend it
    # - must not go beyond the start of the first segment (if one exists)
    # - must not loop around
    # - all intermediate points must be within threshold distance
    if (len(segment_ix) == 0 or (ei % count) != segment_ix[0][0]) and \
       (ei - si) < count and \
       extendable(si, ei):
      ei += 1
      continue

    # segment must have at the minium number of points to be valid.
    if (ei - si) >= min_points:
      start_ix = si % count
      end_ix = (ei - 1) % count
      segment_ix.append((start_ix, end_ix))

    # the start of next segment can start and the end of the previous one,
    # but we should not extend past the end of the first segment (if one exists)
    si = ei - 1
    if len(segment_ix) > 0 and (ei % count) == segment_ix[0][0]:
      break
    ei += 1

  # check if we can combine the first and last segments
  if len(segment_ix) >= 2 and segment_ix[-1][1] == segment_ix[0][0]:
    if max_distance(segment_ix[-1][0], segment_ix[0][1]) < threshold:
      segments_ix[0] = ([segment_ix[-1][0]], segment_ix[0][1])
      segments.pop()

  segments = []
  for seg_ix in segment_ix:
    # grab all the points on the segment
    pts = []
    if seg_ix[0] < seg_ix[1]:
      pts = cartesian_points[seg_ix[0]:seg_ix[1]+1]
    else:
      pts = cartesian_points[seg_ix[0]:] + cartesian_points[:seg_ix[1]]

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
