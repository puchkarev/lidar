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
  segments = []

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

  last = count - 1
  si = 0
  ei = 2
  first_segment_ix = [-1, -1]
  last_segment_ix = [-1, -1]
  while si <= last:
    # If possible to extend the segment we extend it
    if max_distance(si, ei) < threshold and (ei - si) < count and (ei % count) != first_segment_ix[0]:
      ei += 1
      continue

    # segment must have at the minium number of points to be valid.
    if (ei - si) >= min_points:
      start_ix = si % count
      end_ix = (ei - 1) % count
      last_segment_ix = [start_ix, end_ix]
      if first_segment_ix[0] < 0:
        first_segment_ix = [start_ix, end_ix]
      segments.append((cartesian_points[start_ix], cartesian_points[end_ix]))

    # the start of next segment can start and the end of the previous one.
    si = ei - 1
    if (ei % count) == first_segment_ix[0]:
      break
    ei += 1

  # check if we can combine the first and last segments
  if last_segment_ix[1] == first_segment_ix[0] and first_segment_ix[0] >= 0:
    if max_distance(last_segment_ix[0], first_segment_ix[1]) < threshold:
      segments[0] = (cartesian_points[last_segment_ix[0]], caretesian_points[first_segment_ix[1]])
      segments.pop()

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
