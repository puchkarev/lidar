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
  last = count - 1
  si = 0
  ei = 2
  first_segment_start = -1
  while si <= last:
    # measure the maximum distance to the intermediate points
    max_distance = 0
    for ix in range(si + 1, ei):
      max_distance = max(max_distance, point_to_line_distance(cartesian_points[ix % count], \
                                                              (cartesian_points[si % count], \
                                                               cartesian_points[ei % count])))
      if max_distance >= threshold:
        break

    # If possible to extend the segment we extend it
    if max_distance < threshold and (ei - si) < count and (ei % count) != first_segment_start:
      ei += 1
      continue

    # segment must have at the minium number of points to be valid.
    if (ei - si) >= min_points:
      if first_segment_start < 0:
        first_segment_start = (si % count)
      segments.append((cartesian_points[si % count], cartesian_points[(ei - 1) % count]))

    # the start of next segment can start and the end of the previous one.
    si = ei - 1
    if (ei % count) == first_segment_start:
      break
    ei += 1

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
