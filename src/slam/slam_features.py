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

  last = len(cartesian_points) - 1
  si = -1
  ei = 0
  while ei <= last:
    # measure the maximum distance to the intermediate points
    distance = 0
    for ix in range(si + 1, ei):
      distance = max(distance, point_to_line_distance(cartesian_points[ix], \
                                                      (cartesian_points[si], cartesian_points[ei])))
      if distance >= threshold:
        break

    # See if we can keep this point, or start a segment with it.
    if distance < threshold:
      if si < 0:
        si = ei
        points = 0
      ei += 1
      continue

    # segment must have at least two distinct points to be valid.
    if ei - si >= min_points:
      segments.append((cartesian_points[si], cartesian_points[ei - 1]))

    # the start of next segment can start and the end of the previous one.
    si = ei - 1

  # if we finished without terminating the segment, consider terminating it
  if si >= 0:
    if last + 1 - si >= min_points:
      segments.append((cartesian_points[si], cartesian_points[last]))
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
