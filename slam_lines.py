import math
import numpy
import unittest

from sklearn.linear_model import RANSACRegressor
from slam_geometry import normalize_angle

def extract_lines_ransac(cartesian_points, n_lines=3, max_trials=100, min_samples=2, residual_threshold=0.2):
  """
  Extracts a the lines using the RANSAC algorithm. The returned list is slope, and intercept at x=0.0

  Parameters:
  - points are sequential (as from lidar) list of points (x, y)
  - n_lines is the maximum number of lines to return
  - min_samples is the minimum number of points to include
  - residual threshold for grouping

  Returns:
  - List of tuples representing the (slope and intercept) of lines.
  """
  lines = []
  original_points = numpy.array(cartesian_points)

  for _ in range(n_lines):
    if len(original_points) < min_samples:
      break

    # RANSAC regressor to find the best fitting line
    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)
    X = original_points[:, 0].reshape(-1, 1)
    y = original_points[:, 1]

    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = numpy.logical_not(inlier_mask)

    # Store the line model if it is significant
    if inlier_mask.any():
      line_model = ransac.estimator_
      line = (line_model.coef_[0], line_model.intercept_)
      lines.append(line)

      # Prepare points for next iteration
      original_points = original_points[outlier_mask]
    else:
      break

  return lines

def are_lines_mergeable(line1, line2, distance_threshold=1.0, angle_threshold = math.pi / 36):
  """Check if two lines are close enough and aligned enough to be merged."""
  slope1, intercept1 = line1
  slope2, intercept2 = line2

  # Check angular difference
  angle_diff = numpy.abs(normalize_angle(numpy.arctan(slope1) - numpy.arctan(slope2)))
  if angle_diff > angle_threshold:
    return False

  # Check distance at a reasonable x (e.g., the midpoint between line midpoints)
  x_mid = (intercept1 + intercept2) / 2
  y1 = slope1 * x_mid + intercept1
  y2 = slope2 * x_mid + intercept2
  distance = numpy.abs(y1 - y2)

  return distance <= distance_threshold

def merge_lines(lines, distance_threshold=1.0, angle_threshold = math.pi / 36):
  """Merge lines that are close and aligned into fewer, longer lines."""
  if not lines:
    return []

  # Start with the first line
  merged_lines = [lines[0]]

  for current_line in lines[1:]:
    merged = False
    for i, merged_line in enumerate(merged_lines):
      if are_lines_mergeable(merged_line, current_line, distance_threshold, angle_threshold):
        # Compute new slope and intercept by averaging
        slope = (merged_line[0] + current_line[0]) / 2
        intercept = (merged_line[1] + current_line[1]) / 2
        merged_lines[i] = (slope, intercept)
        merged = True
        break
    if not merged:
      merged_lines.append(current_line)

  return merged_lines

def refine_line_parameters(cartesian_points, lines, threshold = 1.0):
  """Adjust line parameters to minimize the error to the given data points."""
  refined_lines = []
  for slope, intercept in lines:
    # Filter data points close to the current line
    close_points = [p for p in cartesian_points if numpy.abs(p[1] - (slope * p[0] + intercept)) < threshold]
    if close_points:
      # Fit a new line to these points
      X = numpy.array([p[0] for p in close_points]).reshape(-1, 1)
      y = numpy.array([p[1] for p in close_points])
      line_fit = numpy.polyfit(X.flatten(), y, 1)
      refined_lines.append((line_fit[0], line_fit[1]))
    else:
      refined_lines.append((slope, intercept))
  return refined_lines

def line_angle(line):
  return numpy.arctan(line[0])

class TestBasicMethods(unittest.TestCase):
  def test_extract_lines_ransac(self):
    numpy.testing.assert_allclose(numpy.array(extract_lines_ransac( \
      cartesian_points = [(14, 11), (15, 12), (16, 13)], \
      n_lines = 2, max_trials = 100, min_samples = 2, residual_threshold = 0.2)),
      numpy.array([(1.0, -3.0)]))

    numpy.testing.assert_allclose(numpy.array(extract_lines_ransac( \
      cartesian_points = [(10, 10), (11, 10), (12, 10), (13, 10)], \
      n_lines = 2, max_trials = 100, min_samples = 2, residual_threshold = 0.2)),
      numpy.array([(0.0, 10.0)]))

  def test_merge_lines(self):
    numpy.testing.assert_allclose(numpy.array(merge_lines( \
      lines = [(0.5, 2.0), (0.52, 1.95), (1.0, 0.0), (0.98, 0.1)], \
      distance_threshold = 1.0, angle_threshold = math.pi / 36)),
      numpy.array([(0.51, 1.975), (0.99, 0.05)]))

  def test_refine_line_parameters(self):
    numpy.testing.assert_allclose(numpy.array(refine_line_parameters( \
      cartesian_points = [(10.1, 9.9), (11.05, 10.05), (11.95, 10.3), (13.05, 9.7), \
                          (14.2, 11.2), (14.92, 12.02), (15.99, 12.99)],
      lines = [(0.0, 10.0), (1.0, -3.0)], \
      threshold = 1.0)),
      numpy.array([(-0.044232, 10.497821), (1.122092, -4.837721)]),
      rtol=0, atol=1e-5)

  def test_line_angle(self):
    self.assertEqual(line_angle((0.0, 5.0)), 0.0)
    self.assertEqual(line_angle((0.0, 10.0)), 0.0)
    self.assertEqual(line_angle((1.0, 5.0)), math.pi / 4)
    self.assertEqual(line_angle((1.0, 10.0)), math.pi / 4)
    self.assertEqual(line_angle((-1.0, 5.0)), -math.pi / 4)
    self.assertEqual(line_angle((-1.0, 10.0)), -math.pi / 4)

if __name__ == '__main__':
  unittest.main()
