#!/usr/bin/env python
import os
import math
import unittest
import numpy as np
import multiprocessing
from itertools import repeat

# Basic methods

def degToRad(deg):
  """Convert degrees to radians"""
  return deg * math.pi / 180.0

def radToDeg(rad):
  """Convert radians to degrees"""
  return rad * 180.0 / math.pi

def normalizeAngle(ang):
  """Normalizes angle to [-pi to pi] range"""
  while ang > math.pi:
    ang = ang - math.pi * 2
  while ang < -math.pi:
    ang = ang + math.pi * 2
  return ang

def getCordFromRefCartesian(ref, cord):
  """Converts coordinates from those expressed in reference frame to global frame"""
  nx = ref[0] + math.cos(ref[2]) * cord[0] - math.sin(ref[2]) * cord[1]
  ny = ref[1] + math.sin(ref[2]) * cord[0] + math.cos(ref[2]) * cord[1]
  if len(cord) == 3:
    return [nx, ny, normalizeAngle(ref[2] + cord[2])]
  else:
    return [nx, ny]

def getCordFromRefPolar(ref, cord):
  """Converts coordinates from those expressed in reference frame to global frame"""
  nx = ref[0] + math.cos(ref[2] + cord[0]) * cord[1]
  ny = ref[1] + math.sin(ref[2] + cord[0]) * cord[1]
  if len(cord) == 3:
    return [nx, ny, normalizeAngle(ref[2] + cord[2])]
  else:
    return [nx, ny]

def roundToResolution(value, resolution):
  """Rounds the coordinates the resolution"""
  res = []
  for v in value:
    res.append(round(v / resolution) * resolution)
  return res

def getDist(pt1, pt2):
  """Returns distance between two points"""
  return math.hypot(pt2[1] - pt1[1], pt2[0] - pt1[0])

def getAng(pt1, pt2):
  """Returns angle between two points"""
  return math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

class TestBasicMethods(unittest.TestCase):
  def testDegToRad(self):
    self.assertEqual(degToRad(0.0), 0.0)
    self.assertEqual(degToRad(180.0), math.pi)
    self.assertEqual(degToRad(360.0), 2 * math.pi)

  def testRadToDeg(self):
    self.assertEqual(radToDeg(0.0), 0.0)
    self.assertEqual(radToDeg(math.pi), 180.0)
    self.assertEqual(radToDeg(2 * math.pi), 360.0)

  def testNormalizeAngle(self):
    self.assertEqual(normalizeAngle(math.pi), math.pi)
    self.assertEqual(normalizeAngle(math.pi * 2), 0)
    self.assertEqual(normalizeAngle(math.pi * 3), math.pi)

  def testGetCordFromRefCartesian(self):
    np.testing.assert_allclose(getCordFromRefCartesian(np.array([0, 0, 0]), np.array([0, 0, 0])), \
                               np.array([0, 0, 0]))
    np.testing.assert_allclose(getCordFromRefCartesian(np.array([1, 2, 0]), np.array([3, 4, 0])), \
                               np.array([4, 6, 0]))
    np.testing.assert_allclose(getCordFromRefCartesian(np.array([1, 2, math.pi]), np.array([3, 4, 0])), \
                               np.array([-2, -2, math.pi]))
    np.testing.assert_allclose(getCordFromRefCartesian(np.array([1, 2, math.pi/2]), np.array([3, 4, math.pi])), \
                               np.array([-3, 5, -math.pi/2]))
    np.testing.assert_allclose(getCordFromRefCartesian(np.array([1, 2, math.pi/2]), np.array([3, 4])), \
                               np.array([-3, 5]))

  def testGetCordFromRefPolar(self):
    np.testing.assert_allclose(getCordFromRefPolar(np.array([0, 0, 0]), np.array([0, 0, 0])), \
                               np.array([0, 0, 0]))
    np.testing.assert_allclose(getCordFromRefPolar(np.array([0, 0, 0]), np.array([0, 1, 0])), \
                               np.array([1, 0, 0]))
    np.testing.assert_allclose(getCordFromRefPolar(np.array([0, 0, 0]), np.array([math.pi / 2, 1, 0])), \
                               np.array([0, 1, 0]), 1e-07, 1e-07)
    np.testing.assert_allclose(getCordFromRefPolar(np.array([0, 0, 0]), np.array([math.pi, 1, 0])), \
                               np.array([-1, 0, 0]), 1e-07, 1e-07)
    np.testing.assert_allclose(getCordFromRefPolar(np.array([2, 3, math.pi / 2]), np.array([math.pi, 1, math.pi])), \
                               np.array([2, 2, -math.pi / 2]), 1e-07, 1e-07)
    np.testing.assert_allclose(getCordFromRefPolar(np.array([2, 3, math.pi / 2]), np.array([math.pi, 1])), \
                               np.array([2, 2]), 1e-07, 1e-07)

  def testRoundToResolution(self):
    np.testing.assert_allclose(roundToResolution(np.array([0, 0.5, 0.9, 1.1, 1.5]), 2.0), np.array([0, 0, 0, 2, 2]))

  def testGetDist(self):
    self.assertEqual(getDist([0.0, 0.0], [1.0, 0.0]), 1.0)
    self.assertEqual(getDist([0.0, 0.0], [0.0, -1.0]), 1.0)
    self.assertEqual(getDist([0.0, 3.0], [0.0, -1.0]), 4.0)

  def testGetAng(self):
    self.assertEqual(getAng([0.0, 0.0], [1.0, 0.0]), 0.0)
    self.assertEqual(getAng([0.0, 0.0], [0.0, -1.0]), -math.pi / 2)
    self.assertEqual(getAng([0.0, 3.0], [0.0, 5.0]), math.pi / 2)


# Functionality generating the occupancy sparse grid from a lidar scan and scan from occupancy sparse grid
def getDefaultOccupancyConfig():
  return {
    "resolution": 20,
    "max_value": 1000,
    "min_value": -1000,
    "old_multiplier": 0.999,
    "old_historisis": 100.0,
    "update_multiplier": 0.990,
    "data": {},
    "robot_cord": np.array([0.0, 0.0, 0.0])
  }

def copyOccupancyConfig(occupancy):
  return {
    "resolution": occupancy["resolution"],
    "max_value": occupancy["max_value"],
    "min_value": occupancy["min_value"],
    "old_multiplier": occupancy["old_multiplier"],
    "old_historisis": occupancy["old_historisis"],
    "update_multiplier": occupancy["update_multiplier"],
    "data": {},
    "robot_cord": np.array([0.0, 0.0, 0.0])
  }

def getScanFromOccupancy(occupancy, robot_cord, resolution, deg_step):
  scan = []
  for deg in np.arange(-180.0, 180.0, deg_step):
    dist = 0
    done = 0
    while done != 1:
      dist = dist + resolution
      pt_pair = tuple(roundToResolution(getCordFromRefPolar(robot_cord, [degToRad(deg), dist]), resolution))
      if pt_pair not in occupancy["data"]:
        done = 1
      elif occupancy["data"][pt_pair] > 0:
        scan.append([1, deg, dist])
        done = 1
  return scan

def getOccupancyFromScan(scan, robot_cord, resolution, occupancy):
  for pt in scan:
    pt_pair = tuple(roundToResolution(getCordFromRefPolar(robot_cord, [degToRad(pt[1]), pt[2]]), resolution))
    occupancy["data"][pt_pair] = 1

  for pt in scan:
    pt_start = getCordFromRefPolar(robot_cord, [degToRad(pt[1]), pt[2]])
    pt_end = robot_cord
    ang = getAng(pt_start, pt_end)
    len = getDist(pt_start, pt_end)

    for s in np.arange(0, len, resolution):
      pt_pair = tuple(roundToResolution((pt_start[0] + s * math.cos(ang), pt_start[1] + s * math.sin(ang)), resolution))
      if pt_pair not in occupancy["data"]:
        occupancy["data"][pt_pair] = -1

  return occupancy

class TestOccupancyConversion(unittest.TestCase):
  def testOccupancyConversion(self):
    occupancy = getDefaultOccupancyConfig()
    occupancy["robot_cord"] = np.array([0, 0, 0])
    occupancy["resolution"] = 50.0
    resolution = occupancy["resolution"]

    r = 2000.0
    for x in np.arange(-r, r, resolution):
      for y in np.arange(-r, r, resolution):
        occupancy["data"][tuple(roundToResolution([x, y], resolution))] = -1

    for c in np.arange(-r, r, resolution):
      occupancy["data"][tuple(roundToResolution([c, -r], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([c, r], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([-r, c], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([r, c], resolution))] = 1

    deg_step = 5;
    scan = getScanFromOccupancy(occupancy, occupancy["robot_cord"], occupancy["resolution"], deg_step)
    self.assertEqual(len(scan), 360.0 / deg_step - 1)
    for v in scan:
      self.assertGreaterEqual(v[1], -180.0)
      self.assertLessEqual(v[1], 180.0)
      self.assertGreater(v[2], r - resolution)
      self.assertLess(v[2], r * math.sqrt(2) * r + resolution)

    new_occupancy = getDefaultOccupancyConfig()
    new_occupancy["robot_cord"] = occupancy["robot_cord"]
    new_occupancy["resolution"] = occupancy["resolution"]
    getOccupancyFromScan(scan, new_occupancy["robot_cord"], new_occupancy["resolution"], new_occupancy)

    for key in new_occupancy["data"]:
      self.assertTrue(key in occupancy["data"], "k=" + str(key))
      self.assertTrue((occupancy["data"][key] > 0) == (new_occupancy["data"][key] > 0), \
                      "k=" + str(key) + \
                      " o=" + str(occupancy["data"][key]) + \
                      " n = " + str(new_occupancy["data"][key]))

# Map Update Functionality
def updateOccupancy(occupancy_old, occupancy_new):
  old_multiplier = occupancy_new["old_multiplier"]
  old_historisis = occupancy_new["old_historisis"]
  update_multiplier = occupancy_new["update_multiplier"]
  min_value = occupancy_new["min_value"]
  max_value = occupancy_new["max_value"]

  for key in occupancy_old["data"]:
    if abs(occupancy_old["data"][key]) > old_historisis:
      occupancy_old["data"][key] = max(min(occupancy_old["data"][key] * old_multiplier, max_value), min_value)

  for key in occupancy_new["data"]:
    if key not in occupancy_old["data"]:
      occupancy_old["data"][key] = occupancy_new["data"][key]
    else:
      occupancy_old["data"][key] = max(min(occupancy_old["data"][key] * update_multiplier + occupancy_new["data"][key], max_value), min_value)

  return occupancy_old

class TestOccupancyUpdate(unittest.TestCase):
  def testOccupancyUpdate(self):
    occupancy1 = getDefaultOccupancyConfig()
    occupancy1["data"][(0, 0)] = 10
    occupancy1["data"][(1, 0)] = 20
    occupancy1["data"][(0, 2)] = 30
    occupancy1["data"][(0, 5)] = 40
    occupancy1["data"][(3, 3)] = -50
    occupancy1["data"][(3, 4)] = -60

    occupancy2 = getDefaultOccupancyConfig()
    occupancy2["data"][(0, 0)] = 10
    occupancy2["data"][(1, 0)] = -10
    # occupancy2["data"]((0, 2)] = not populated
    occupancy2["data"][(0, 5)] = 4
    occupancy2["data"][(3, 3)] = 10
    occupancy2["data"][(5, 5)] = -10

    occupancy2["old_multiplier"] = 0.9
    occupancy2["old_historisis"] = 15
    occupancy2["update_multiplier"] = 0.7

    occupancy_res = updateOccupancy(occupancy1, occupancy2)
    self.assertAlmostEqual(occupancy_res["data"][(0, 0)], 10.0 * 0.7 + 10.0)
    self.assertAlmostEqual(occupancy_res["data"][(1, 0)], 20.0 * 0.9 * 0.7 - 10.0)
    self.assertAlmostEqual(occupancy_res["data"][(0, 2)], 30.0 * 0.9)
    self.assertAlmostEqual(occupancy_res["data"][(0, 5)], 40.0 * 0.9 * 0.7 + 4.0)
    self.assertAlmostEqual(occupancy_res["data"][(3, 3)], -50.0 * 0.9 * 0.7 + 10.0)
    self.assertAlmostEqual(occupancy_res["data"][(3, 4)], -60.0 * 0.9)
    self.assertAlmostEqual(occupancy_res["data"][(5, 5)], -10.0)

# Localization Functionality
def scoreOccupancyCandidate(occupancy_old, occupancy_new, dx = 0.0, dy = 0.0):
  score = 0.0

  scoreMatchingWalls = 1.0
  scoreMatchingEmpty = 0.5
  scoreNewWall = 0.0
  scoreOldWall = 0.0

  for key_new in occupancy_new["data"]:
    key_old = tuple(roundToResolution([key_new[0] - dx, key_new[1] - dy], occupancy_old["resolution"]))
    if key_old not in occupancy_old["data"]:
      continue
    if occupancy_new["data"][key_new] > 0 and occupancy_old["data"][key_old] > 0:
      score = score + scoreMatchingWalls
    elif occupancy_new["data"][key_new] < 0 and occupancy_old["data"][key_old] < 0:
      score = score + scoreMatchingEmpty
    elif occupancy_new["data"][key_new] > 0 and occupancy_old["data"][key_old] < 0:
      score = score + scoreNewWall
    elif occupancy_new["data"][key_new] < 0 and occupancy_old["data"][key_old] > 0:
      score = score + scoreOldWall

  return score

def evaluateCandidateForScan(occupancy_old, robot_cord, scan):
  occupancy_new = copyOccupancyConfig(occupancy_old)
  occupancy_new["robot_cord"] = robot_cord
  getOccupancyFromScan(scan, occupancy_new["robot_cord"], occupancy_new["resolution"], occupancy_new)
  return scoreOccupancyCandidate(occupancy_old, occupancy_new)

def scoreOccupancyCandidates(occupancy_old, occupancy_new, dxs = [0], dys = [0]):
  results = {}
  for dx in dxs:
    for dy in dys:
      results[tuple([dx, dy])] = scoreOccupancyCandidate(occupancy_old, occupancy_new, dx, dy)
  return results

def localize_linear(occupancy_old, robot_cord, max_dx, step_dx, max_dy, step_dy, max_dt, step_dt, scan):
  best_robot_cord = robot_cord.copy()
  best_score = -1e7

  cord_list = []
  cord_list.append(robot_cord)
  for dx in np.arange(-max_dx, max_dx, step_dx):
    for dy in np.arange(-max_dy, max_dy, step_dy):
      for dt in np.arange(-max_dt, max_dt, step_dt):
        cord_list.append(np.array([robot_cord[0] + dx, robot_cord[1] + dy, normalizeAngle(robot_cord[2] + dt)]))

  for candidate_robot_cord in cord_list:
    score = evaluateCandidateForScan(occupancy_old, candidate_robot_cord, scan)
    if score > best_score:
      best_score = score
      best_robot_cord = candidate_robot_cord

  return best_robot_cord

def localize_parallel(occupancy_old, robot_cord, max_dx, step_dx, max_dy, step_dy, max_dt, step_dt, scan):
  best_robot_cord = robot_cord.copy()
  best_score = -1e7

  cord_list = []
  cord_list.append(robot_cord)
  for dx in np.arange(-max_dx, max_dx, step_dx):
    for dy in np.arange(-max_dy, max_dy, step_dy):
      for dt in np.arange(-max_dt, max_dt, step_dt):
        cord_list.append(np.array([robot_cord[0] + dx, robot_cord[1] + dy, normalizeAngle(robot_cord[2] + dt)]))

  pool_obj = multiprocessing.Pool()
  results = pool_obj.starmap(evaluateCandidateForScan, zip(repeat(occupancy_old), cord_list, repeat(scan)))
  pool_obj.close()
  for [candidate_coord, score] in zip(cord_list, results):
    if score > best_score:
      best_score = score
      best_robot_cord = candidate_coord

  return best_robot_cord

def localize_parallel_shifts(occupancy_old, robot_cord, max_dx, step_dx, max_dy, step_dy, max_dt, step_dt, scan):
  best_robot_cord = robot_cord.copy()
  best_score = -1e7

  dts = np.arange(-max_dt, max_dt, step_dt)
  rotated_cords = []
  rotated_occupancy = []
  for dt in dts:
    rotated_cords.append(np.array([robot_cord[0], robot_cord[1], normalizeAngle(robot_cord[2] + dt)]))
    rotated_occupancy.append(copyOccupancyConfig(occupancy_old))

  pool_obj = multiprocessing.Pool()
  rotated_occupancy = pool_obj.starmap(getOccupancyFromScan, \
                   zip(repeat(scan), rotated_cords, repeat(occupancy_old["resolution"]), rotated_occupancy))
  results = pool_obj.starmap(scoreOccupancyCandidates, \
                   zip(repeat(occupancy_old), rotated_occupancy, \
                   repeat(np.arange(-max_dx, max_dx, step_dx)), \
                   repeat(np.arange(-max_dy, max_dy, step_dy))))
  pool_obj.close()

  for [result, rotated_cord] in zip(results, rotated_cords):
    for offsets in result.keys():
      score = result[offsets]
      if score > best_score:
        best_score = score
        best_robot_cord = np.array([rotated_cord[0] - offsets[0], rotated_cord[1] - offsets[1], rotated_cord[2]])

  return best_robot_cord


def localize(occupancy_old, robot_cord, max_dx, step_dx, max_dy, step_dy, max_dt, step_dt, scan, method = 0):
  if method == 1:
    return localize_linear(occupancy_old, robot_cord, max_dx, step_dx, max_dy, step_dy, max_dt, step_dt, scan)
  elif method == 2:
    return localize_parallel(occupancy_old, robot_cord, max_dx, step_dx, max_dy, step_dy, max_dt, step_dt, scan)
  else:
    return localize_parallel_shifts(occupancy_old, robot_cord, max_dx, step_dx, max_dy, step_dy, max_dt, step_dt, scan)

class TestLocalization(unittest.TestCase):
  def testScoreOccupancyCandidate(self):
    occupancy = getDefaultOccupancyConfig()
    occupancy["robot_cord"] = np.array([0, 0, 0])
    occupancy["resolution"] = 50.0
    resolution = occupancy["resolution"]

    r = 2000.0
    for x in np.arange(-r, r, resolution):
      for y in np.arange(-r, r, resolution):
        occupancy["data"][tuple(roundToResolution([x, y], resolution))] = -1

    for c in np.arange(-r, r, resolution):
      occupancy["data"][tuple(roundToResolution([c, -r], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([c, r], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([-r, c], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([r, c], resolution))] = 1

    deg_step = 1.0

    # This is without error, so score should be high.
    scan = getScanFromOccupancy(occupancy, np.array([0.0, 0.0, 0.0]), resolution, deg_step)
    occupancy_new = copyOccupancyConfig(occupancy)
    getOccupancyFromScan(scan, occupancy_new["robot_cord"], occupancy_new["resolution"], occupancy_new)
    score = scoreOccupancyCandidate(occupancy, occupancy_new)
    self.assertGreater(score, 10.0)
    print("original score", score)

    # shift the origin, score should be less
    scan = getScanFromOccupancy(occupancy, np.array([resolution * 2, resolution * 4, 0.0]), resolution, deg_step)
    occupancy_new = copyOccupancyConfig(occupancy)
    getOccupancyFromScan(scan, occupancy_new["robot_cord"], occupancy_new["resolution"], occupancy_new)
    score_shifted = scoreOccupancyCandidate(occupancy, occupancy_new)
    self.assertGreater(score, score_shifted)
    print("shifted score", score_shifted)

    # rotate, score should be less
    scan = getScanFromOccupancy(occupancy, np.array([0.0, 0.0, math.pi / 40.0]), resolution, deg_step)
    occupancy_new = copyOccupancyConfig(occupancy)
    getOccupancyFromScan(scan, occupancy_new["robot_cord"], occupancy_new["resolution"], occupancy_new)
    score_rotated = scoreOccupancyCandidate(occupancy, occupancy_new)
    self.assertGreater(score, score_rotated)
    print("rotated score", score_rotated)

    # more rotation should result in lower score
    scan = getScanFromOccupancy(occupancy, np.array([0.0, 0.0, math.pi / 20.0]), resolution, deg_step)
    occupancy_new = copyOccupancyConfig(occupancy)
    getOccupancyFromScan(scan, occupancy_new["robot_cord"], occupancy_new["resolution"], occupancy_new)
    score_rotated_more = scoreOccupancyCandidate(occupancy, occupancy_new)
    self.assertGreater(score_rotated, score_rotated_more)
    print("rotated more score", score_rotated_more)

  def testLocalize(self):
    occupancy = getDefaultOccupancyConfig()
    occupancy["robot_cord"] = np.array([0, 0, 0])
    occupancy["resolution"] = 50.0
    resolution = occupancy["resolution"]

    r = 2000.0
    for x in np.arange(-r, r, resolution):
      for y in np.arange(-r, r, resolution):
        occupancy["data"][tuple(roundToResolution([x, y], resolution))] = -1

    for c in np.arange(-r, r, resolution):
      occupancy["data"][tuple(roundToResolution([c, -r], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([c, r], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([-r, c], resolution))] = 1
      occupancy["data"][tuple(roundToResolution([r, c], resolution))] = 1

    deg_step = 5.0
    real_cord = np.array([50.0, -50.0, math.pi / 8.0])

    scan = getScanFromOccupancy(occupancy, real_cord, resolution, deg_step)
    for m in [0, 1, 2]:
      robot_cord = localize(occupancy, np.array([0.0, 0.0, 0.0]), resolution * 4, resolution, resolution * 4, resolution, math.pi / 4.0, math.pi / 8.0, scan, m)
      np.testing.assert_allclose(robot_cord, real_cord, 1e-07, 1e-07)

    print("expected", str(real_cord), " got ", str(robot_cord))

if __name__ == '__main__':
  unittest.main()
