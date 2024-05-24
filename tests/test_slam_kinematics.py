import math
import numpy
import unittest

from slam.slam_kinematics import *

class TestBasicMethods(unittest.TestCase):
  def test_apply_turn_move_turn(self):
    x1, y1, theta1 = apply_turn_move_turn(x0 = 0.0, y0 = 0.0, theta0 = 0.0, turn1 = 0.0, move = 0.0, turn2 = 0.0)
    self.assertAlmostEqual(x1, 0.0)
    self.assertAlmostEqual(y1, 0.0)
    self.assertAlmostEqual(theta1, 0.0)

    x1, y1, theta1 = apply_turn_move_turn(x0 = -1.0, y0 = 2.0, theta0 = 1.0, turn1 = 0.5, move = 0.0, turn2 = 0.2)
    self.assertAlmostEqual(x1, -1.0)
    self.assertAlmostEqual(y1, 2.0)
    self.assertAlmostEqual(theta1, 1.0 + 0.5 + 0.2)

    x1, y1, theta1 = apply_turn_move_turn(x0 = -1.0, y0 = 2.0, theta0 = numpy.deg2rad(30), \
                                          turn1 = numpy.deg2rad(60), move = 1.0, turn2 = numpy.deg2rad(30))
    self.assertAlmostEqual(x1, -1.0)
    self.assertAlmostEqual(y1, 3.0)
    self.assertAlmostEqual(theta1, numpy.deg2rad(120))

  def test_get_turn_move_turn(self):
    turn1, move1, turn2 = get_turn_move_turn(x0 = 0.0, y0 = 0.0, theta0 = 0.0, \
                                             x1 = 0.0, y1 = 0.0, theta1 = 0.0, backward = False)
    self.assertAlmostEqual(turn1, 0.0)
    self.assertAlmostEqual(move1, 0.0)
    self.assertAlmostEqual(turn2, 0.0)

    turn1, move1, turn2 = get_turn_move_turn(x0 = 3.0, y0 = 2.0, theta0 = 0.0, \
                                             x1 = 4.0, y1 = 2.0, theta1 = 1.0, backward = False)
    self.assertAlmostEqual(turn1, 0.0)
    self.assertAlmostEqual(move1, 1.0)
    self.assertAlmostEqual(turn2, 1.0)

    turn1, move1, turn2 = get_turn_move_turn(x0 = 3.0, y0 = 2.0, theta0 = 0.0, \
                                             x1 = 4.0, y1 = 2.0, theta1 = 1.0, backward = True)
    self.assertAlmostEqual(turn1, numpy.deg2rad(180))
    self.assertAlmostEqual(move1, 1.0)
    self.assertAlmostEqual(turn2, normalize_angle(numpy.deg2rad(180) + 1.0))

  def test_differential_drive_stopped(self):
    x, y, th = differential_drive(x0 = 0.0, y0 = 0.0, theta0 = 0.0, vL = 0.0, vR = 0.0, base = 10.0, dt = 1.0)
    self.assertEqual(x, 0.0)
    self.assertEqual(y, 0.0)
    self.assertEqual(th, 0.0)

    x, y, th = differential_drive(x0 = 1.0, y0 = 2.0, theta0 = 5.0, vL = 0.0, vR = 0.0, base = 10.0, dt = 1.0)
    self.assertEqual(x, 1.0)
    self.assertEqual(y, 2.0)
    self.assertEqual(th, 5.0)

  def test_differential_drive_forward(self):
    x, y, th = differential_drive(x0 = 0.0, y0 = 0.0, theta0 = 0.0, vL = 1.0, vR = 1.0, base = 10.0, dt = 1.0)
    self.assertAlmostEqual(x, 1.0)
    self.assertAlmostEqual(y, 0.0)
    self.assertAlmostEqual(th, 0.0)

    x, y, th = differential_drive(x0 = 1.0, y0 = 2.0, theta0 = numpy.deg2rad(90), vL = 1.0, vR = 1.0, \
                                  base = 10.0, dt = 2.0)
    self.assertAlmostEqual(x, 1.0)
    self.assertAlmostEqual(y, 4.0)
    self.assertAlmostEqual(th, numpy.deg2rad(90))

  def test_differential_drive_turn_on_spot(self):
    x, y, th = differential_drive(x0 = 1.0, y0 = 2.0, theta0 = 3.0, vL = 1.0, vR = -1.0, base = 10.0, dt = 1.0)
    self.assertAlmostEqual(x, 1.0)
    self.assertAlmostEqual(y, 2.0)
    self.assertAlmostEqual(th, 3-0.2)

    x, y, th = differential_drive(x0 = 2.0, y0 = 1.0, theta0 = 2.0, vL = -1.0, vR = 1.0, base = 10.0, dt = 1.0)
    self.assertAlmostEqual(x, 2.0)
    self.assertAlmostEqual(y, 1.0)
    self.assertAlmostEqual(th, 2+0.2)

  def test_differential_drive_drive_and_turn(self):
    x, y, th = differential_drive(x0 = 1.0, y0 = 2.0, theta0 = 1.0, vL = 1.0, vR = 2.0, base = 10.0, dt = 1.0)
    self.assertAlmostEqual(x, 1.7460456288030837)
    self.assertAlmostEqual(y, 3.3005927666384367)
    self.assertAlmostEqual(th, 1.0 + 0.1)

if __name__ == '__main__':
  unittest.main()
