import math
import numpy

from slam.slam_geometry import *

def apply_turn_move_turn(x0, y0, theta0, turn1, move, turn2):
  """
  Returns the position where robot ends up:
   starting from (x0, y0)
   then turning by turn1
   then moving by move
   then turning by turn2
  """
  return x0 + move * numpy.cos(theta0 + turn1), y0 + move * numpy.sin(theta0 + turn1), theta0 + turn1 + turn2

def get_turn_move_turn(x0, y0, theta0, x1, y1, theta1, backward = False):
  """
  Returns the values necessary for getting from (x0, y0, theta0) to (x1, y1, theta1) by
    turning turn1
    moving dist
    turning turn2
  """
  direction = numpy.arctan2(y1 - y0, x1 - x0)
  dist = math.hypot(y1 - y0, x1 - x0)
  if backward:
    direction = normalize_angle(direction + math.pi)
    dist = -dist

  return normalize_angle(direction - theta0), math.hypot(y1 - y0, x1 - x0), normalize_angle(theta1 - direction)

def get_turn_move_turn_from_differential_drive(vL, vR, base, dt):
  """
  Returns the turn, move, turn operation for a differential drive with speeds and wheel base
  """
  (x1, y1, th1) = differential_drive(x0 = 0.0, y0 = 0.0, theta0 = 0.0, vL = vL, vR = vR, base = base, dt = dt)
  return get_turn_move_turn(x0 = 0.0, y0 = 0.0, theta0 = 0.0, x1 = x1, y1 = y1, theta1 = th1, backward = False)

def differential_drive(x0, y0, theta0, vL, vR, base, dt):
  """Updates the position of the robot with differential drive"""
  if vL == vR:
      # Straight line motion
      V = (vL + vR) / 2
      x = x0 + V * dt * numpy.cos(theta0)
      y = y0 + V * dt * numpy.sin(theta0)
      theta = theta0
  else:
      # Circular motion
      omega = (vR - vL) / base
      R = base / 2 * (vL + vR) / (vR - vL)
      theta = theta0 + omega * dt
      x = x0 + R * (numpy.sin(theta) - numpy.sin(theta0))
      y = y0 - R * (numpy.cos(theta) - numpy.cos(theta0))

  return x, y, theta
