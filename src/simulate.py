# Simulation
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy
import pickle

from slam.slam_geometry import normalize_angle
from slam.slam_kinematics import *
from slam.slam_plots import PlotData
from slam.slam_planner import *
from slam.slam_robot import SimulatedRobot
from slam.slam import Slam

def get_map():
  return [
    ((100.0, 100.0), (200.0, 100.0)),
    ((200.0, 100.0), (200.0, 150.0)),
    ((200.0, 150.0), (300.0, 150.0)),
    ((300.0, 150.0), (300.0, 100.0)),
    ((300.0, 100.0), (500.0, 100.0)),
    ((500.0, 100.0), (500.0, 500.0)),
    ((500.0, 500.0), (100.0, 500.0)),
    ((100.0, 500.0), (100.0, 100.0))
  ]

def run(repeat = True, frames = 100, animate = True):
  # Define the environment
  segments = get_map()

  # Initial position and parameters of the robot
  initial_position = [400.0, 250.0, numpy.deg2rad(35.0)]
  move_error = 2.0
  turn_error = numpy.deg2rad(2.0)
  sensor_noise = 5.0
  sensor_angle_std_dev = numpy.deg2rad(5.0)

  # Initialize the robot
  robot = SimulatedRobot(initial_position = initial_position, \
                         segments = segments, \
                         noise_std_dev = move_error, \
                         angle_noise_std_dev = turn_error, \
                         sensor_noise = sensor_noise, \
                         sensor_angle_std_dev = sensor_angle_std_dev, \
                         num_points = 100, \
                         field_of_view = numpy.deg2rad(360.0), \
                         max_distance = 1000.0)

  # Initialize the mapping environment.
  mapping = Slam(initial_position=initial_position, \
                 robot_covariance=[[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, numpy.deg2rad(25.0)]], \
                 num_points=20,
                 segments = segments)

  # Set up the plotting
  plot_data = PlotData()
  fig1, (map_plot, graph_plot1, graph_plot2, graph_plot3, graph_plot4) = \
    plt.subplots(5, gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]})
  plt.subplots_adjust(wspace=0, hspace=0)

  drive_type = 1

  def handle_robot():

    # Determine if we are moving or staying put, on large error do not move
    if drive_type == 0:
      # Pick the movement direction and angle
      move_distance, turn_angle = plan_motion_from_lidar(mapping.lidar_points, mapping.localized)

      # Move the robot
      robot.move(distance = move_distance)
      robot.rotate(rotation = turn_angle)

      # Inform mapping of motion
      mapping.move_robot(move_distance = move_distance, distance_error = move_error)
      mapping.rotate_robot(rotate_angle = turn_angle, rotation_error = turn_error)

    elif drive_type == 1:
      old_left = robot.left
      old_right = robot.right

      # Pick the speeds for the motors
      left, right = plan_speeds_from_lidar(mapping.lidar_points, mapping.localized)

      # Move the robot
      robot.advance_time(dt = 1.0)
      robot.set_speed(left, right)

      # Inform mapping of motion
      turn1, dist2, turn3 = get_turn_move_turn_from_differential_drive(vL = old_left, vR = old_right, base = robot.wheel_base, dt = 1.0)
      if turn1 != 0.0:
        mapping.rotate_robot(rotate_angle = turn1, rotation_error = turn_error)
      if dist2 != 0.0:
        mapping.move_robot(move_distance = dist2, distance_error = move_error)
      if turn3 != 0.0:
        mapping.rotate_robot(rotate_angle = turn3, rotation_error = turn_error)

    # update the mapping environment based on lidar data
    lidar_data = robot.sense_environment()
    mapping.lidar_update(lidar_data)

  def update(frame):
    if frame == 0:
      print(mapping.to_json())

    handle_robot()

    if animate:
      plot_data.collect_data(mapping = mapping, robot = robot)
      plot_data.truncate(100)

      map_plot.cla()
      map_plot.set_aspect('equal')
      map_plot.set_xlim([0, 600])
      map_plot.set_ylim([0, 600])
      plot_data.plot_reality(robot = robot, map_plot = map_plot)
      plot_data.plot_mapping(mapping = mapping, map_plot = map_plot)

      graph_plot1.cla()
      graph_plot1.grid()
      graph_plot2.cla()
      graph_plot2.grid()
      graph_plot3.cla()
      graph_plot3.grid()
      graph_plot4.cla()
      graph_plot4.grid()
      plot_data.plot_graphs(map_plot, graph_plot1, graph_plot2, graph_plot3, graph_plot4)

    if not repeat and frame + 1 == frames:
      plt.close()  # Close the plot window
      sys.exit()  # Exit Python when the animation is complete

  # start the animation
  ani = animation.FuncAnimation(fig1, update, frames=frames, repeat=repeat, cache_frame_data=False, interval=1)

  #writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  #ani.save('animation.gif', writer=writer)

  plt.show()

if __name__ == '__main__':
  if 'profile' in sys.argv:
    import cProfile
    cProfile.run('run(repeat = False, frames = 500, animate = False)')
  else:
    run(repeat = True, frames = 100, animate = True)
