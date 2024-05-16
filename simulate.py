# Simulation
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy

from slam_geometry import normalize_angle
from slam import MappingEnvironment
from robot import SimulatedRobot
from plots import PlotData

if __name__ == '__main__':
  # Define the environment
  segments = [
    ((100.0, 100.0), (200.0, 100.0)),
    ((200.0, 100.0), (200.0, 150.0)),
    ((200.0, 150.0), (300.0, 150.0)),
    ((300.0, 150.0), (300.0, 100.0)),
    ((300.0, 100.0), (500.0, 100.0)),
    ((500.0, 100.0), (500.0, 500.0)),
    ((100.0, 100.0), (100.0, 500.0)),
    ((100.0, 500.0), (500.0, 500.0))
  ]

  initial_position = [400.0, 250.0, numpy.deg2rad(35.0)]

  # type of noise we expect from movement
  move_error = 1.0
  turn_error = numpy.deg2rad(1.0)

  # Initialize the robot
  robot = SimulatedRobot(initial_position = initial_position, \
                         segments = segments, \
                         noise_std_dev = move_error, \
                         angle_noise_std_dev = turn_error, \
                         sensor_noise = 1.0, \
                         sensor_angle_std_dev = numpy.deg2rad(1.0), \
                         num_points = 100, \
                         field_of_view = numpy.deg2rad(360.0), \
                         max_distance = 1000.0)

  # Initialize the mapping environment.
  mapping = MappingEnvironment(initial_position=initial_position, \
                               robot_covariance=[[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, numpy.deg2rad(25.0)]], \
                               num_points=50,
                               segments = segments)

  # Set up the plotting
  fig1, (map_plot, graph_plot1, graph_plot2, graph_plot3, graph_plot4) = \
    plt.subplots(5, gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]})

  # Configuration flags for what to plot
  show_real_position = True
  show_real_map = True

  show_mapped_position_all = True
  show_mapped_position_mean = True
  show_mapped_lidar = True
  show_mapped_map = True
  show_past_path = True

  use_localize = True
  scale_points = False

  plot_data = PlotData()

  def handle_robot():
    # Pick the movement direction and angle
    move_distance = numpy.random.uniform(0.0, 10.0)
    turn_angle = numpy.random.uniform(numpy.deg2rad(-3.0), numpy.deg2rad(10.0))

    # Determine if we are moving or staying put, on large error do not move
    if use_localize and not mapping.localized:
      move_distance = 0
      turn_angle = 0

    # if we want to play around with number of sample points
    if use_localize and scale_points:
      if mapping.localized and len(mapping.poses) > 10:
        mapping.reinitialize_particles(len(mapping.poses) - 1)
      elif not mapping.localized and len(mapping.poses) < 100:
        mapping.reinitialize_particles(len(mapping.poses) + 1)

    # If we chose to move or turn, then move the robot and let mapping know
    if move_distance != 0.0 or turn_angle != 0.0:
      moved = robot.move(distance = move_distance, rotation = turn_angle)
      if not moved:
        move_distance = 0.0
      mapping.move_robot(move_distance = move_distance, rotate_angle = turn_angle, \
                                     distance_error = move_error, rotation_error = turn_error)

    # update the mapping invironment based on lidar data
    if use_localize:
      mapping.lidar_update(robot.sense_environment())

  def update(frame):
    handle_robot()

    plot_data.collect_data(mapping = mapping, robot = robot)
    plot_data.truncate(100)

    map_plot.cla()
    map_plot.set_aspect('equal')
    plot_data.plot_reality(robot = robot, map_plot = map_plot)
    plot_data.plot_mapping(mapping = mapping, map_plot = map_plot)

    graph_plot1.cla()
    graph_plot2.cla()
    graph_plot3.cla()
    graph_plot4.cla()
    plot_data.plot_graphs(map_plot, graph_plot1, graph_plot2, graph_plot3, graph_plot4)

  ani = animation.FuncAnimation(fig1, update, frames=100, repeat=True, cache_frame_data=False, interval=100)
  plt.show()
