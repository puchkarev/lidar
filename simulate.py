# Simulation
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy

from slam_geometry import normalize_angle
from slam import MappingEnvironment
from simulated_robot import SimulatedRobot

class PlotData:
  def __init__(self):
    self.vals_frame_nums = []
    self.vals_pose_error = []
    self.vals_angle_error = []
    self.vals_pose_std_dev = []
    self.vals_angle_std_dev = []
    self.vals_localized = []

  def grow_list(self):
    if len(self.vals_frame_nums) == 0:
      self.vals_frame_nums.append(0)
    else:
      self.vals_frame_nums.append(max(self.vals_frame_nums) + 1)

  def truncate(self):
    want_elements = 100
    if len(self.vals_frame_nums) > want_elements:
      self.vals_frame_nums = self.vals_frame_nums[-want_elements:]
      self.vals_pose_error = self.vals_pose_error[-want_elements:]
      self.vals_angle_error = self.vals_angle_error[-want_elements:]
      self.vals_pose_std_dev = self.vals_pose_std_dev[-want_elements:]
      self.vals_angle_std_dev = self.vals_angle_std_dev[-want_elements:]
      self.vals_localized = self.vals_localized[-want_elements:]

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
                               num_points=100,
                               segments = segments)

  # Set up the plotting
  fig1, (map_plot, graph_plot1, graph_plot2, graph_plot3) = \
    plt.subplots(4, gridspec_kw={'height_ratios': [4, 1, 1, 1]})

  # Configuratino flags for what to plot
  show_real_position = True
  show_real_map = True

  show_mapped_position_all = True
  show_mapped_position_mean = True
  show_mapped_lidar = True
  show_mapped_map = True

  plot_data = PlotData()

  def handle_robot():
    # Pick the movement direction and angle
    move_distance = numpy.random.uniform(0.0, 5.0)
    turn_angle = numpy.random.uniform(numpy.deg2rad(-1.0), numpy.deg2rad(5.0))

    # Determine if we are moving or staying put, on large error do not move
    if not mapping.localized:
      move_distance = 0
      turn_angle = 0

    # If we chose to move or turn, then move the robot and let mapping know
    if move_distance != 0.0 or turn_angle != 0.0:
      robot.move(distance = move_distance, rotation = turn_angle)
      mapping.move_robot(move_distance = move_distance, rotate_angle = turn_angle, \
                                     distance_error = move_error, rotation_error = turn_error)

    # update the mapping invironment based on lidar data
    mapping.lidar_update(robot.sense_environment())

  def plot_map_plot():
    # Re-initialize the plot
    map_plot.cla()
    map_plot.set_aspect('equal')
    map_plot.set_xlim(0, 600)
    map_plot.set_ylim(0, 600)

    # show the actual robot position
    if show_real_position:
      map_plot.plot([robot.position[0]], [robot.position[1]], 'bo', markersize=10)

    # show the real segments from the environment.
    if show_real_map:
      for segment in robot.segments:
        map_plot.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'k-')

    # show the robot positions
    if show_mapped_position_all:
      map_plot.plot([p[0] for p in mapping.poses], [p[1] for p in mapping.poses], 'ro', markersize=1)

    # show the mean of the robot position
    if show_mapped_position_mean:
      map_plot.plot([mapping.robot_mean[0]], [mapping.robot_mean[1]], 'ro', markersize=4)

    # show the lidar returns as understood by the mapping
    if show_mapped_lidar:
      map_plot.plot([p[0] for p in mapping.cartesian_points], [p[1] for p in mapping.cartesian_points], 'bo')

    # show the map as understood by the mapping
    if show_mapped_map:
      for segment_association in mapping.segment_associations:
        segment = segment_association[0]
        map_plot.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'g-')

      for segment in mapping.new_segments:
        map_plot.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r-')
      for corner_association in mapping.corner_associations:
        corner = corner_association[0]
        map_plot.plot([corner[0]], [corner[1]], 'go', markersize = 5)
      for corner in mapping.new_corners:
        map_plot.plot([corner[0]], [corner[1]], 'ro', markersize = 5)

  def plot_graphs():
    # Generate plots overtime of actual error
    plot_data.grow_list()
    plot_data.vals_pose_error.append(math.hypot(robot.position[0] - mapping.robot_mean[0], \
                                                robot.position[1] - mapping.robot_mean[1]))
    plot_data.vals_angle_error.append(numpy.rad2deg(normalize_angle(robot.position[2] - mapping.robot_mean[2])))
    plot_data.vals_pose_std_dev.append(math.sqrt(mapping.robot_covariance[0][0] + mapping.robot_covariance[1][1]))
    plot_data.vals_angle_std_dev.append(numpy.rad2deg(math.sqrt(mapping.robot_covariance[2][2])))
    if mapping.localized:
      plot_data.vals_localized.append(1)
    else:
      plot_data.vals_localized.append(0)

    plot_data.truncate()

    graph_plot1.cla()
    graph_plot1.plot(plot_data.vals_frame_nums, plot_data.vals_pose_error, 'k-',)
    graph_plot1.plot(plot_data.vals_frame_nums, plot_data.vals_pose_std_dev, 'r-')
    graph_plot1.legend(["pose_error", "pose_std_dev"])

    graph_plot2.cla()
    graph_plot2.plot(plot_data.vals_frame_nums, plot_data.vals_angle_error, 'k-')
    graph_plot2.plot(plot_data.vals_frame_nums, plot_data.vals_angle_std_dev, 'r-')
    graph_plot2.legend(["angle_error", "angle_std_dev"])

    graph_plot3.cla()
    graph_plot3.plot(plot_data.vals_frame_nums, plot_data.vals_localized, 'k-')
    graph_plot3.legend(["localized"])

  def update(frame):
    handle_robot()
    plot_map_plot()
    plot_graphs()

  ani = animation.FuncAnimation(fig1, update, frames=1000, interval=1)
  plt.show()
