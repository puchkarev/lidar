# Simulation
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy
import pickle

from slam.slam_geometry import normalize_angle
from slam.slam_plots import PlotData
from slam.slam_planner import plan_motion_from_lidar
from slam.slam_robot import SimulatedRobot
from slam.slam import Slam

def save_map():
  segments = [
    ((100.0, 100.0), (200.0, 100.0)),
    ((200.0, 100.0), (200.0, 150.0)),
    ((200.0, 150.0), (300.0, 150.0)),
    ((300.0, 150.0), (300.0, 100.0)),
    ((300.0, 100.0), (500.0, 100.0)),
    ((500.0, 100.0), (500.0, 500.0)),
    ((500.0, 500.0), (100.0, 500.0)),
    ((100.0, 500.0), (100.0, 100.0))
  ]
  with open('sample_map.pkl', 'wb') as f:
    pickle.dump(segments, f)

def run(repeat = True, frames = 100, animate = True):
  # Define the environment
  segments = pickle.load(open('sample_map.pkl', 'rb'))

  # Initial position and parameters of the robot
  initial_position = [400.0, 250.0, numpy.deg2rad(35.0)]
  move_error = 3.0
  turn_error = numpy.deg2rad(2.0)
  sensor_noise = 5.0
  sensor_angle_std_dev = numpy.deg2rad(2.0)

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
                 num_points=10,
                 segments = segments)

  # Set up the plotting
  plot_data = PlotData()
  fig1, (map_plot, graph_plot1, graph_plot2, graph_plot3, graph_plot4) = \
    plt.subplots(5, gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]})
  plt.subplots_adjust(wspace=0, hspace=0)

  localize_steps = 1 # how many steps we should use for localizing on single lidar data set
  scale_points = False # should we adjust the number of particles for localization
  use_motion = True # should we inform mapping of robot motion

  def handle_robot():
    # Pick the movement direction and angle
    move_distance, turn_angle = plan_motion_from_lidar(robot.sense_environment())

    # This woudl be random movement and angle
    #move_distance = numpy.random.uniform(0.0, 10.0)
    #turn_angle = numpy.random.uniform(numpy.deg2rad(-3.0), numpy.deg2rad(10.0))

    # Determine if we are moving or staying put, on large error do not move
    if localize_steps > 0 and not mapping.localized:
      move_distance = 0
      turn_angle = 0

    # if we want to play around with number of sample points
    if localize_steps > 0 and scale_points:
      if mapping.localized and len(mapping.poses) > 10:
        mapping.reinitialize_particles(len(mapping.poses) - 1)
      elif not mapping.localized and len(mapping.poses) < 100:
        mapping.reinitialize_particles(len(mapping.poses) + 1)

    # If we chose to move or turn, then move the robot and let mapping know
    if move_distance != 0.0 or turn_angle != 0.0:
      moved = robot.move(distance = move_distance, rotation = turn_angle)
      if not moved:
        move_distance = 0.0
      if use_motion:
        mapping.move_robot(move_distance = move_distance, rotate_angle = turn_angle, \
                           distance_error = move_error, rotation_error = turn_error)

    # update the mapping invironment based on lidar data
    lidar_data = robot.sense_environment()
    for _ in range(localize_steps):
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
  plt.show()

if __name__ == '__main__':
  if 'profile' in sys.argv:
    import cProfile
    cProfile.run('run(repeat = False, frames = 100, animate = False)')
  else:
    run(repeat = True, frames = 100, animate = True)
