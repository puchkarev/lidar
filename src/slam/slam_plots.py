# Generates Plots
import math
import numpy

from slam.slam_geometry import *

class PlotData:
  def __init__(self):
    self.vals_frame_nums = []
    self.vals_pose_error = []
    self.vals_angle_error = []
    self.vals_pose_std_dev = []
    self.vals_angle_std_dev = []
    self.vals_particles = []
    self.vals_max_score = []
    self.vals_good_particles = []
    self.vals_matched_segments = []
    self.vals_total_segments = []
    self.vals_matched_corners = []
    self.vals_total_corners = []
    self.vals_localized = []
    self.vals_actual_pose_x = []
    self.vals_actual_pose_y = []
    self.vals_estimated_pose_x = []
    self.vals_estimated_pose_y = []
    self.vals_estimated_pose_t = []
    self.vals_move_speed = []
    self.vals_turn_speed = []

  def collect_data(self, mapping, robot):
    if len(self.vals_frame_nums) == 0:
      self.vals_frame_nums.append(0)
    else:
      self.vals_frame_nums.append(max(self.vals_frame_nums) + 1)

    if robot and mapping:
      self.vals_pose_error.append(math.hypot(robot.position[0] - mapping.robot_mean[0], \
                                             robot.position[1] - mapping.robot_mean[1]))
      self.vals_angle_error.append(numpy.rad2deg(normalize_angle(robot.position[2] - mapping.robot_mean[2])))

    if mapping:
      self.vals_pose_std_dev.append(math.sqrt(mapping.robot_covariance[0][0] + mapping.robot_covariance[1][1]))
      self.vals_angle_std_dev.append(numpy.rad2deg(math.sqrt(mapping.robot_covariance[2][2])))
      self.vals_particles.append(len(mapping.poses))
      self.vals_max_score.append(max(mapping.weights) * 100.0)
      self.vals_good_particles.append(len([m for m in mapping.weights if m > (1.0 / len(mapping.weights))]))
      self.vals_matched_segments.append(len(mapping.segment_associations))
      self.vals_total_segments.append(len(mapping.segment_associations) + len(mapping.new_segments))
      self.vals_matched_corners.append(len(mapping.corner_associations))
      self.vals_total_corners.append(len(mapping.corner_associations) + len(mapping.new_corners))
      self.vals_estimated_pose_x.append(mapping.robot_mean[0])
      self.vals_estimated_pose_y.append(mapping.robot_mean[1])
      self.vals_estimated_pose_t.append(mapping.robot_mean[2])
      if len(self.vals_estimated_pose_x) < 2:
        self.vals_move_speed.append(0.0)
      else:
        self.vals_move_speed.append(math.hypot(self.vals_estimated_pose_y[-1] - self.vals_estimated_pose_y[-2], \
                                               self.vals_estimated_pose_x[-1] - self.vals_estimated_pose_x[-2]))
      if len(self.vals_estimated_pose_t) < 2:
        self.vals_turn_speed.append(0.0)
      else:
        self.vals_turn_speed.append(numpy.rad2deg(normalize_angle(self.vals_estimated_pose_t[-1] - \
                                                                  self.vals_estimated_pose_t[-2])))

      if mapping.localized:
        self.vals_localized.append(100)
      else:
        self.vals_localized.append(0)

    if robot:
      self.vals_actual_pose_x.append(robot.position[0])
      self.vals_actual_pose_y.append(robot.position[1])

  def truncate(self, want_elements):
    def tr(list):
      if len(list) > want_elements:
        del list[:-want_elements]

    tr(self.vals_frame_nums)
    tr(self.vals_pose_error)
    tr(self.vals_angle_error)
    tr(self.vals_pose_std_dev)
    tr(self.vals_angle_std_dev)
    tr(self.vals_particles)
    tr(self.vals_max_score)
    tr(self.vals_good_particles)
    tr(self.vals_matched_segments)
    tr(self.vals_total_segments)
    tr(self.vals_matched_corners)
    tr(self.vals_total_corners)
    tr(self.vals_localized)
    tr(self.vals_actual_pose_x)
    tr(self.vals_actual_pose_y)
    tr(self.vals_estimated_pose_x)
    tr(self.vals_estimated_pose_y)
    tr(self.vals_estimated_pose_t)
    tr(self.vals_move_speed)
    tr(self.vals_turn_speed)

  def plot_reality(self, robot, map_plot):
    # show the actual map that the robot understands
    if True:
      for segment in robot.segments:
        map_plot.plot([segment[0][0], segment[1][0]], \
                      [segment[0][1], segment[1][1]], 'k-')

    # show the real robot position
    if True:
      face = 10.0
      map_plot.plot([robot.position[0]], \
                    [robot.position[1]], 'bo', markersize=7)
      map_plot.plot([robot.position[0], robot.position[0] + face * numpy.cos(robot.position[2])], \
                    [robot.position[1], robot.position[1] + face * numpy.sin(robot.position[2])], 'b-', markersize=1)

  def plot_mapping(self, mapping, map_plot):
    # show the robot position distribution
    if True:
      face = 5.0
      map_plot.plot([p[0] for p in mapping.poses], \
                    [p[1] for p in mapping.poses], 'ro', markersize=1)
      for p in mapping.poses:
        map_plot.plot([p[0], p[0] + face * numpy.cos(p[2])], \
                      [p[1], p[1] + face * numpy.sin(p[2])], 'r-', markersize=1)


    # show the mean of the distribution
    if True:
      face = 10.0
      map_plot.plot([mapping.robot_mean[0]], \
                    [mapping.robot_mean[1]], 'ro', markersize=5)
      map_plot.plot([mapping.robot_mean[0], mapping.robot_mean[0] + face * numpy.cos(mapping.robot_mean[2])], \
                    [mapping.robot_mean[1], mapping.robot_mean[1] + face * numpy.sin(mapping.robot_mean[2])], \
                     'r-', markersize=1)
      # Also draw where the lidar is and where it is facing
      lidar_pose = to_world_from_ref(mapping.lidar_offset, mapping.robot_mean)
      map_plot.plot([lidar_pose[0], lidar_pose[0] + face * numpy.cos(lidar_pose[2])], \
                    [lidar_pose[1], lidar_pose[1] + face * numpy.sin(lidar_pose[2])], \
                     'r-', markersize=1)
      # And draw the contour of the robot
      adjusted_contour = [to_world_from_ref(p, mapping.robot_mean) \
                          for p in mapping.robot_params["contour"]]
      map_plot.plot([p[0] for p in adjusted_contour] + [adjusted_contour[0][0]], \
                    [p[1] for p in adjusted_contour] + [adjusted_contour[0][1]], \
                     'r-', markersize=1)

    # show the features we have from the mapping map
    if True:
      for segment in mapping.map_segments:
        map_plot.plot([segment[0][0], segment[1][0]], \
                      [segment[0][1], segment[1][1]], 'b-')
      for corner in mapping.map_corners:
        map_plot.plot([corner[0]], \
                      [corner[1]], 'bo', markersize = 5)

    # show the lidar returns
    if True:
      map_plot.plot([p[0] for p in mapping.cartesian_points], \
                    [p[1] for p in mapping.cartesian_points], 'bo', markersize=1)

    # show the features that we have from extraction
    if True:
      for segment_association in mapping.segment_associations:
        segment = segment_association[0]
        map_plot.plot([segment[0][0], segment[1][0]], \
                      [segment[0][1], segment[1][1]], 'g-')
      for segment in mapping.new_segments:
        map_plot.plot([segment[0][0], segment[1][0]], \
                      [segment[0][1], segment[1][1]], 'r-')
      for corner_association in mapping.corner_associations:
        corner = corner_association[0]
        map_plot.plot([corner[0]], \
                      [corner[1]], 'go', markersize = 5)
      for corner in mapping.new_corners:
        map_plot.plot([corner[0]], \
                      [corner[1]], 'ro', markersize = 5)

  def plot_graphs(self, map_plot, graph_plot1, graph_plot2, graph_plot3, graph_plot4):
    if len(self.vals_actual_pose_x) == len(self.vals_actual_pose_y):
      map_plot.plot(self.vals_actual_pose_x, self.vals_actual_pose_y, 'k-')
    if len(self.vals_estimated_pose_x) == len(self.vals_estimated_pose_y):
      map_plot.plot(self.vals_estimated_pose_x, self.vals_estimated_pose_y, 'r-')

    legend1 = []
    if len(self.vals_frame_nums) == len(self.vals_pose_error):
      graph_plot1.plot(self.vals_frame_nums, self.vals_pose_error, 'k-')
      legend1.append("pose_error")
    if len(self.vals_frame_nums) == len(self.vals_pose_std_dev):
      graph_plot1.plot(self.vals_frame_nums, self.vals_pose_std_dev, 'r-')
      legend1.append("estimated_error")
    if len(self.vals_frame_nums) == len(self.vals_move_speed):
      graph_plot1.plot(self.vals_frame_nums, self.vals_move_speed)
      legend1.append("move_speed")
    graph_plot1.legend(legend1, loc='upper left')

    legend2 = []
    if len(self.vals_frame_nums) == len(self.vals_angle_error):
      graph_plot2.plot(self.vals_frame_nums, self.vals_angle_error, 'k-')
      legend2.append("angle_error")
    if len(self.vals_frame_nums) == len(self.vals_angle_std_dev):
      graph_plot2.plot(self.vals_frame_nums, self.vals_angle_std_dev, 'r-')
      legend2.append("estimated_error")
    if len(self.vals_frame_nums) == len(self.vals_turn_speed):
      graph_plot2.plot(self.vals_frame_nums, self.vals_turn_speed)
      legend2.append("turn_speed")
    graph_plot2.legend(legend2, loc='upper left')

    legend3 = []
    if len(self.vals_frame_nums) == len(self.vals_matched_segments):
      graph_plot3.plot(self.vals_frame_nums, self.vals_matched_segments)
      legend3.append("matched_segments")
    if len(self.vals_frame_nums) == len(self.vals_total_segments):
      graph_plot3.plot(self.vals_frame_nums, self.vals_total_segments)
      legend3.append("total_segments")
    if len(self.vals_frame_nums) == len(self.vals_matched_corners):
      graph_plot3.plot(self.vals_frame_nums, self.vals_matched_corners)
      legend3.append("matched_corners")
    if len(self.vals_frame_nums) == len(self.vals_total_corners):
      graph_plot3.plot(self.vals_frame_nums, self.vals_total_corners)
      legend3.append("total_corners")
    graph_plot3.legend(legend3, loc='upper left')

    legend4 = []
    if len(self.vals_frame_nums) == len(self.vals_localized):
      graph_plot4.plot(self.vals_frame_nums, self.vals_localized)
      legend4.append("localized")
    if len(self.vals_frame_nums) == len(self.vals_particles):
      graph_plot4.plot(self.vals_frame_nums, self.vals_particles)
      legend4.append("particles")
    if len(self.vals_frame_nums) == len(self.vals_max_score):
      graph_plot4.plot(self.vals_frame_nums, self.vals_max_score)
      legend4.append("max_score")
    if len(self.vals_frame_nums) == len(self.vals_good_particles):
      graph_plot4.plot(self.vals_frame_nums, self.vals_good_particles)
      legend4.append("good_particles")
    graph_plot4.legend(legend4, loc='upper left')

