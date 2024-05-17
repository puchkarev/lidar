# lidar
Handling lidar data from rpilidar

To run a simulation simply execute

```python3 simulate.py```

The primary usage is in slam.py and example of usinge is in simulate.py

The basics are:
```
from lidar.slam import Slam

# Initialize the mapping with initial position (x, y, th_rad) and covariance matrix
# number of points is how many candidates we will evaluate in distribution per step
# segments provides an initial map as a list of ((x1, y1), (x2, y2)) for wall segments
mapping = Slam(initial_position = [0.0, 0.0, 0.0], \
               robot_covariance = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], \
               num_points = 50, segments = [])

# when the real robot moves we need to inform mapping about the change in position
# rotation and rotation error are specified in radians
# the distance error and rotation error are standard deviations
mapping.move_robot(move_distance = 1.0, rotate_angle = 2.0, distance_error = 0.1, rotation_error = 0.2)

# when we receive a new lidar scan of data we need to inform mapping of the new position
# note that the lidar data is expressed as a combinatin of (intensity, angle in degrees, and distance)
# note that you can continue feeding the same lidar data and hope localization gets better, this is
# possible since we do this based on sampling the distribution
mapping.lidar_update(lidar_data = [[1, 0.0, 10.0], [1, 90.0, 15.0], [1, 180.0, 20.0], [1, 270.0, 30.0]])

# to determine if mapping is localized you can check
mapping.localized

# to determine the localized position (x, y, angle in radians)
mapping.robot_mean

# to determine the covariance of position
mapping.robot_covariance

# to determine the current best understood map
mapping.map_segments.keys()
```

# Derivation
Measurement of positions
  cartesian_x = robot_x + sensor_distance * numpy.cos(sensor_angle + robot_angle)
  cartesian_y = robot_y + sensor_distance * numpy.sin(sensor_angle + robot_angle)

With errors becomes:
  x = robot_x + robot_x_err + (sensor_distance + sensor_distance_err) * \
                               numpy.cos(sensor_angle + sensor_angle_err + robot_angle + robot_angle_err)

  y = robot_y + robot_y_err + (sensor_distance + sensor_distance_err) * \
                               numpy.sin(sensor_angle + sensor_angle_err + robot_angle + robot_angle_err)

Taking the derivative with respect to errors and assuming that errors are small (values can be assumed 0):
  d_x = d_robot_x_err - \
        sensor_distance * numpy.sin(sensor_angle + robot_angle) * (d_sensor_angle_err + d_robot_angle_err) + \
        d_sensor_distance_err * numpy.cos(sensor_angle + robot_angle)

  d_y = d_robot_y_err + \
        sensor_distance * numpy.cos(sensor_angle + robot_angle) * (d_sensor_angle_err + d_robot_angle_err) + \
        d_sensor_distance_err * numpy.sin(sensor_angle + robot_angle)
