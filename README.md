# lidar
Handling lidar data from rpilidar

To run a simulation simply execute

```python3 simulate.py```



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
