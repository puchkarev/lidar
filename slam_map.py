import numpy as np

def update_map(line_associations, corner_associations, unassociated_lines, unassociated_corners, map_lines, map_corners, max_unobserved_time=5):
  """
  Update the map with associated and new features.

  Args:
  - line_associations (list): List of tuples (new_line, associated_map_line).
  - corner_associations (list): List of tuples (new_corner, associated_map_corner).
  - unassociated_lines (list): List of new lines not associated with the map.
  - unassociated_corners (list): List of new corners not associated with the map.
  - map_lines (dict): Existing lines in the map with their observation details.
  - map_corners (dict): Existing corners in the map with their observation details.
  - max_unobserved_time (int): The maximum cycles a feature can remain unobserved before being pruned.

  Returns:
  - Updated map_lines and map_corners.
  """
  # Update lines based on associations
  for new_line, map_line in line_associations:
      # Simple average update
      map_lines[map_line]['coords'] = ((new_line[0] + map_line[0]) / 2, (new_line[1] + map_line[1]) / 2)
      # Reset the observation timer
      map_lines[map_line]['last_observed'] = 0

  # Add new unassociated lines
  for line in unassociated_lines:
    map_lines[line] = {'coords': line, 'last_observed': 0}

  # Update corners based on associations
  for new_corner, map_corner in corner_associations:
    # Simple average update
    map_corners[map_corner]['coords'] = ((new_corner[0] + map_corner[0]) / 2, (new_corner[1] + map_corner[1]) / 2)
    # Reset the observation timer
    map_corners[map_corner]['last_observed'] = 0

  # Add new unassociated corners
  for corner in unassociated_corners:
    map_corners[corner] = {'coords': corner, 'last_observed': 0}

  # Increment observation timer and prune old features
  for feature in list(map_lines.keys()):
    map_lines[feature]['last_observed'] += 1
    if map_lines[feature]['last_observed'] > max_unobserved_time:
      del map_lines[feature]

  for feature in list(map_corners.keys()):
    map_corners[feature]['last_observed'] += 1
    if map_corners[feature]['last_observed'] > max_unobserved_time:
      del map_corners[feature]

  return map_lines, map_corners

if __name__ == '__main__':
  # Update Map
  # Example usage:
  # Assuming map_lines and map_corners are dictionaries keyed by feature with properties 'coords' and 'last_observed'
  map_lines = {}
  map_corners = {}
  line_associations = []
  corner_associations = []
  unassociated_lines = [((1, 1), (2, 2))]
  unassociated_corners = [((3, 3), (4, 4))]
  updated_map_lines, updated_map_corners = update_map(line_associations, corner_associations, unassociated_lines, unassociated_corners, map_lines, map_corners)
  print("Updated Map Lines:", updated_map_lines)
  print("Updated Map Corners:", updated_map_corners)

