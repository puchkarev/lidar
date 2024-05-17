import math

def associate_features(new_features, map_features, scoring_function, threshold=1.0):
  """
  Associates detected features with known map features.

  Parameters:
  - new_features: List of new features detected.
  - map_features: List of known features in the map.
  - scoring_function: function that takes a new feature and a map feature and returns a score
  - threshold: if score is below the threshold then the association will be made.

  Returns:
  - associations: List of tuples (new_feature, map_feature).
  - unassociated_new: New features not associated with any map feature.
  - unassociated_map: Map features not associated with any new feature.
  """
  associations = []
  unassociated_new = list(new_features)
  unassociated_map = list(map_features)

  for new_feature in new_features:
    best_feature = None
    min_score = float('inf')
    for map_feature in map_features:
      score = scoring_function(new_feature, map_feature)
      if score < min_score:
        min_score = score
        best_feature = map_feature
    if min_score <= threshold:
      associations.append((new_feature, best_feature))
      if new_feature in unassociated_new:
        unassociated_new.remove(new_feature)
      if best_feature in unassociated_map:
        unassociated_map.remove(best_feature)

  return associations, unassociated_new, unassociated_map
