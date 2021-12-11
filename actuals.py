import modelbuilder
import os
import json

players_path = os.getcwd() + "/Fantasy-Premier-League/data/2021-22"

actual_points = modelbuilder.build_actual_scores(players_path)
actual_points_json = json.dumps(list(actual_points.values()))

print(actual_points_json)
