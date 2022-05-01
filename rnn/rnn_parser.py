import pandas as pd
import numpy as np

# Takes in a dataframe, returns the parsed data as a list
def parse_rnn_dataframe(data_frame):
  data_list = []
  for _, sample in data_frame.iterrows():
    # Skip unrecognized sample
    if sample["recognized"] == False:
      continue

    class_name = sample["word"]
    drawing = sample["drawing"]
    points_per_stroke = [len(stroke[0]) for stroke in drawing] # points per stroke
    total_points = sum(points_per_stroke) # total points

    # (total_points, [x, y, ])
    sample_drawing_arr = np.zeros((total_points, 3), dtype=np.float32)
    current_point_index = 0

    # merge all points from all strokes, size [number of points, 3] where 3 consists of
    # [x, y, (0 = stroke not ended, 1 = stroke ended)]
    for stroke in drawing: # [[x], [y]]
      stroke_points = len(stroke[0])
      for i in range(len(stroke)): # 0, 1
        sample_drawing_arr[current_point_index:(current_point_index + stroke_points), i] = stroke[i]
      current_point_index += stroke_points
      sample_drawing_arr[current_point_index - 1, 2] = 1  # stroke_end

    # Compute deltas
    sample_drawing_arr[1:, 0:2] -= sample_drawing_arr[:-1, 0:2]
    sample_drawing_arr = sample_drawing_arr[1:, :]
    data_list.append([sample_drawing_arr, class_name])
  
  return data_list