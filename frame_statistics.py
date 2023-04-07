import pandas as pd
import numpy as np
import os

import pyarrow.parquet as pq

stop_in = np.inf
i = 0
nframes = []
for root, dirs, training_files in os.walk("data/train_landmark_files"):
    for f in training_files:
        if i < stop_in:
            file_name = os.path.join(root, f)

            nframes.append(int(pq.read_metadata(file_name).num_rows / 543))
            # nframes.append(
            #    len(np.unique(pd.read_parquet(file_name, columns=["frame"]).values))
            # )

            i = i + 1

nframes = np.array(nframes)
print("nfiles", len(nframes))
print("min", nframes.min())
print("max", nframes.max())
print("mean", nframes.mean())

"""
    nfiles 94477
    min 2
    max 537
    mean 37.93502122209638
"""
