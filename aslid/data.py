import os
import json

import numpy as np
import pandas as pd

import aslid.model as model
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm  # progress bar

import tensorflow.keras as keras

import math


def get_training_data_paths(train_spec, prediction_map_path, rootdir=".", limit=None):
    with open(prediction_map_path, "r") as j:
        prediction_map = json.load(j)

    training_data_paths = []
    training_data_Y = []
    with open(train_spec, "r") as f:
        lines = f.readlines()[1:]  # skip header
    for line in lines[:limit]:
        p, _, _, y = line.strip().split(",")
        training_data_paths.append(os.path.join(rootdir, p))
        training_data_Y.append(prediction_map[y])

    training_data_Y = np.array(training_data_Y)
    training_data_paths = np.array(training_data_paths)
    return training_data_paths, training_data_Y


def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / model.ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, model.ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def load_training_data(train_spec, prediction_map_path, limit=None):
    with open(prediction_map_path, "r") as j:
        prediction_map = json.load(j)

    training_data_paths = []
    Y = []
    with open(train_spec, "r") as f:
        lines = f.readlines()[1:]  # skip header
    for line in lines[:limit]:
        p, _, _, y = line.strip().split(",")
        training_data_paths.append(p)
        Y.append(prediction_map[y])

    # encode truth
    # Y = OneHotEncoder(sparse_output=False).fit_transform(np.array(Y)[:, np.newaxis])

    # load data from files
    X = []
    for path in tqdm(training_data_paths):
        # for path in training_data_paths:
        X.append(load_relevant_data_subset(os.path.join("data", path)))

    X = np.asarray(X, dtype=object)
    Y = np.asarray(Y)
    return X, Y


def make_uniform(X, nframes):
    """_summary_

    Args:
        X (np.array): Jagged array. N entries of (nframes, nlandmarks, coordinates)
        nframes (int): truncate/pad to nframes
    """
    Xt = np.zeros((len(X), nframes, X[0].shape[1], X[0].shape[2]))

    for i, x in enumerate(X):
        this_nframes = min(x.shape[0], nframes)
        pad = nframes - this_nframes
        Xt[i, pad:, :, :] = x[:this_nframes]

    return Xt


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X_paths, Y, batch_size, nframes, pipeline, shuffle=True):
        self.X_paths = X_paths
        self.Y = Y
        self.batch_size = batch_size
        self.nframes = nframes
        self.pipeline = pipeline
        self.indices = np.arange(len(self.X_paths))
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.X_paths) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of the batch size
        high = min(low + self.batch_size, len(self.X_paths))
        batch_indices = self.indices[low:high]

        batch_X_paths = self.X_paths[batch_indices]
        batch_Y = self.Y[batch_indices]

        batch_X = [load_relevant_data_subset(p) for p in batch_X_paths]
        batch_X = make_uniform(batch_X, self.nframes)
        batch_X = self.pipeline.fit_transform(batch_X)

        return batch_X, batch_Y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
