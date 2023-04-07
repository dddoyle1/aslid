import aslid.model
import aslid.data

import pandas as pd

import numpy as np


def get_landmark_connections():
    nlandmarks = aslid.model.ROWS_PER_FRAME
    ncoords = 3
    nframes = 10
    N = 1
    data = np.zeros((N, nframes, nlandmarks, ncoords))

    rh_landmarks = slice(
        aslid.model.ROWS_OFFSET_RIGHT,
        aslid.model.ROWS_OFFSET_RIGHT + aslid.model.ROWS_PER_HAND,
    )
    lh_landmarks = slice(
        aslid.model.ROWS_OFFSET_LEFT,
        aslid.model.ROWS_OFFSET_LEFT + aslid.model.ROWS_PER_HAND,
    )
    rh_val = 3
    lh_val = 5

    # frame * 10000 + feature * 100 + hand * 10 + coord + 1
    frames = np.linspace(0, nframes - 1, nframes, dtype=int)
    features = np.linspace(
        0, aslid.model.ROWS_PER_HAND - 1, aslid.model.ROWS_PER_HAND, dtype=int
    )
    coords = np.linspace(0, 2, 3, dtype=int)

    data[:, :, rh_landmarks, :] = rh_val * 10
    data[:, :, lh_landmarks, :] = lh_val * 10

    for frame in frames:
        data[:, frame, lh_landmarks, :] = (
            data[:, frame, lh_landmarks, :] + frame * 10000
        )
        data[:, frame, rh_landmarks, :] = (
            data[:, frame, rh_landmarks, :] + frame * 10000
        )

    for feature in features:
        data[:, :, aslid.model.ROWS_OFFSET_RIGHT + feature, :] = (
            data[:, :, aslid.model.ROWS_OFFSET_RIGHT + feature, :] + feature * 100
        )
        data[:, :, aslid.model.ROWS_OFFSET_LEFT + feature, :] = (
            data[:, :, aslid.model.ROWS_OFFSET_LEFT + feature, :] + feature * 100
        )

    for coord in coords:
        data[:, :, rh_landmarks, coord] = data[:, :, rh_landmarks, coord] + coord + 1
        data[:, :, lh_landmarks, coord] = data[:, :, lh_landmarks, coord] + coord + 1

    df = pd.read_parquet(
        aslid.data.get_training_data_paths("data/train.csv", rootdir="data")[0]
    )

    pipeline = aslid.model.PreprocessingPipeline()

    # (N, feature, frame)
    X = pipeline.fit_transform(data)

    # get all rows with matching parity
    # pplus is non-reflected
    # pminus is reflected
    # pplus and pminus share y coordinate rows

    X0 = X[0]
    y_rows = np.multiply.reduce(abs(X0) % 10 == 2, axis=0)
    x_rows = np.multiply.reduce(abs(X0) % 10 == 1, axis=0)
    x_reflect_rows = np.multiply.reduce(X0 < 0, axis=0)

    rh_rows = np.multiply.reduce(((abs(X0) / 10) % 10).astype(int) == rh_val, axis=0)
    lh_rows = np.multiply.reduce(((abs(X0) / 10) % 10).astype(int) == lh_val, axis=0)

    pplus = (rh_rows & (x_rows | y_rows)) | (lh_rows & (x_rows | y_rows))
    pminus = (rh_rows & (x_reflect_rows | y_rows)) | (
        lh_rows & (x_reflect_rows | y_rows)
    )

    return np.argwhere(pplus == 1).flatten(), np.argwhere(pminus == 1).flatten()


if __name__ == "__main__":
    get_landmark_connections()
