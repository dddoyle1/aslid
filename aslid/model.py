import pandas as pd
import numpy as np
import os
import json

from sklearn.preprocessing import FunctionTransformer

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion

import aslid.transformer_model as transformer_model


ROWS_PER_FRAME = 543  # number of landmarks per frame
ROWS_PER_FACE = 468  # number of face landmarks
ROWS_PER_LEFT_HAND = 21  # number of left hand landmarks
ROWS_PER_POSE = 33  # number of pose landmarks
ROWS_PER_RIGHT_HAND = 21  # number of right hand landmarks
ROWS_PER_HAND = 21

ROWS_OFFSET_RIGHT = ROWS_PER_FACE + ROWS_PER_LEFT_HAND + ROWS_PER_POSE
ROWS_OFFSET_LEFT = ROWS_PER_FACE

FRAMES = 30
FEATURES = ROWS_PER_HAND * 2 * 3  # 2 hands * 3 coordinates


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array

    def inverse_transform(self, Y):
        return Y


class DeserializeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rows_per_frame):
        super().__init__()
        self.rows_per_frame = rows_per_frame

    def transform(self, X):
        nframes = int(X.shape[0] / self.rows_per_frame)
        return X.reshape((nframes, self.rows_per_frame, X.shape[-1]))

    def fit(self, x, y=None):
        return self

    def inverse_transform(self, Y):
        pass


class ExtractHandsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset, rows_per_frame):
        super().__init__()
        self.offset = offset
        self.rows_per_frame = rows_per_frame

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """

        Args:
            X (np.array): (N, nframes, ROWS_PER_FRAME, 3)

        Returns:
            np.array: (N, nframes, ROWS_PER_HAND, 3)
        """
        return X[:, :, self.offset : self.offset + self.rows_per_frame, :]


class FlattenLandmarksTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.nlandmarks = None
        self.ncoordinates = None

    def fit(self, X, Y=None):
        self.nlandmarks = X.shape[2]
        self.ncoordinates = X.shape[3]
        return self

    def transform(self, X):
        """Flatten array of landmark data

        Args:
            X (np.array): (N, nframes, ROWS_PER_HAND, ncoordinates)

        Returns:
           np.array: (N, nframe, ROWS_PER_HAND * ncoordinates)
        """
        return X.reshape((X.shape[0], X.shape[1], X.shape[2] * X.shape[3]))

    def inverse_transform(self, Y):
        return Y.reshape((Y.shape[0], Y.shape[1], self.nlandmarks, self.ncoordinates))


class FillNA(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.nan_to_num(X, copy=False, nan=self.fill_value)


def LHReflection(X):
    """
    Args:
        X (np.array): (N, nframes, ROWS_PER_HAND, 3)

    Returns:
        nparray: (N, nframes, ROWS_PER_HAND, 4)
    """
    return np.concatenate(
        (X[:, :, :, :], (1 - X[:, :, :, 0])[:, :, :, np.newaxis]), axis=3
    )


def LHReflection_inv(Y):
    return Y[:, :, :, :-1]


def LHReflectionTransformer():
    return FunctionTransformer(
        LHReflection, inverse_func=LHReflection_inv, check_inverse=False
    )


def RHReflection(X):
    """
    Args:
        X (np.array): (N, nframes, ROWS_PER_HAND, 3)

    Returns:
        np.array: (N, nframes, ROWS_PER_HAND, 4)
    """
    return np.concatenate(
        ((1 - X[:, :, :, 0])[:, :, :, np.newaxis], X[:, :, :, [1, 0]]),
        axis=3,
    )


def RHReflection_inv(Y):
    return Y[:, :, :, [2, 1]]


def RHReflectionTransformer():
    return FunctionTransformer(
        RHReflection, inverse_func=RHReflection_inv, check_inverse=False
    )


def DropZ(X):
    """

    Args:
        X (np.array): (N, nframes, ROWS_PER_HAND, 3)

    Returns:
        np.array (N, nframes, ROWS_PER_HAND, 2)
    """
    return X[:, :, :, [0, 1]]


def DropZTransformer():
    return FunctionTransformer(DropZ)


def Transpose(X):
    """
    Args:
        X (np.array): (N, nframes, nfeatures)

    Returns:
        nparray: (N, nfeatures, nframes)
    """
    return np.transpose(X, axes=[0, 2, 1])


def Transpose_inv(Y):
    return Transpose(Y)


def TransposeTransformer():
    return FunctionTransformer(
        Transpose, inverse_func=Transpose_inv, check_inverse=False
    )


def RHPipeline():
    return Pipeline(
        [
            (
                "rh_extract",
                ExtractHandsTransformer(
                    offset=ROWS_OFFSET_RIGHT, rows_per_frame=ROWS_PER_HAND
                ),
            ),
            ("rh_dropz", DropZTransformer()),
            ("rh_reflection", RHReflectionTransformer()),
            ("rh_flatten", FlattenLandmarksTransformer()),
            ("rh_transpose", TransposeTransformer()),
        ]
    )


def LHPipeline():
    return Pipeline(
        [
            (
                "lh_extract",
                ExtractHandsTransformer(
                    offset=ROWS_OFFSET_LEFT, rows_per_frame=ROWS_PER_HAND
                ),
            ),
            ("lh_dropz", DropZTransformer()),
            ("lh_reflection", LHReflectionTransformer()),
            ("lh_flatten", FlattenLandmarksTransformer()),
            ("lh_transpose", TransposeTransformer()),
        ]
    )


def MergeHands():
    return FeatureUnion([("lh", LHPipeline()), ("rh", RHPipeline())])


def PreprocessingPipeline(**kwargs):
    return Pipeline(
        [
            ("merge", MergeHands()),
            ("transpose", TransposeTransformer()),
            ("impute", FillNA(fill_value=0)),
            # ("padd", PaddingTransformer(pad_length=30, fill_value=0)),
            # ("truncate", TruncationTransformer(lower=30)),
            ("passthrough", IdentityTransformer()),
        ],
        **kwargs
    )


def TransformerModel(*args, preprocessing_args=None, **kwargs):
    pipeline = PreprocessingPipeline(preprocessing_args)
    pipeline.steps.append(
        ("transformer", transformer_model.TransformerNetwork(*args, **kwargs))
    )

    return pipeline
