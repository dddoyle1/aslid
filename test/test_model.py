import sys

sys.path.append("../")

import aslid.model as model
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array

    def inverse_transform(self, Y):
        return Y


def get_test_landmark_data(N=10, nframes=11, nlandmarks=4, ncoords=3):
    return np.random.uniform(0, 1, (N, nframes, nlandmarks, ncoords))


def test_LHReflection():
    data = get_test_landmark_data()
    transformer = model.LHReflectionTransformer()
    reflected = transformer.fit_transform(data)
    reflected_inv = transformer.inverse_transform(reflected)

    assert (data == reflected_inv).all()


def test_RHReflection():
    data = get_test_landmark_data(ncoords=2)
    transformer = model.RHReflectionTransformer()
    reflected = transformer.fit_transform(data)
    reflected_inv = transformer.inverse_transform(reflected)

    assert (data == reflected_inv).all()


def test_FlattenLandmarks():
    data = get_test_landmark_data()
    flatten = model.FlattenLandmarksTransformer()
    flattened = flatten.fit_transform(data)
    flatten_inv = flatten.inverse_transform(flattened)

    assert (data == flatten_inv).all() and (
        data[0, 0, :].flatten() == flattened[0, 0]
    ).all()


def test_RHPipeline():
    nlandmarks = model.ROWS_PER_FRAME
    ncoords = 2
    nframes = 10
    N = 3
    data = get_test_landmark_data(
        N=N, nframes=nframes, nlandmarks=nlandmarks, ncoords=ncoords
    )

    pipeline = model.RHPipeline()
    pipeline.steps.append(("passthrough", IdentityTransformer()))

    xform = pipeline.fit_transform(data)
    assert xform.shape == (N, (ncoords + 1) * model.ROWS_PER_HAND, nframes)


def test_LHPipeline():
    nlandmarks = model.ROWS_PER_FRAME
    ncoords = 2
    nframes = 10
    N = 3
    data = get_test_landmark_data(
        N=N, nframes=nframes, nlandmarks=nlandmarks, ncoords=ncoords
    )
    pipeline = model.LHPipeline()
    pipeline.steps.append(("passthrough", IdentityTransformer()))

    xform = pipeline.fit_transform(data)
    assert xform.shape == (N, (ncoords + 1) * model.ROWS_PER_HAND, nframes)


def test_CombineHandsPipeline():
    nlandmarks = model.ROWS_PER_FRAME
    ncoords = 2
    nframes = 10
    N = 1
    data = get_test_landmark_data(
        N=N, nframes=nframes, nlandmarks=nlandmarks, ncoords=ncoords
    )

    pipeline = model.MergeHands()
    xform = pipeline.fit_transform(data)

    assert xform.shape == (N, model.FEATURES, nframes)


def test_TransformerPipeline():
    nlandmarks = model.ROWS_PER_FRAME
    ncoords = 3
    nframes = 10
    N = 1
    data = get_test_landmark_data(
        N=N, nframes=nframes, nlandmarks=nlandmarks, ncoords=ncoords
    )
    pipeline = model.PreprocessingPipeline()
    xform = pipeline.fit_transform(data)

    assert xform.shape == (N, nframes, model.FEATURES)
