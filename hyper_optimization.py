import argparse

parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("project_name")
parser.add_argument("--overwrite", action="store_true", default=False)

args = parser.parse_args()

import numpy as np

import aslid.model
import aslid.data
from aslid.transformer_model import build_model

from sklearn.model_selection import train_test_split

from aslid.transformer_model import TransformerNetwork

import keras_tuner
from tensorflow import keras


X_paths, Y = aslid.data.get_training_data_paths(
    "data/train.csv",
    "data/sign_to_prediction_index_map.json",
    rootdir="data",
    limit=1000,
)

X_paths_train, X_paths_test, Y_train, Y_test = train_test_split(
    X_paths, Y, train_size=0.5, random_state=1
)

batch_size = 64
epochs = 20
pipeline = aslid.model.PreprocessingPipeline()
training_data_generator = aslid.data.DataGenerator(
    X_paths_train,
    Y_train,
    batch_size=batch_size,
    nframes=aslid.model.FRAMES,
    pipeline=pipeline,
)

validation_data_generator = aslid.data.DataGenerator(
    X_paths_test,
    Y_test,
    batch_size=batch_size,
    nframes=aslid.model.FRAMES,
    pipeline=pipeline,
)


def hypermodel(hp):
    # ff_dim = hp.Int(
    #     "ff_dim",
    #     min_value=aslid.model.FEATURES / 2,
    #     max_value=aslid.model.FEATURES * 2,
    #     step=63,
    # )  # 3 steps
    ff_dim = hp.Fixed("ff_dim", 126)
    # num_transformer_blocks = hp.Int(
    #     "num_transformer_blocks", min_value=1, max_value=10, step=3
    # )  # 3 steps
    num_transformer_blocks = hp.Fixed("num_transformer_blocks", 4)

    num_transformer_heads = hp.Choice(
        "num_transformer_heads",
        values=[
            int(aslid.model.FEATURES / 2),
            int(aslid.model.FEATURES),
            int(aslid.model.FEATURES * 2),
        ],
    )
    # mlp_units = hp.Int(
    #     "mlp_units", min_value=64, max_value=256, step=2, sampling="log"
    # )

    mlp_units = hp.Fixed("mlp_units", 256)
    mlp_dropout = hp.Fixed("mlp_dropout", 0.25)
    dropout = hp.Fixed("dropout", 0.25)

    model = build_model(
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_transformer_heads=num_transformer_heads,
        mlp_units=mlp_units,
        mlp_dropout=mlp_dropout,
        dropout=dropout,
    )

    # learning_rate = hp.Float(
    #     "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    # )

    learning_rate = hp.Fixed("learning_rate", 1e-3)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics="sparse_categorical_accuracy",
    )

    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=hypermodel,
    objective="val_sparse_categorical_accuracy",
    max_trials=3,
    executions_per_trial=3,
    directory=args.directory,
    project_name=args.project_name,
    overwrite=args.overwrite,
)

tuner.search(
    training_data_generator, validation_data=validation_data_generator, epochs=epochs
)

tuner.results_summary()
