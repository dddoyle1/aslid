from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import BaseEstimator

from .map_landmark_connections import get_landmark_connections


def transformer_encoder(inputs, num_heads, ff_dim, dropout=0):
    # Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    # Attention
    x = layers.MultiHeadAttention(
        key_dim=x.shape[-1],
        num_heads=num_heads,
        dropout=dropout,
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    n_classes,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


class TransformerNetwork(BaseEstimator):
    def __init__(
        self,
        *args,
        initial_learning_rate=0.01,
        learning_rate_decay_steps=10000,
        learning_rate_decay_rate=0.9,
        batch_size=10,
        epochs=100,
        validation_split=0.2,
        callbacks=[],
        patience=10,
        **kwargs
    ):
        self.model = build_model(*args, **kwargs)

        callbacks.append(
            [
                keras.callbacks.EarlyStopping(
                    patience=patience, restore_best_weights=True
                )
            ]
        )
        self._fit_args = {
            "batch_size": batch_size,
            "epochs": epochs,
            "callbacks": callbacks,
        }

        self._compile_args = {
            "loss": "sparse_categorical_crossentropy",
            "optimizer": keras.optimizers.Adam(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=learning_rate_decay_steps,
                    decay_rate=learning_rate_decay_rate,
                )
            ),
            # "optimizer": keras.optimizers.Adam(learning_rate=learning_rate),
            "metrics": ["sparse_categorical_accuracy"],
        }

    def compile_args(self, **kwargs):
        self._compile_args.update(**kwargs)

    def fit_args(self, **kwargs):
        self._fit_args.update(**kwargs)

    def fit(self, training_generator, validation_generator):
        self.model.compile(**self._compile_args)
        self.model.fit(
            training_generator, validation_data=validation_generator, **self._fit_args
        )
