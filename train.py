import aslid.model as model
import aslid.data as data

from sklearn.model_selection import train_test_split

from aslid.transformer_model import TransformerNetwork

X_paths, Y = data.get_training_data_paths(
    "data/train.csv",
    "data/sign_to_prediction_index_map.json",
    rootdir="data",
    limit=10000,
)

# X, Y = data.load_training_data(
#     "data/train.csv", "data/sign_to_prediction_index_map.json", limit=10000
# )
# X = data.make_uniform(X, model.FRAMES)


# X = pipeline.fit_transform(X)

X_paths_train, X_paths_test, Y_train, Y_test = train_test_split(
    X_paths, Y, train_size=0.5, random_state=1
)

batch_size = 32
pipeline = model.PreprocessingPipeline()
training_data_generator = data.DataGenerator(
    X_paths_train,
    Y_train,
    batch_size=batch_size,
    nframes=model.FRAMES,
    pipeline=pipeline,
)
validation_data_generator = data.DataGenerator(
    X_paths_test, Y_test, batch_size=batch_size, nframes=model.FRAMES, pipeline=pipeline
)

epochs = 1000

input_shape = (model.FRAMES, model.FEATURES)
num_heads = model.FEATURES
ff_dim = model.FEATURES
classifier = TransformerNetwork(
    input_shape=input_shape,
    n_classes=250,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
    initial_learning_rate=1e-3,
    patience=40,
    batch_size=batch_size,
    epochs=epochs,
)

classifier.fit(training_data_generator, validation_data_generator)
