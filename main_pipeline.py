"""
Script to train an efficient classifier with Keras.
Load the images with Keras from folders.
"""

import tensorflow as tf
    
IMG_FOLDER_TRAIN = "data/train"
IMG_FOLDER_TEST = "data/test"
NUM_CLASSES = 2
# Load the images from the folders
train_datagen = tf.keras.utils.image_dataset_from_directory(
    IMG_FOLDER_TRAIN,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=2,
    image_size=(112, 112),
    shuffle=True,
    seed=42,
    interpolation="bilinear",
    follow_links=False,
)

val_datagen = tf.keras.utils.image_dataset_from_directory(
    IMG_FOLDER_TEST,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=2,
    image_size=(112, 112),
    shuffle=False,
    seed=42,
    interpolation="bilinear",
    follow_links=False,
)

# Create the model (efficientnet) with imagenet weights
model = tf.keras.applications.EfficientNetB0(
    input_shape=(112, 112, 3),
    include_top=False,
    weights="imagenet",
)

# Add a global spatial average pooling layer
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = tf.keras.layers.Dense(512, activation="relu")(x)
# Add a logistic layer
predictions = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

# Create the model
model = tf.keras.Model(inputs=model.input, outputs=predictions)

# ModelCheckpoint to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(train_datagen, epochs=10, validation_data=val_datagen, callbacks=[checkpoint])