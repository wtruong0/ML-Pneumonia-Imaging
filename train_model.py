import keras
from keras import layers, utils, ops

train_TFdataset = utils.image_dataset_from_directory(
    'C:/! random/ML Proj/dataset/train',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    class_names=['NORMAL', 'PNEUMONIA'],
    shuffle=True,
)

model = keras.Sequential([
    layers.Normalization(axis=-1),
    layers.RandomZoom(0.1),
    layers.RandomRotation(0.1),
    layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])