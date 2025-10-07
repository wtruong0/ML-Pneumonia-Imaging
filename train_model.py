import keras
from keras import layers, utils, callbacks

train_dataset = utils.image_dataset_from_directory(
    'dataset/train', #change if you're not using a venv
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    class_names=['NORMAL', 'PNEUMONIA'],
    shuffle=True,
)

val_dataset = utils.image_dataset_from_directory(
    'dataset/val', #change if you're not using a venv
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

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    min_delta=0.001,
    restore_best_weights=True
)

model_save = callbacks.ModelCheckpoint(
    filepath='models/premier_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

historyRecords = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stop, model_save]
)