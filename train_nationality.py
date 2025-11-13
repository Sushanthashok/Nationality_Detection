# train_nationality.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks

IMG_SIZE = 224
BATCH = 32
EPOCHS = 12   # reduce on CPU; increase if you have GPU
DATA_DIR = "data/nationality"   # contains Indian/, United States/, African/, Other/
OUTPUT = "models/nationality_mobilenetv2.h5"
NUM_CLASSES = 4

os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH, class_mode='categorical',
    subset='training', seed=42
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH, class_mode='categorical',
    subset='validation', seed=42
)

base = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet', pooling='avg')
for layer in base.layers:
    layer.trainable = False

x = base.output
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs=base.input, outputs=out)

model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

cbs = [
    callbacks.ModelCheckpoint(OUTPUT, monitor='val_accuracy', save_best_only=True, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

# TRAIN (no workers argument)
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cbs)

# Optional fine-tune (unfreeze last layers)
for layer in base.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=4, callbacks=cbs)

model.save(OUTPUT)
print("Saved nationality model at:", OUTPUT)
