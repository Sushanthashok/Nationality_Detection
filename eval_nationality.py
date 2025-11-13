# eval_nationality.py
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json

IMG_SIZE = 224
BATCH = 32
DATA_DIR = "data/nationality"

model = load_model("models/nationality_mobilenetv2.h5")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

val_gen = datagen.flow_from_directory(DATA_DIR, target_size=(IMG_SIZE,IMG_SIZE),
                                       batch_size=BATCH, class_mode='categorical',
                                       subset='validation', shuffle=False)

preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
labels = list(val_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Nationality Confusion Matrix")
plt.tight_layout()
plt.savefig("nationality_confusion_matrix.png", dpi=150)
print("Saved nationality_confusion_matrix.png")
# save labels order for app
with open("models/nationality_labels.json","w") as f:
    json.dump(labels, f)
print("Saved models/nationality_labels.json")
