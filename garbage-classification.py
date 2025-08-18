
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import pandas as pd
import os
from typing import Dict, List, Union

# %%
# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ## Data Setup Instructions
# 
# ### Option A: Run on Kaggle (Recommended - No Download Needed)
# 1. Upload this notebook to Kaggle
# 2. Add the "Garbage Classification" dataset to your Kaggle notebook:
#    - Click "Add data" → Search "Garbage Classification" → Add dataset by mostafaabla
# 3. Uncomment the Kaggle paths below:
#    - data_dir = '/kaggle/input/garbage-classification/garbage_classification'
#    - base_dir = "/kaggle/working"
# 4. Run the notebook - dataset is automatically available!
# 
# ### Option B: Run Locally (Requires Dataset Download)
# 1. Download dataset from: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
#    - You'll need a Kaggle account to download
#    - Dataset size: ~2-3 GB
# 2. Extract the downloaded ZIP file
# 3. Place the extracted 'garbage_classification' folder in your project directory
# 4. Keep the local paths below (already configured)
# 5. Install dependencies: pip install -r requirements.txt
# 
# ### Current Configuration:

# %% [markdown]
# ### Data Loading

# %%
# CONFIGURATION: Choose your environment
# 
# 🚀 KAGGLE (Recommended - No download needed):
# Uncomment these lines if running on Kaggle:
# data_dir = '/kaggle/input/garbage-classification/garbage_classification'
# base_dir = "/kaggle/working"

# 💻 LOCAL (Current configuration):
# Your dataset is located at: C:\wastenet-garbage-classifier\dataset
data_dir = 'dataset'  # Your dataset folder

# Alternative if you want to use full path:
# data_dir = r'C:\wastenet-garbage-classifier\dataset'

# Expected folder structure after download and extraction:
# your-project/
# ├── garbage_classification/
# │   ├── battery/
# │   ├── biological/
# │   ├── brown-glass/
# │   ├── cardboard/
# │   ├── clothes/
# │   ├── green-glass/
# │   ├── metal/
# │   ├── paper/
# │   ├── plastic/
# │   ├── shoes/
# │   ├── trash/
# │   └── white-glass/
# ├── garbage-classification.py
# └── requirements.txt
#   ├── trash/
#   └── white-glass/

# %%
# Exploration


def print_images_resolution(directory):
    unique_sizes = set()
    total_images = 0

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        image_files = os.listdir(subdir_path)
        num_images = len(image_files)
        print(f"{subdir}: {num_images}")
        total_images += num_images

        for img_file in image_files:
            img_path = os.path.join(subdir_path, img_file)
            with Image.open(img_path) as img:
                unique_sizes.add(img.size)

        for size in unique_sizes:
            print(f"- {size}")
        print("---------------")

    print(f"\nTotal: {total_images}")


# %%
print_images_resolution(data_dir)

# %%
classes = os.listdir(data_dir)
print(f"Classes: {classes}")

for class_name in classes:
    num_images = len(os.listdir(os.path.join(data_dir, class_name)))
    print(f"{class_name}: {num_images} images")


# %%
image_dir = Path(data_dir)

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + \
    list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

# %%
# Display 16 picture of the dataset with their labels
random_index = np.random.randint(0, len(image_df), 20)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
    ax.set_title(image_df.Label[random_index[i]])
plt.tight_layout()
plt.show()

# %%
# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
BATCH_SIZE = 64
EPOCHS = 20

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Split Dataset

# %%
# Working directory for train/val/test splits
# 
# 🚀 KAGGLE: Uncomment this line if running on Kaggle:
# base_dir = "/kaggle/working"

# 💻 LOCAL: Current configuration for local environment:
base_dir = "working"  # Creates 'working' folder in current directory

# The script will create these folders automatically:
# working/
# ├── train/
# ├── val/
# └── test/

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Divide for each class
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    images = np.array(os.listdir(class_path))

    # Split data: 80% train, 10% val, 10% test
    train_images, temp_images = train_test_split(
        images, test_size=0.2, random_state=42)
    val_images, test_images = train_test_split(
        temp_images, test_size=0.5, random_state=42)

    for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        split_class_dir = os.path.join(base_dir, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(split_class_dir, img)
            shutil.copy(src_path, dst_path)

print("Dataset successfully divided into train, validation and test set!")


# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Data Loaded Successfully!")


# %% [markdown]
# ## Modelling

# %%
base_model = MobileNetV2(input_shape=(
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), include_top=False, weights='imagenet')
base_model.trainable = False

for layer in base_model.layers[:20]:
    layer.trainable = True

# Sequential Model
with tf.device('/device:GPU:0'):
    model = Sequential([
        base_model,
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])


initial_learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# Config Callbaks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# %%
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# %% [markdown]
# ## Evaluasi dan Visualisasi

# %%
# Evaluate the model
evaluation = model.evaluate(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size
)

print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

# %%
# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# %%
# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Konversi Model

# %%
# Save model in SavedModel format for deployment
tf.saved_model.save(model, 'saved_model')
print("Model saved in SavedModel format to 'saved_model/' directory")

# %%
# TF Lite conversion

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with tf.io.gfile.GFile('tflite/model.tflite', 'wb') as f:
    f.write(tflite_model)


# %%
# Labels for TF Lite
LABEL_MAP = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}

with open("tflite/labels.txt", "w") as f:
    for i in range(len(LABEL_MAP)):
        f.write(f"{LABEL_MAP[i]}\n")


# %%


# %% [markdown]
# ## Local Inference (No Docker Required)

# %%
# Load the best model for local inference
loaded_model = tf.keras.models.load_model('best_model.keras')

# Label Constants
LABEL_MAP = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}


def load_and_preprocess_image(file_path: Union[str, Path], target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image for classification.

    Args:
        file_path: Path to the image file
        target_size: Tuple of (height, width) for resizing

    Returns:
        Preprocessed image tensor as numpy array
    """
    try:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Load image using PIL
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_tensor = np.expand_dims(image_array, axis=0)

        return image_tensor

    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise


def predict_image_local(image_path: str) -> Dict[str, Union[str, float, List[Dict[str, float]]]]:
    """
    Predict image class using the loaded model locally.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing prediction results and confidence scores for all classes
    """
    try:
        # Preprocess image
        image_tensor = load_and_preprocess_image(image_path)
        
        # Make prediction
        predictions = loaded_model.predict(image_tensor)[0]
        predicted_class_idx = np.argmax(predictions)
        
        # Create all predictions list
        all_predictions = [
            {
                'label': LABEL_MAP[idx],
                'confidence': float(score)
            }
            for idx, score in enumerate(predictions)
        ]
        
        # Sort predictions by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'confidence_scores': predictions.tolist(),
            'predicted_class': LABEL_MAP[predicted_class_idx],
            'confidence': float(predictions[predicted_class_idx]),
            'all_predictions': all_predictions
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise


# %%
# Test local inference
image_path = 'metal118.jpg'

try:
    result = predict_image_local(image_path)
    print(f"\nImage: {image_path}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop 5 predictions:")
    for i, pred in enumerate(result['all_predictions'][:5]):
        print(f"{i+1}. {pred['label']}: {pred['confidence']:.4f}")
except Exception as e:
    print(f"Process failed: {str(e)}")
