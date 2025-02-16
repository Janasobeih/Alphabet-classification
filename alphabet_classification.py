
import os
import string
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Define the local dataset path
dataset_path = r"C:\Users\Mostafa\A_Z Handwritten Data.csv"

# Check if the dataset exists at the specified path
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The dataset was not found at the path: {dataset_path}")

# Load the dataset
try:
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except PermissionError:
    raise PermissionError(f"Permission denied when trying to read the file: {dataset_path}")
except Exception as e:
    raise Exception(f"An error occurred while reading the file: {e}")

# Separate features and labels
label_column = '0'  # Change if your label column has a different name
if label_column not in data.columns:
    raise ValueError(f"Label column '{label_column}' not found in the dataset.")

X = data.drop(label_column, axis=1).values
y = data[label_column].values

# Data Exploration
unique_classes, counts = np.unique(y, return_counts=True)
class_distribution = pd.DataFrame({'Class': unique_classes, 'Count': counts})

print("\nNumber of unique classes:", len(unique_classes))
print("\nClass distribution:")
print(class_distribution)

# Map class numbers to corresponding alphabets (0 -> 'A', 1 -> 'B', ..., 25 -> 'Z')
class_distribution['Alphabet'] = class_distribution['Class'].apply(lambda x: chr(65 + int(x)))  # 65 is ASCII for 'A'

# Plot class distribution with alphabet labels
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Alphabet',
    y='Count',
    data=class_distribution,
    hue='Alphabet',          # Assigning 'Alphabet' to hue
    palette='viridis',
    dodge=False,             # Ensures bars are not separated
    legend=False             # Hides the redundant legend
)
plt.title('Class Distribution')
plt.xlabel('Alphabet')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.show(block=False)  # Non-blocking rendering
plt.pause(0.2)  # Pause to allow rendering
plt.close()

# Verify all alphabets are present
all_alphabets = set(string.ascii_uppercase)
present_alphabets = set(class_distribution['Alphabet'])
missing_alphabets = all_alphabets - present_alphabets

if missing_alphabets:
    print("\nMissing Alphabets in the Dataset:")
    print(', '.join(sorted(missing_alphabets)))
else:
    print("\nAll alphabets (A-Z) are present in the dataset.")

# Normalize the images
X_normalized = X / 255.0

# Reshape the flattened vectors to 28x28 images
X_reshaped = X_normalized.reshape(-1, 28, 28)

# --- Display 8 Samples for Each Alphabet and a Combined Image ---

# Define the number of samples per class
samples_per_class = 8

# Create a dictionary to hold samples
sample_dict = {}

for class_num in unique_classes:
    # Get all indices for the current class
    indices = np.where(y == class_num)[0]

    # Check if there are enough samples
    if len(indices) < samples_per_class:
        print(f"Not enough samples for class {class_num} ({chr(65 + int(class_num))}). Available: {len(indices)}")
        selected_indices = indices  # Select all available samples
    else:
        # Randomly select 8 unique indices without replacement
        selected_indices = random.sample(list(indices), samples_per_class)

    # Store the selected images in the dictionary
    sample_dict[class_num] = X_reshaped[selected_indices]

    # Plot the 8 samples for the current class
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    fig.suptitle(f"Samples for '{chr(65 + int(class_num))}'", fontsize=14)

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_indices):
            ax.imshow(X_reshaped[selected_indices[i]], cmap='gray')
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')
        else:
            ax.axis('off')  # Hide any unused subplots

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the title
    plt.show(block=False)  # Non-blocking rendering
    plt.pause(0.2)  # Pause to allow rendering
    plt.close()

# Create and display the combined image with one sample per alphabet
combined_samples = []
combined_alphabets = []

for class_num in unique_classes:
    # Select the first sample for the combined image
    first_index = np.where(y == class_num)[0][0]
    combined_samples.append(X_reshaped[first_index])
    combined_alphabets.append(chr(65 + int(class_num)))

# Define grid size for combined image
combined_rows, combined_cols = 4, 7  # 4 rows x 7 columns = 28 slots (more than needed for 26 alphabets)

plt.figure(figsize=(14, 8))
for i, (image, alphabet) in enumerate(zip(combined_samples, combined_alphabets)):
    plt.subplot(combined_rows, combined_cols, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(alphabet, fontsize=12)
    plt.axis('off')

# Hide any unused subplots
for j in range(len(combined_samples), combined_rows * combined_cols):
    plt.subplot(combined_rows, combined_cols, j + 1).axis('off')

plt.suptitle('One Sample for Each Alphabet (A-Z)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show(block=False)  # Non-blocking rendering
plt.pause(0.2)  # Pause to allow rendering
plt.close()

#!!!!!!!!!!!!!!!!!!!!!!
#1!!!!!!!!!!!!!!!!!

# !!!!!!!!!!!!!! KAFRAWY & AYMAN START HEREE !!!!!!!!!!!!!!!!!!!!!!!

team_names = ["youssef","youssef","ahmed","omar","mostafa","nourhan"]
letters_to_test = set(''.join(team_names).upper())  # Unique letters
# letters_to_test might have duplicates, set() removes duplicates.

# Convert labels (y) to categorical
num_classes = len(unique_classes)
y_categorical = to_categorical(y, num_classes=num_classes)

# Train/validation/test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

# Flatten the input for MLP
input_shape = (28, 28)
X_train_flat = X_train.reshape(-1, 28*28)
X_val_flat = X_val.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

# Model 1: Simpler MLP
model1 = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model 2: Deeper MLP
model2 = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model 1
checkpoint_m1 = ModelCheckpoint('best_model_1.h5', monitor='val_accuracy', save_best_only=True, verbose=0)
history1 = model1.fit(
    X_train_flat, y_train,
    validation_data=(X_val_flat, y_val),
    epochs=10, batch_size=128, verbose=1,
    callbacks=[checkpoint_m1]
)

# Train Model 2
checkpoint_m2 = ModelCheckpoint('best_model_2.h5', monitor='val_accuracy', save_best_only=True, verbose=0)
history2 = model2.fit(
    X_train_flat, y_train,
    validation_data=(X_val_flat, y_val),
    epochs=10, batch_size=128, verbose=1,
    callbacks=[checkpoint_m2]
)

#  training/validation accuracy and loss for both models
def plot_history(hist, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    fig.suptitle(f'{title_prefix} Model Performance', fontsize=14)

    # Plot accuracy
    axes[0].plot(hist.history['accuracy'], label='Train Accuracy')
    axes[0].plot(hist.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Plot loss
    axes[1].plot(hist.history['loss'], label='Train Loss')
    axes[1].plot(hist.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()

plot_history(history1, "Model 1")
plot_history(history2, "Model 2")

# bngeeb best model ->  on validation accuracy
best_model_path = ''
best_val_acc_model1 = max(history1.history['val_accuracy'])
best_val_acc_model2 = max(history2.history['val_accuracy'])

if best_val_acc_model1 > best_val_acc_model2:
    best_model_path = 'best_model_1.h5'
else:
    best_model_path = 'best_model_2.h5'

best_model = tf.keras.models.load_model(best_model_path)

# Evaluate on test data
test_loss, test_acc = best_model.evaluate(X_test_flat, y_test, verbose=0)
print(f"\nBest Model Test Accuracy: {test_acc:.4f}")

# Predictions for test data
y_pred_proba = best_model.predict(X_test_flat)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[chr(65+i) for i in range(num_classes)]))

average_f1 = f1_score(y_true, y_pred, average='macro')
print(f"\nAverage F1-Score (Macro): {average_f1:.4f}")

# Team member images


# Function to get random!!! "bonus" sample for a given letter
def get_random_sample_for_letter(letter, X_data, y_data):
    class_num = ord(letter.upper()) - 65
    indices = np.where(np.argmax(y_data, axis=1) == class_num)[0]
    if len(indices) == 0:
        return None, None
    chosen_idx = random.choice(indices)
    return X_data[chosen_idx], class_num


# Loop through each name and plot its corresponding letters
for name in team_names:
    letters = list(name.upper())  # Convert name to uppercase and split into letters
    fig, axes = plt.subplots(1, len(letters), figsize=(len(letters) * 3, 3))

    fig.suptitle(f"Predictions for '{name.capitalize()}'", fontsize=16)

    for i, letter in enumerate(letters):
        img, class_num = get_random_sample_for_letter(letter, X_test, y_test)
        if img is not None:
            # Predict
            img_flat = img.reshape(1, 28 * 28)
            pred_proba = best_model.predict(img_flat)
            pred_class = np.argmax(pred_proba)
            predicted_letter = chr(65 + pred_class)

            # Plot the letter
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"True: {letter}\nPred: {predicted_letter}")
            axes[i].axis('off')
        else:
            axes[i].axis('off')  # Hide unused axes

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()

test_letters = sorted(letters_to_test)  # Sort for consistent order
print(f"\nTesting letters from team members' names: {', '.join(test_letters)}")

fig, axes = plt.subplots(2, int(np.ceil(len(test_letters)/2)), figsize=(14,6))
axes = axes.flatten()

for i, letter in enumerate(test_letters):
    img, class_num = get_random_sample_for_letter(letter, X_test, y_test)
    if img is not None:
        # Predict
        img_flat = img.reshape(1, 28*28)
        pred_proba = best_model.predict(img_flat)
        pred_class = np.argmax(pred_proba)
        predicted_letter = chr(65 + pred_class)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {letter}\nPred: {predicted_letter}")
        axes[i].axis('off')
    else:
        axes[i].axis('off')

plt.suptitle("Predictions on Team Members' Name Letters")
plt.tight_layout()
plt.show(block=False)
plt.pause(3.5)
plt.close()

# The code above:
# - Keeps the original logic and code
# - Appends new code to implement two neural networks
# - Trains both and plots the training and validation accuracy/loss
# - Saves the best model, reloads it, and evaluates it on the test set
# - Produces confusion matrix, classification report, and average F1-score
# - Tests the best model on letters from the given team names
# References:
# - TensorFlow Keras documentation: https://www.tensorflow.org/guide/keras
