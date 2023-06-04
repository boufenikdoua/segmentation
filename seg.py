import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
    concatenate,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model


def unet(input_shape=(128, 128, 128, 1), filters=64):
    inputs = Input(input_shape)
    # Downsample path
    conv1 = Conv3D(filters, (3, 3, 3), activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(filters, (3, 3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(filters * 2, (3, 3, 3), activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(filters * 2, (3, 3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(filters * 4, (3, 3, 3), activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(filters * 4, (3, 3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(filters * 8, (3, 3, 3), activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(filters * 8, (3, 3, 3), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    # Upsample path
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(filters * 4, (3, 3, 3), activation="relu", padding="same")(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(filters * 4, (3, 3, 3), activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(filters * 2, (3, 3, 3), activation="relu", padding="same")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(filters * 2, (3, 3, 3), activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(filters, (3, 3, 3), activation="relu", padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(filters, (3, 3, 3), activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    outputs = Conv3D(1, (1, 1, 1), activation="sigmoid")(conv7)

    model = Model(inputs, outputs)
    return model


def load_files(data_path, mask_path):
    data_files = glob.glob(os.path.join(data_path, "*.dcm"))
    mask_files = glob.glob(os.path.join(mask_path, "*.png"))
    return data_files, mask_files


def load_data(data_files, mask_files):
    data = []
    masks = []
    for data_file in data_files:
        # Load data
        data_array = pydicom.dcmread(data_file).pixel_array
        data_array = cv2.resize(data_array, (128, 128, 128))
        data.append(data_array)

        # Get corresponding mask file path
        file_name = os.path.basename(data_file)
        mask_file = os.path.join(mask_files, file_name.replace(".dcm", ".png"))
        mask_array = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask_array = cv2.resize(mask_array, (128, 128, 128))
        masks.append(mask_array)

    data = np.array(data)
    masks = np.array(masks)
    return data, masks


# Load CT training data
train_ct_data_path = "CHAOS/Train_Sets/CT/DICOM_anon"
train_ct_mask_path = "CHAOS/Train_Sets/CT/Ground"
train_ct_data_files, train_ct_mask_files = load_files(
    train_ct_data_path, train_ct_mask_path
)

# Load MRI training data
train_mri_data_path = "CHAOS/Train_Sets/MRI"
train_mri_t1dual_data_path = os.path.join(
    train_mri_data_path, "T1DUAL/DICOM_anon/InPhase"
)
train_mri_t1dual_mask_path = os.path.join(train_mri_data_path, "T1DUAL/Ground")
train_mri_t2spir_data_path = os.path.join(train_mri_data_path, "T2SPIR/DICOM_anon")
train_mri_t2spir_mask_path = os.path.join(train_mri_data_path, "T2SPIR/Ground")
train_mri_t1dual_data_files, train_mri_t1dual_mask_files = load_files(
    train_mri_t1dual_data_path, train_mri_t1dual_mask_path
)
train_mri_t2spir_data_files, train_mri_t2spir_mask_files = load_files(
    train_mri_t2spir_data_path, train_mri_t2spir_mask_path
)

# Load CT test data
test_ct_data_path = "CHAOS/Test_Sets/CT"
test_ct_data_files, _ = load_files(test_ct_data_path, "")

# Load MRI test data
test_mri_data_path = "CHAOS/Test_Sets/MRI"
test_mri_t1dual_data_path = os.path.join(
    test_mri_data_path, "T1DUAL/DICOM_anon/InPhase"
)
test_mri_t2spir_data_path = os.path.join(test_mri_data_path, "T2SPIR/DICOM_anon")
test_mri_t1dual_data_files, _ = load_files(test_mri_t1dual_data_path, "")
test_mri_t2spir_data_files, _ = load_files(test_mri_t2spir_data_path, "")

# Load CT training data
train_ct_data, train_ct_masks = load_data(train_ct_data_files, train_ct_mask_files)

# Load MRI training data
train_mri_t1dual_data, train_mri_t1dual_masks = load_data(
    train_mri_t1dual_data_files, train_mri_t1dual_mask_files
)
train_mri_t2spir_data, train_mri_t2spir_masks = load_data(
    train_mri_t2spir_data_files, train_mri_t2spir_mask_files
)

# Load CT test data
test_ct_data, _ = load_data(test_ct_data_files, [])

# Load MRI test data
test_mri_t1dual_data, _ = load_data(test_mri_t1dual_data_files, [])
test_mri_t2spir_data, _ = load_data(test_mri_t2spir_data_files, [])


def preprocess_data(data, masks):
    # Check if arrays have non-zero size
    if data.size == 0 or masks.size == 0:
        raise ValueError("Input arrays have zero size.")
    # Normalize images
    data = data / np.max(data)

    # Normalize masks
    masks = masks / np.max(masks)

    # Add channel dimension to images and masks
    data = np.expand_dims(data, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    return data, masks


# Define paths
train_data_path = "CHAOS/Train_Sets"
train_ct_data_path = os.path.join(train_data_path, "CT", "DICOM_anon")
train_mri_t1dual_data_path = os.path.join(
    train_data_path, "MRI", "T1DUAL", "DICOM_anon", "InPhase"
)
train_mri_t2spir_data_path = os.path.join(
    train_data_path, "MRI", "T2SPIR", "DICOM_anon"
)
train_ct_mask_path = os.path.join(train_data_path, "CT", "Ground")
train_mri_t1dual_mask_path = os.path.join(train_data_path, "MRI", "T1DUAL", "Ground")
train_mri_t2spir_mask_path = os.path.join(train_data_path, "MRI", "T2SPIR", "Ground")

# Load file paths
train_ct_data_files, train_ct_mask_files = load_files(
    train_ct_data_path, train_ct_mask_path
)
train_mri_t1dual_data_files, train_mri_t1dual_mask_files = load_files(
    train_mri_t1dual_data_path, train_mri_t1dual_mask_path
)
train_mri_t2spir_data_files, train_mri_t2spir_mask_files = load_files(
    train_mri_t2spir_data_path, train_mri_t2spir_mask_path
)

# Load and preprocess data
train_ct_data, train_ct_masks = load_data(train_ct_data_files, train_ct_mask_files)
train_mri_t1dual_data, train_mri_t1dual_masks = load_data(
    train_mri_t1dual_data_files, train_mri_t1dual_mask_files
)
train_mri_t2spir_data, train_mri_t2spir_masks = load_data(
    train_mri_t2spir_data_files, train_mri_t2spir_mask_files
)

train_ct_data, train_ct_masks = preprocess_data(train_ct_data, train_ct_masks)
train_mri_t1dual_data, train_mri_t1dual_masks = preprocess_data(
    train_mri_t1dual_data, train_mri_t1dual_masks
)
train_mri_t2spir_data, train_mri_t2spir_masks = preprocess_data(
    train_mri_t2spir_data, train_mri_t2spir_masks
)

# Combine all data and masks
data = np.concatenate(
    [train_ct_data, train_mri_t1dual_data, train_mri_t2spir_data], axis=0
)
masks = np.concatenate(
    [train_ct_masks, train_mri_t1dual_masks, train_mri_t2spir_masks], axis=0
)

# Define input shape
input_shape = data[0].shape

# Create the model
model = unet(input_shape=input_shape, filters=64)

# Compile the model
model.compile(
    optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

# Define callbacks
checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=5)

# Train the model
history = model.fit(
    data,
    masks,
    validation_split=0.2,
    batch_size=1,
    epochs=10,
    callbacks=[checkpoint, early_stopping],
)

# Plot the training history
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Load the best model
model = load_model("model.h5")

# Make predictions on test data
test_data = data[-1:]  # Assuming the last data sample is for testing
predictions = model.predict(test_data)

# You can use the 'predictions' variable to analyze or visualize the predicted masks.
