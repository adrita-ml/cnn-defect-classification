import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Filter to 2 classes (siulate defect vs no defect)
# Let's say:
# class 3 = defect
# class 5 = no defect

train_filter = np.where((train_labels == 3) | (train_labels == 5))
test_filter = np.where((test_labels == 3) | (test_labels == 5))

train_images, train_labels = train_images[train_filter[0]], train_labels[train_filter]
test_images, test_labels = test_images[test_filter[0]], test_labels[test_filter]

# Convert to binary (0 = no defect, 1 = defect)
train_labels = (train_labels == 3).astype(int)
test_labels = (test_labels == 3).astype(int)

# Normalize images
train_images = train_images/255.0
test_images = test_images/255.0

# Check shapes (important)
print(train_images.shape)
print(train_labels.shape)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu',
    input_shape = (32,32,3)),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    train_images, train_labels,
    epochs=5,
    validation_data = (test_images, test_labels)
)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)