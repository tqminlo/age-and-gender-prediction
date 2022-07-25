# Khai báo thư viện
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from keras import applications

# Setup file
train_file = "Data25k\Train"
val_file = "Data25k\Validation"

"""
224, 224, 64 -> 224, 224, 8, 8
"""

# Setup Pre Model
M = VGG16(input_shape=(224, 224, 3), include_top=False, weights=None)
M.summary()
model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model.summary()

# Cố định các lớp
for layer in model.layers:
    layer.trainable = False

# Setup Train Model
Act = np.ndarray(dtype=int, shape=100)
for i in range(100):
    Act[i] = i+1
print(Act)
initializer = tf.keras.initializers.Constant(Act)
print(initializer)
layer_chose = model.get_layer(name='block5_pool')
output_layer_chose = layer_chose.output
x = layers.Flatten()(output_layer_chose)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Dense(units=100, activation='softmax')(x)
x = layers.Dense(units = 1, kernel_initializer=initializer, use_bias=False, activation= 'relu')(x)
model2 = Model(model.input, x)
model2.layers[-1].trainable = False
model2.summary()

# Training Model
train_processdata_func = ImageDataGenerator(rescale=1./255)
train_data = train_processdata_func.flow_from_directory(directory=train_file, batch_size=20, target_size=(224, 224), class_mode='sparse')
val_data = train_processdata_func.flow_from_directory(directory=val_file, batch_size=20, target_size=(224, 224),class_mode='sparse')

model2.compile(optimizer=RMSprop(learning_rate=0.0001), loss='mean_absolute_error', metrics=['acc'])

# Train
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=100),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.4, patience=30)
    ]
train = model2.fit_generator(generator=train_data, steps_per_epoch=50, epochs=500,validation_data=val_data, validation_steps=19, verbose=1, callbacks=my_callbacks)
# Lưu weights
model2.save_weights('weights_DEX_Data25k_(callbacks).h5')

# In đồ thị Acc, Loss
acc = train.history['acc']
val_acc = train.history['val_acc']
loss = train.history['loss']
val_loss = train.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, color='blue', label='Training Accuracy')
plt.plot(epochs, val_acc, color='orange', label='Validation Accuracy')
plt.title('Training VS Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, color='blue', label='Training Loss')
plt.plot(epochs, val_loss, color='orange', label='Validation Loss')
plt.title('Training VS Validation Loss')
plt.legend()
plt.show()
