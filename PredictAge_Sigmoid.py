# Khai báo thư viện
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from keras import applications

# Setup file
train_file = "Train"
val_file = "Validation"

# Setup Pre Model
model = applications.VGG16(input_shape=(200, 200, 3), include_top=False, weights="imagenet")
model.summary()

# Cố định các lớp
for layer in model.layers:
    layer.trainable = False

# Setup Train Model
    #Đoạn này để khởi tạo tham số lớp Dense cuối (hằng số và ko được train)
Act = np.ndarray(dtype=int, shape=1)
Act[0] = 100
print(Act)
initializer = tf.keras.initializers.Constant(Act)
print(initializer)
    # Đoạn này để thêm các lớp kết nối, có cả lớp Dense cuối với trọng số mặc định như đã nêu ở trên
layer_chose = model.get_layer(name='block5_pool')
output_layer_chose = layer_chose.output
x = layers.Flatten()(output_layer_chose)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Dense(units=100, activation='relu')(x)
x = layers.Dense(units=1, activation='sigmoid')(x)
x = layers.Dense(units=1, kernel_initializer=initializer, use_bias=False, activation= 'relu')(x)
model2 = Model(model.input, x)
model2.layers[-1].trainable = False
model2.summary()

# Training Model
train_processdata_func = ImageDataGenerator(rescale=1./255)
train_data = train_processdata_func.flow_from_directory(directory=train_file, batch_size=20, target_size=(200, 200), class_mode='sparse')
val_data = train_processdata_func.flow_from_directory(directory=val_file, batch_size=20, target_size=(200, 200),class_mode='sparse')
model2.compile(optimizer=RMSprop(learning_rate=0.0001), loss='mean_absolute_error', metrics=['acc'])
#model2.load_weights('weight_imagenet_25k_UseVal_LossAbs(120ep).h5')
    # Lệnh train
train = model2.fit_generator(generator=train_data, steps_per_epoch=50, epochs=200,validation_data=val_data, validation_steps=17, verbose=1)
# Lưu weights
model2.save_weights('imagenet_30k_Sigmoid_(200)_2.h55')

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