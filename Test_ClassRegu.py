# Khai báo thư viện
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
import cv2
from keras import applications

# Setup Pre Model
    #Đoạn này để khởi tạo tham số lớp Dense cuối (hằng số và ko được train)
Act = np.ndarray(dtype=int, shape=100)
for i in range(100):
    Act[i] = i+1
print(Act)
initializer = tf.keras.initializers.Constant(Act)
print(initializer)
    #Đoạn này để thêm các lớp kết nối, có cả lớp Dense cuối với trọng số mặc định như đã nêu ở trên
model = applications.VGG16(input_shape=(200, 200, 3), include_top=False, weights='imagenet')
layer_chose = model.get_layer(name='block5_pool')
output_layer_chose = layer_chose.output
x = layers.Flatten()(output_layer_chose)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Dense(units=100, activation='softmax')(x)
x = layers.Dense(units=1, kernel_initializer=initializer, use_bias=False, activation= 'relu')(x)
model2 = Model(model.input, x)
model2.summary()
model2.load_weights('imagenet_30k_ClassRegu_(200)_pycharm.h5')

# Crop ảnh test
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
anh = "MP.jpg"
img = cv2.imread(anh)
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for (X_start, Y_start, X_range, Y_range) in faces:
    cut_img = img[Y_start:Y_start+Y_range, X_start:X_start+X_range]
    cv2.imwrite("crop.jpg", cut_img)

# Test ảnh crop
img2 = image.load_img("crop.jpg", target_size=(200, 200))
img3 = image.img_to_array(img2)
img3 = np.expand_dims(img3, axis=0)
img3 /= 255.
t = model2.predict(img3)
print(t[0][0])
show = cv2.imread('crop.jpg')
h, w = show.shape[:2]
resized = cv2.resize(show, (700, 700))
cv2.imshow('Predict: Age = {0:.2f}'.format(t[0][0]), resized)
cv2.waitKey(0)

