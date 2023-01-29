import tensorflow as tf
import cv2
import numpy as np
import plotly as px


model = tf.keras.models.load_model('./object_detection.h5')
print('done')

path = '../images/N16.jpeg'
test_image = cv2.imread(path)
test_image = np.array(test_image, dtype=np.uint8)
test_image1 = tf.keras.utils.load_img(test_image, target_size=(224,224))
test_image1_array = tf.keras.utils.img_to_array(test_image1)/255.0
h,w,d = test_image.shape

test_figure = px.imshow(test_image)
test_figure.update_layout(width = 700, height = 500, margin = dict(l=10, r=10, b=10, t=10), xaxsis_title="Image 16")

test_array = test_image1_array.reshape(1,224,224,3)

coordinates = model.predict(test_array)
denorm = np.array([w,w,h,h])
coordinates = coordinates * denorm
coordinates = np.int32(coordinates)

#Drawing bounding on top of image

test_xmin, test_xmax, test_ymin, test_ymax = coordinates[0]

point1 = [test_xmin,test_ymin]
point2 = [test_xmax,test_ymax]

cv2.rectangle(test_image,point1, point2, (0,255,0), 3)

test_figure = px.imshow(test_image)
test_figure.update_layout(width = 700, height = 500, margin = dict(l=10, r=10, b=10, t=10))

