import os
import cv2
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import pytesseract as pt
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
from keras.models import Model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.applications import InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten, Input
import tensorboard

#converting data

col = ['filepath','xmin','xmax','ymin','ymax']
rows = []

for k in range(248):    
    try:
        xmlparse = xet.parse('images/N'+str(k+1)+'.xml')
        root = xmlparse.getroot()
        for i in root:
            if i.tag=='object':
                for j in i:
                    if j.tag=='bndbox':
                        xmin = j.find("xmin").text
                        xmax = j.find("xmax").text
                        ymin = j.find("ymin").text
                        ymax = j.find("ymax").text
                        rows.append({"filepath":"images/N"+str(k+1)+".xml",
                                    "xmin":xmin,
                                    "xmax":xmax,
                                    "ymin":ymin,
                                    "ymax":ymax})
    except FileNotFoundError:
        continue
    
df = pd.DataFrame(rows,columns=col)
df.to_csv('output.csv', index=False)
print("1")
filename = df['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('images',filename_image)
    return filepath_image
getFilename(filename)

image_path = list(df['filepath'].apply(getFilename))

#Normalization

data = []
output = []


for i in range(len(image_path)):
    img = image_path[i]
    img_array = np.array(cv2.imread(img))
    rows,col,width= img_array.shape
    load_img = tf.keras.utils.load_img(img, target_size=(224,224))
    ld_img_array = tf.keras.utils.img_to_array(load_img)
    #normalize
    norm_img = np.divide(load_img,255.0)
    xmax = df.loc[i]["xmax"]
    ymax = df.loc[i]["ymax"]
    xmin = df.loc[i]["xmin"]
    ymin = df.loc[i]["ymin"]
    nxmax = np.divide(float(xmax),float(col))
    nymax = np.divide(float(ymax),float(rows))
    nxmin = np.divide(float(xmin),float(col))
    nymin = np.divide(float(ymin),float(rows))
    norm_labels = (nxmin, nxmax, nymin, nymax)

    data.append(norm_img)
    output.append(norm_labels)
print("2")   
X = np.array(data,dtype=np.float32)
Y = np.array(output,dtype=np.float32)
#splitting the data
train_x,test_x,train_y,test_y = train_test_split(X,Y,train_size=0.8,random_state=0)

#Model making
Incep_renet = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
main_model = Incep_renet.output
main_model = Flatten()(main_model)
main_model = Dense(500, activation="relu")(main_model)
main_model = Dense(250, activation="relu")(main_model)
main_model = Dense(4, activation="sigmoid")(main_model)
print("3")
model = Model(inputs = Incep_renet.input, outputs = main_model)
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
print("4")
tfb = TensorBoard('object_detection')
history = model.fit(x=train_x,y=train_y,batch_size=10,epochs=50, validation_data=(test_x,test_y),callbacks=[tfb])
print('5')
model.save('./object_detection.h5')
print('6')



