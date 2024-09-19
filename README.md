# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model
![Screenshot 2024-09-19 154721](https://github.com/user-attachments/assets/9929ed55-fbe6-4a2d-b048-afc65cb0cff6)

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test

### STEP 6:
Build the convolutional neural network model

### STEP 7:
Train the model with the training data

### STEP 8:
Plot the performance plot

### STEP 9:
Evaluate the model with the testing data

### STEP 10:
Fit the model and predict the single input

## PROGRAM

### Name: PAVITHRAN MJ
### Register Number: 212223240112
```py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape


X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')
print("Pavithran MJ")

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

 type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
print("Pavithran MJ ")

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

Name:Pavithran MJ

Register Number: 212223240112

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

print("Pavithran MJ")
metrics[['accuracy','val_accuracy']].plot()

print("Pavithran MJ")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print('Pavithran MJ')

print('Pavithran MJ')
print(classification_report(y_test,x_test_predictions))

img = image.load_img('imagefive.jpeg')

type(img)

img = image.load_img('imagefive.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print('Pavithran MJ')
print(x_single_prediction)


plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print("Pavithran MJ")
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-09-19 153423](https://github.com/user-attachments/assets/bd71f979-34bc-4381-8299-c3542c1c2a84)
![Screenshot 2024-09-19 153415](https://github.com/user-attachments/assets/45183fa7-621c-4998-af24-bf6c4be0e4e9)


### Classification Report
![Screenshot 2024-09-19 154301](https://github.com/user-attachments/assets/aca03f50-bcff-4713-a5e2-0383f144bdc7)

### Confusion Matrix
![Screenshot 2024-09-19 154253](https://github.com/user-attachments/assets/74ba2838-d3f2-41c4-a9ef-75e9fef9a2d9)

### New Sample Data Prediction
## Input
![Screenshot 2024-09-19 154437](https://github.com/user-attachments/assets/028df5a2-46ed-4cf2-bab0-b9eb4af36bd8)

## Output
![Screenshot 2024-09-19 154631](https://github.com/user-attachments/assets/732eb62f-67db-45b1-bb43-777dcbd74dd5)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
