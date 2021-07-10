#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('wget --no-check-certificate   https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip')


# In[10]:


import zipfile,os
local_zip = 'rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('')
zip_ref.close()


# Check Location Directory

# In[11]:


os.listdir('rockpaperscissors/rps-cv-images')


# Make data directory for 3 class:

# In[12]:


base_dir=('rockpaperscissors/rps-cv-images')

rock_dir = os.path.join(base_dir, 'rock')
paper_dir = os.path.join(base_dir, 'paper')
scissors_dir = os.path.join(base_dir, 'scissors')


# Check total images rock paper scissors

# In[13]:


print('Total rock images\t: ', len(os.listdir(rock_dir)))
print('Total paper images\t: ', len(os.listdir(paper_dir)))
print('Total scissors images\t: ', len(os.listdir(scissors_dir)))


# Image augmentation and sharing of training data sets and data validation sets:

# In[78]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    validation_split=0.4)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.4)


#NOTE:
#Validation_split = Membagi data validation menjadi 40% dari total dataset


# In[79]:


train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    subset='training')

validation_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    subset='validation')


# 
# *   Building the CNN (Convolutional Neural Network) architecture
# *   Using the sequential model
# 
# 

# In[80]:


import tensorflow as tf

model = tf.keras.models.Sequential([
  # the first convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
  tf.keras.layers.MaxPooling2D(2,2),
  # the second convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # the third convolution
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # the fourth convolution
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # flatten the results to feed into a DNN
  tf.keras.layers.Flatten(),
  # 512 neuron hidden layer
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])


# Make callback function

# In[81]:


from tensorflow import keras

DESIRED_ACCURACY = 0.952

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') is not None and logs.get('acc') >= DESIRED_ACCURACY):
      print('\nReached 99.9% accuracy so cancelling training!')
      self.model.stop_training = True

callbacks = myCallback()


# * Call the compile function on the model object
# * Determine the loss function and optimizer

# In[83]:


model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])


# train the model with model.fit

# In[86]:


model.fit(
    train_generator,
    steps_per_epoch=25,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=5,
    verbose=2,
    callbacks=[callbacks]
)


# In[95]:


import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

uploaded = files.upload()

for i in uploaded.keys():
  #predicting images
  path = i
  img = image.load_img(path, target_size=(150, 150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  print(i)
  if classes[0][0]==1:
     print('this is paper!')
  elif classes[0][1]==1:
    print('this is rock!')
  elif classes[0][2]==1:
    print('this is scissors')
  else:
    print('unknown!, try again!')


# In[ ]:




