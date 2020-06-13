# execution command: python res_net.py
from keras import backend as K
from keras.applications import ResNet50
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

train_path = 'train'
valid_path = 'valid'
test_path = 'test'

classes = os.listdir(train_path)
image_size = 224
batch_size = 2
epochs = 10
learning_rate = 0.001

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.15,
                               height_shift_range=0.15,
                               horizontal_flip=True,
                               fill_mode='nearest')

# This is the augmentation configuration we will use for testing                    
test_datagen = ImageDataGenerator()
                               


# This is a generator that will read pictures found in
# subfolers of 'train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(image_size,image_size),
                                                    classes=classes,
                                                    batch_size=batch_size)

# This is a similar generator for validation data                                                   
valid_generator = test_datagen.flow_from_directory(valid_path,
                                                   target_size=(image_size,image_size),
                                                   classes=classes,
                                                   batch_size=batch_size, shuffle=False)
# This is a similar generator for test data                                                  
test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=(image_size,image_size),
                                                  classes=classes,
                                                  batch_size=batch_size, shuffle=False)




#Load the ResNet model
print('Loading ResNet50 Weights …')
resnet = ResNet50(include_top=False,
                  weights='imagenet',
                  input_shape=(image_size, image_size, 3))


output = resnet.get_layer(index = -1).output
output = GlobalAveragePooling2D()(output)
output = Dense(10, activation='softmax', name='predictions')(output)


model = Model(resnet.input, output)

#for layer in model.layers:
#    print(layer.trainable)
    
optimizer = SGD(lr = learning_rate, momentum = 0.9)
#optimizer = RMSprop(lr = learning_rate)
#optimizer = Adam(lr = learning_rate)
    
model.compile(loss='categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])

#print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit_generator(train_generator,
                    steps_per_epoch = 1022 // batch_size + 1,
                    validation_data = valid_generator,
                    validation_steps = 218 // batch_size + 1,
                    epochs = epochs, verbose = 2)


# Clear the values of previous plot
plt.cla()
                 
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Training & Validation Accuracy Plot')

# Clear the values of previous plot
plt.cla()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Training & Validation Loss Plot')
    
test_generator.reset()                
# Model's predictions from test dataset
pred = model.predict_generator(test_generator, steps = 214 // batch_size, verbose=1)
predicted_class_indices = np.argmax(pred,axis=1)


# Get the paths for every single image
filenames = test_generator.filenames[:(len(predicted_class_indices))]

#print(len(predicted_class_indices), len(filenames))

true_labels = []

# Get the true labels of every image
for p in filenames:
    name = os.path.dirname(p)
    true_labels.append(name)

test_generator.reset()
classes = (test_generator.class_indices)
true_test_labels = [classes[k] for k in true_labels]

# Print into csv file for manual check   
# results = pd.DataFrame({"True Labels":true_test_labels, "Predictions":predicted_class_indices})
# results.to_csv("results0.csv",index=True)

# Calculate Confusion Matrix
cm = confusion_matrix(true_test_labels, predicted_class_indices)

# Clear the values of previous plot
plt.cla()

# Plot Confusion Matrix and store it as image (.png)
plt.figure()  
plt.imshow(cm, interpolation='nearest')
plt.title('Training & Validation Loss Plot')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = '.0f'
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="orange" if cm[i, j] == cm[i, i] else "white" if cm[i, j] == 0 else "red")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

np.set_printoptions(precision=2)
plt.savefig('cm.png')

# Models evaluation.. returns to console accuracy% for our trained model
score = model.evaluate_generator(test_generator, steps = 214 // batch_size + 1, verbose = 1)
print(model.metrics_names)
print("/n/n",score)

