# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tensorflow import keras
from tqdm import tqdm
import tensorflow
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
K.clear_session()
import itertools
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import *
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, roc_curve
from sklearn.cluster import KMeans


#Data to be loaded as X_train,Y_train and X_test,Y_test with onehot encoded labels
disease_types = ['COVID', 'non-COVID']
data_dir = r'C:\Users\ivan.mitkov\Documents\unibit\data'
train_dir = os.path.join(data_dir)

train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])

train = pd.DataFrame(train_data, columns=['File', 'DiseaseID', 'Disease Type'])
train.head()

train.info()

SEED = 42
train = train.sample(frac=1, random_state=SEED)
train.index = np.arange(len(train))  # Reset indices
train.head()

train.info()


IMAGE_SIZE = 224


def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath))  # Loading a color image is the default flag


# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)


X = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X = X / 255.
print('Train Shape: {}'.format(X.shape))

y = train['DiseaseID'].values
y = to_categorical(y, num_classes=2)
print(y.shape)

BATCH_SIZE = 64

# Split the train and validation sets through kmeans
X0 = X[train['DiseaseID'] == 0, :, :].reshape(len(X[train['DiseaseID'] == 0, :, :]), -1)
X1 = X[train['DiseaseID'] == 1, :, :].reshape(len(X[train['DiseaseID'] == 1, :, :]), -1)

k = 60
kmeans = KMeans(k)
cluster0 = kmeans.fit_predict(X0)
cluster1 = kmeans.fit_predict(X1)
cluster1 += k
cluster = np.concatenate([cluster0, cluster1])

# split data
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
train_idx, val_idx = next(GroupShuffleSplit(test_size = 0.2,
                                            n_splits = 2,
                                            random_state = SEED).split(X, groups = cluster))

X_train, X_val, Y_train, Y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

print(f'X_train:', X_train.shape)
print(f'X_val:', X_val.shape)
print(f'Y_train:', Y_train.shape)
print(f'Y_val:', Y_val.shape)

# reshape
input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)

# reduce learning rate and configure early stop
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=5,
                                         cooldown=2,
                                         min_lr=1e-8,
                                         verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

EPOCHS = 50

"""# **VGG16**"""

from tensorflow.keras.applications import VGG16
tmodel_base = VGG16(input_shape = input_shape,
                                include_top = False,
                                weights = 'imagenet')
for layer in tmodel_base.layers:
    layer.trainable = False

#Getting desired layer output
last_layer = tmodel_base.get_layer('block5_pool')
last = last_layer.output

x = Flatten()(last)
x = Dense(512, activation = 'relu')(x)
x = Dropout(rate = 0.25)(x)
x = Dense(2, activation = 'softmax')(x)
#Compiling model
model1 = Model(inputs = tmodel_base.input, outputs = x, name = 'VGG16')
opt1 = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999)


model1.compile(optimizer = opt1 , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model1.summary()

vgg_checkpoint = ModelCheckpoint("vgg_best_kmeans.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

history1 = model1.fit(X_train, Y_train,  epochs=EPOCHS, validation_data = (X_val, Y_val)
                       #,class_weight=class_weight
                      ,callbacks=[vgg_checkpoint])

# make predictions
Y_pred = model1.predict(X_val)
Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

# confisopn matrix
cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types, fmt='g')
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

# evaluate the model
final_loss, final_accuracy = model1.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

# summarize history for accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])

# AUC
pred_proba = model1.predict(X_val)
auc_score = roc_auc_score(Y_val, pred_proba)
fpr, tpr, th = roc_curve(Y_val.argmax(axis=1), pred_proba.argmax(axis=1))
print('AUC Score:\t', round(auc_score, 2))

plt.figure(figsize = (7, 5))
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'r')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc = 4)
plt.show()