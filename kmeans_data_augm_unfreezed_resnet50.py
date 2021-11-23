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
from sklearn.metrics import precision_recall_fscore_support


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


"""# **VGG16**"""
EPOCHS = 50

def resnet50():
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    from tensorflow.keras.applications.resnet50 import ResNet50
    tmodel_base = ResNet50(input_shape=input_shape,
                           include_top=False,
                           weights='imagenet')
    for layer in tmodel_base.layers:
        layer.trainable = True

    last = tmodel_base.output

    x = Conv2D(1024, (2, 2), strides=(1, 1))(last)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(rate=0.15)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(2, activation='softmax')(x)

    # Compiling model
    model = Model(inputs=tmodel_base.input, outputs=x)
    model.summary()

    opt1 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    resnet_checkpoint = ModelCheckpoint("resnetbest.h5", monitor='val_accuracy', verbose=1,
                                        save_best_only=True, mode='auto', period=1)

    model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = resnet50()


final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

Y_pred = model.predict(X_val)

Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types,
                 fmt='g')
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

# accuracy plot
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#auc
pred_proba = model.predict(X_val)
pred = pred_proba[:, 1]
auc_score = roc_auc_score(Y_true, pred)
fpr, tpr, th = roc_curve(Y_true, pred)
print('AUC Score:\t', round(auc_score, 2))
plt.figure(figsize = (7, 5))
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'r')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc = 4)
plt.show()

# Last Model
model = load_model('kmeans_data_augm_unfreezed_resnet50.h5')
final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))
print('Precision:', precision_recall_fscore_support(Y_true, Y_pred, average='weighted')[0])
print('Recall:', precision_recall_fscore_support(Y_true, Y_pred, average='weighted')[1])
print('F1:', precision_recall_fscore_support(Y_true, Y_pred, average='weighted')[2])