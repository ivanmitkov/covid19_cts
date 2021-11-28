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
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report, confusion_matrix

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
train_idx, val_idx = next(GroupShuffleSplit(test_size = 0.2,
                                            n_splits = 2,
                                            random_state = SEED).split(X, groups = cluster))

X_train, X_val, Y_train, Y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

print(f'X_train:', X_train.shape)
print(f'X_val:', X_val.shape)
print(f'Y_train:', Y_train.shape)
print(f'Y_val:', Y_val.shape)

NO_OF_CHANNELS = 3
input_shape=(IMAGE_SIZE, IMAGE_SIZE, NO_OF_CHANNELS)
n_classes=2

"""# **Evaluation**"""

our_vgg = tf.keras.models.load_model('kmeans_data_augm_unfreezed_vgg16.h5')
our_dense = tf.keras.models.load_model('densenet_unfreeze.h5')
our_incep = tf.keras.models.load_model('inception_unfreeze.h5')

print("-"*30)
print("VGG16")
print(our_vgg.evaluate(X_val, Y_val))
print("-"*30)
print("DenseNet")
print(our_dense.evaluate(X_val, Y_val))
print("-"*30)
print("Inception")
print(our_incep.evaluate(X_val, Y_val))
print("-"*30)

our_vgg.trainable = False
our_dense.trainable = False
our_incep.trainable = False

#Delete unused variables and clear garbage values
#del history1, model1, hist_df
import gc

gc.collect()

"""# **Ensemble**"""

def stacking_ensemble(members,input_shape,n_classes):
    commonInput = Input(shape=input_shape)
    out=[]

    for model in members:
        model._name= model.get_layer(index = 0)._name +"-test"+ str(members.index(model)+1)
        out.append(model(commonInput))

    modeltmp = concatenate(out,axis=-1)
    #modeltmp = Dense(256, activation='relu')(modeltmp)
    modeltmp = Dense(128, activation='relu')(modeltmp)
    modeltmp = Dropout(0.1)(modeltmp)
    modeltmp = Dense(n_classes, activation='softmax')(modeltmp)
    stacked_model = Model(commonInput,modeltmp)
    stacked_model.compile( loss='categorical_crossentropy',optimizer= optimizer, metrics=['accuracy'])

    return stacked_model

members =[our_vgg, our_dense, our_incep]

batch=16
optimizer= Adam(lr=5e-5, beta_1=0.9, beta_2=0.999)

stacked = stacking_ensemble(members, input_shape, n_classes)
print(stacked.summary())

stacked_checkpoint = ModelCheckpoint("stacked_best.h5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='auto', period=1)

# reduce learning rate and configure early stop
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=5,
                                         cooldown=2,
                                         min_lr=1e-8,
                                         verbose=1)

EPOCHS = 50

stacked_hist = stacked.fit(X_train,Y_train,
                            epochs=EPOCHS, #epochs,
                            verbose=1,
                            batch_size = 16,
                            validation_data= (X_val, Y_val),
                            callbacks=[reduce_learning_rate, stacked_checkpoint])



stacked.evaluate(X_val, Y_val)

#del stacked_hist, stacked
gc.collect()

stacked = tf.keras.models.load_model("stacked_best.h5")
print("-.-"*30)
print('Model Name: Stacked')
print(stacked.evaluate(X_val, Y_val))
print("-.-"*30)

"""# **Finding F1 score etc.**"""

# True values
y_true = Y_val.argmax(axis=-1)

def confusion(model):
  #Predicted values
  prob = model.predict(X_val)
  y_pred = prob.argmax(axis= -1)
  # Print the confusion matrix
  print("--"*30)
  print(confusion_matrix(y_true, y_pred))
  print("--"*30)
  # Print the precision and recall, among other metrics
  print(classification_report(y_true, y_pred, digits=6))
  print("--"*30)

our_models =  {"VGG16": our_vgg, "DenseNet": our_dense, "Inception": our_incep, "Stacked": stacked}
for m in our_models:
  print("--"*30)
  print(m)
  print("--"*30)
  confusion(our_models[m])

# compare similar metrics to the previous approaches
```final_loss, final_accuracy = stacked.evaluate(X_val, Y_val)
Y_pred = stacked.predict(X_val)
Y_pred = np.argmax(Y_pred, axis=1)

print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))
print('Precision:', precision_recall_fscore_support(y_true, Y_pred, average='weighted')[0])
print('Recall:', precision_recall_fscore_support(y_true, Y_pred, average='weighted')[1])
print('F1:', precision_recall_fscore_support(y_true, Y_pred, average='weighted')[2])
```
# graphs
cm = confusion_matrix(y_true, Y_pred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types,
                 fmt='g')
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

# accuracy plot
plt.plot(stacked_hist.history['accuracy'])
plt.plot(stacked_hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss plot
plt.plot(stacked_hist.history['loss'])
plt.plot(stacked_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#auc
pred_proba = stacked.predict(X_val)
pred = pred_proba[:, 1]
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, roc_curve
auc_score = roc_auc_score(y_true, pred)
fpr, tpr, th = roc_curve(y_true, pred)
print('AUC Score:\t', round(auc_score, 2))
plt.figure(figsize = (7, 5))
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'r')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc = 4)
plt.show()