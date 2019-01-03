import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import dlib
import os
from PIL import Image
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

def reconstruct(str_arr):
    vec = np.array(str_arr.split()).astype(float)
    return vec.reshape(1,48,48)

def face_detected(img):
    img_copy = img.copy()
    img_copy = img_copy.reshape(img_copy.shape[1], img_copy.shape[2]).astype(np.uint8)
    dets = detector(img_copy)
    
    if len(dets) > 0:
        return True
    else:
        return False

def crop(img):
    h, w = img.shape[1:]

    if h%2 != 0:
        img = img[:, 1:, :]
    if w%2 != 0:
        img = img[:, :, 1:]
    
    h, w = img.shape[1:]
    center = (int(h/2), int(w/2))
    if w > h:
        img = img[:,:,center[1]-center[0]:center[1]+center[0]]
    elif h > w:
        img = img[:,center[0]-center[1]:center[0]+center[1],:]
    elif h == w:
        return img
    return img

def scale_img(img, size=(96, 96)):
    img = img.reshape(img.shape[1], img.shape[2])
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, resample=Image.LANCZOS)
    scaled_img = np.asarray(pil_img)
    scaled_img = scaled_img.reshape(1, scaled_img.shape[0], scaled_img.shape[1])
    return scaled_img

def remap(x):
    map = {0:0, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
    return map[x]

## Import fer2013 dataset
print('Loading dataset...')
fer_df = pd.read_csv('Datasets/fer2013/fer2013.csv')
print('Creating dataframe...')
fer_df['img'] = fer_df['pixels'].map(lambda x: reconstruct(x))
fer_df['org_dim'] = fer_df['img'].map(lambda x: x.shape)


## Import Jaffe Dataset
print('Import Jaffe Dataset..')
jaffe_emotion_map = {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3,
           'SA': 4, 'SU': 5, 'NE': 6}

jaffe_imgs_dict = {'emotion':[], 'img':[]}
for root,dirs,files in os.walk('Datasets/jaffe', topdown=False):
    for file in files:
        img = Image.open(os.path.join(root,file)).convert('L')
        img_arr = np.asarray(img, dtype=np.float32)
        img_arr = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1])
        jaffe_imgs_dict['img'].append(img_arr)
        emotion_index = file.find('.') + 1
        emotion = file[emotion_index:emotion_index + 2]
        jaffe_imgs_dict['emotion'].append(jaffe_emotion_map[emotion])

jaffe_imgs_df = DataFrame.from_dict(jaffe_imgs_dict)
jaffe_imgs_df['org_dim'] = jaffe_imgs_df['img'].map(lambda x: x.shape)


## Merge dfs
print('Merge dfs..')
fer_df = fer_df[['emotion','img','org_dim']]
# all_imgs = pd.concat([imgs_df,fer_df, jaffe_imgs_df])
all_imgs = pd.concat([fer_df, jaffe_imgs_df])
all_imgs = all_imgs.reset_index()

## Crop non-squares images
print('Crop non-squares images..')
detector = dlib.get_frontal_face_detector()
all_imgs['is_sqr'] = all_imgs['org_dim'].map(lambda x: x[1] == x[2])
all_imgs['img'] = all_imgs['img'].map(crop)

## Resize Images to 96x96
print('Resize Images to 96x96..')
all_imgs['is_square'] = all_imgs['img'].map(lambda x: x.shape[1] == x.shape[2])
sqr_imgs = all_imgs[all_imgs['is_square'] == True]
sqr_imgs['scaled_img'] = sqr_imgs['img'].map(lambda x: scale_img(x))

## Resize Images to 192x192
# print('Resize Images to 192x192..')
# sqr_imgs['scaled_img2'] = sqr_imgs['img'].map(lambda x: scale_img(x, size=(192,192)))

#Normalize dataset
print('Normalize dataset..')
# X = np.asarray(sqr_imgs['scaled_img2'].values.tolist())
X = np.asarray(sqr_imgs['scaled_img'].values.tolist())
# X = X.reshape(len(X),36864)
X = X.reshape(len(X),9216)
min_max_scaler = MinMaxScaler()
X_min_max = min_max_scaler.fit_transform(X)
# sqr_imgs['normalized_img'] = [arr.reshape(1, 192, 192) for arr in X_min_max]
sqr_imgs['normalized_img'] = [arr.reshape(1, 96, 96) for arr in X_min_max]
# X_scaled = scale(X)
# sqr_imgs['z_normalized_img'] = [arr.reshape(1, 192, 192) for arr in X_min_max]

## Face Detected
print('Face detected..')
sqr_imgs['face_detected'] = sqr_imgs['scaled_img'].map(face_detected)

emo = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
sqr_imgs['emotion'] = sqr_imgs['emotion'].map(remap)

dlib_dataset = sqr_imgs[sqr_imgs['face_detected'] == True][['normalized_img','emotion']]
X = np.array(dlib_dataset['normalized_img'].values.tolist())
y = np.array(dlib_dataset['emotion'].values.tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
#create cross validation set
print('Training Set:\nX: {}\ny: {}\n'.format(len(X_train), len(y_train)))
print('Test Set:\nX: {}\ny: {}\n'.format(len(X_test), len(y_test)))

Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

print('X:')
print(X_train.shape, X_test.shape)
print('Y:')
print(Y_train.shape, Y_test.shape)

print('Saving training data..')
np.save('data/X_train_scaled_jaffe_ck_dlib', X_train)
np.save('data/Y_train_scaled_jaffe_ck_dlib', Y_train)
np.save('data/X_test_scaled_jaffe_ck_dlib', X_test)
np.save('data/Y_test_scaled_jaffe_ck_dlib', Y_test)