import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('agg')
from tensorflow.keras.utils import to_categorical


categories = os.listdir('dataset/dataset')

label = [i for i in range(len(categories))]

label_dict = dict(zip(categories,label))

data = []
labels = []
path = 'dataset/dataset'
for category in tqdm(categories):
    data_path = os.path.join(path,category)
    image_names = os.listdir(data_path)
    for image in tqdm(image_names):
        image_path = os.path.join(data_path,image)
        images = cv2.imread(image_path)
        try:
            imag = cv2.cvtColor(images,cv2.COLOR_RGB2GRAY)
            img = cv2.resize(imag,(100,100))
            data.append(img)
            labels.append(label_dict[category])
        except Exception as e:
            print('Exception :',e)

data = np.array(data)/255.0
data = np.reshape(data,(data.shape[0],100,100,1))
labels = np.array(labels)
labels = to_categorical(labels)

np.save('data',data)
np.save('labels',labels)






