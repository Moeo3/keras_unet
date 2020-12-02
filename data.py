from PIL import Image
import os
import numpy as np



def read_train_data(data_path = './train', feature = 'image', label = 'label'):
    feature_path = os.path.join(data_path, feature)
    label_path = os.path.join(data_path, label)


    features = []
    for png in os.listdir(feature_path):
        img = Image.open(os.path.join(feature_path, png))
        img = np.asarray(img)[:, :, None]
        features.append(img/ 255)
        pass
    features = np.asarray(features).astype(np.float32)
    
    labels = []
    for png in os.listdir(label_path):
        img = Image.open(os.path.join(label_path, png))
        img = np.asarray(img)[:, :, None]
        labels.append(img / 255)
        pass
    labels = np.asarray(labels).astype(np.float32)

    return features, labels