import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras
import tensorflow as tf
from model_training.softmax import Softmax
import os

class TrainSoftmax:
    def __init__(self, args):
        self.args = args
        with open(self.args['embeddings'], 'rb') as f:
            self.data = pickle.loads(f.read())
    def train(self):
        
        le = LabelEncoder()
        labels = le.fit_transform(self.data['names'])
        labels = labels.reshape(-1, 1)
        ohe = OneHotEncoder()
        labels = ohe.fit_transform(labels).toarray()

        lis = []
        for embedding_set in self.data['embeddings']:
            for embedding in embedding_set:
                embedding = embedding['embedding']
                lis.append(embedding)
        embeddings = np.array(lis)
        print(embeddings.shape[1])

        cv = KFold(n_splits=len(embeddings), shuffle=True, random_state=42)

        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        softmax = Softmax((input_shape,), num_classes=len(le.classes_))
        model = softmax.build()

        for train_idx, valid_idx in cv.split(embeddings):
            x_train, y_train = embeddings[train_idx], labels[train_idx]
            x_valid, y_valid = embeddings[valid_idx], labels[valid_idx]

            model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_valid, y_valid))

        model.save('softmax_model.h5')
        with open('label_encoder.pickle', 'wb') as f:
            f.write(pickle.dumps(le))

# TrainSoftmax({'embeddings' : os.path.join('embeddings.pickle')}).train()
