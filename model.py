# mounting drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
drive_path = '/content/gdrive/My Drive/Assignments_Drive/Case_Study_2/Medical_Data'
# specifying paths
txt_path = drive_path + '/ecgen'
img_path = drive_path + '/images'

import os
from os import listdir
import io
import time
import re
import random
import pandas as pd
import numpy as np
from numpy import zeros
from numpy import array
from numpy import asarray
from numpy import save
from bs4 import BeautifulSoup
from tqdm import tqdm
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from pickle import dump
from pickle import load
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from gensim.models import word2vec
from gensim.models import Word2Vec

import sklearn
print(sklearn.__version__)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pickle


import tensorflow
print(tensorflow.__version__)
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
# setting the random seeds
SEED = 4
os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_CUDNN_DETERMINISTIC'] = '4'  # new flag present in tf 2.0+
np.random.seed(SEED)
tensorflow.random.set_seed(SEED)

# load saved pickle files
with open(drive_path + '/impression_tokenizer.pickle', 'rb') as handle:
    impression_tokenizer = pickle.load(handle)

with open(drive_path + '/embedding_matrix_impression.pickle', 'rb') as handle:
    embedding_matrix_impression = pickle.load(handle)

# set some variables
embedding_dim = 100
pad_length_impression = 40
units = 256
vocab_tar_size = len(impression_tokenizer.word_index) + 1
impression_matrix = embedding_matrix_impression


# this class is for the x-ray features encoder
class Encoder_Xray(tensorflow.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(Encoder_Xray, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        #self.fc1 = tensorflow.keras.layers.Dense(embedding_dim)
        #self.fc2 = tensorflow.keras.layers.Dense(512)
        self.fc = tensorflow.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tensorflow.nn.relu(x)
        return x

# this class is for the xray features Attention


class BahdanauAttention_Xray(tensorflow.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention_Xray, self).__init__()
        self.W1 = tensorflow.keras.layers.Dense(units)
        self.W2 = tensorflow.keras.layers.Dense(units)
        self.W3 = tensorflow.keras.layers.Dense(units)
        self.W4 = tensorflow.keras.layers.Dense(units)
        self.V = tensorflow.keras.layers.Dense(1)
        self.add = tensorflow.keras.layers.Add()

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tensorflow.expand_dims(hidden, 1)
        # features = self.add([feature1, feature2])
        # hidden_m_with_time_axis = tensorflow.expand_dims(hidden_m, 1)
        # hidden_c_with_time_axis = tensorflow.expand_dims(hidden_c, 1)
        # score shape == (batch_size, 64, hidden_size)
        score = tensorflow.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        #score = tensorflow.nn.tanh(self.W1(features) + self.W2(hidden_m_with_time_axis)+ self.W3(hidden_c_with_time_axis))
        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tensorflow.nn.softmax(self.V(score), axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tensorflow.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
# this class is for the decoder


class Decoder(tensorflow.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units

        self.embedding = tensorflow.keras.layers.Embedding(
            vocab_size, embedding_dim, weights=[impression_matrix], mask_zero=True)
        self.gru = tensorflow.keras.layers.GRU(self.units,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')
        self.lstm = tensorflow.keras.layers.LSTM(self.units,
                                                 return_sequences=True,
                                                 return_state=True,
                                                 recurrent_initializer='glorot_uniform')
        self.fc1 = tensorflow.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tensorflow.keras.layers.Dense(vocab_size)

        self.attention1 = BahdanauAttention_Xray(self.units)
        self.attention2 = BahdanauAttention_Xray(self.units)

        # self.attention = BahdanauAttention_Xray(self.units)

    def call(self, x, features1, features2, hidden):
        # defining attention as a separate model
        context_vector1, attention_weights1 = self.attention1(
            features1, hidden)
        context_vector2, attention_weights2 = self.attention2(
            features2, hidden)
        #context_vector2, attention_weights2 = self.attention(features1, features2, hidden_m, hidden_c)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x = x.reshape(batch_size, embedding_dim, -1)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tensorflow.concat([tensorflow.expand_dims(
            context_vector1, 1), tensorflow.expand_dims(context_vector2, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        #output, state = self.gru(x)
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tensorflow.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        # return x, state

        return x, state

    def reset_state(self, batch_size):
        return tensorflow.zeros((batch_size, self.units))


def get_pretrained_model():
    """
    This function will return a pretrained model
    """
    # load the pretrained Model
    encoder1 = Encoder_Xray(embedding_dim)
    encoder2 = Encoder_Xray(embedding_dim)
    decoder = Decoder(embedding_dim, units, vocab_tar_size)
    encoder1.load_weights(drive_path + '/encoder1_weights')
    encoder2.load_weights(drive_path + '/encoder2_weights')
    decoder.load_weights(drive_path + '/decoder_weights')
    return encoder1, encoder2, decoder


def get_model():
    """
    This function will return a the model without pretrained weights
    """
    encoder1 = Encoder_Xray(embedding_dim)
    encoder2 = Encoder_Xray(embedding_dim)
    decoder = Decoder(embedding_dim, units, vocab_tar_size)
    return encoder1, encoder2, decoder
