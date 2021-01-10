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

# load the pickle files
with open(drive_path + '/impression_tokenizer.pickle', 'rb') as handle:
    impression_tokenizer = pickle.load(handle)

with open(drive_path + '/embedding_matrix_impression.pickle', 'rb') as handle:
    embedding_matrix_impression = pickle.load(handle)

# load the data
data = pd.read_pickle(drive_path + '/data_final.pkl')
Y_Data = data.IMPRESSION
data.drop('IMPRESSION', axis=1, inplace=True)

# train test split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    data, Y_Data, test_size=0.2, random_state=42)


data['IMPRESSION'] = Y_Data

# get the shapes of train cv and test data

# lets convert the Y_Train to a dataframe
Y_DTrain = pd.DataFrame(data=Y_Train.to_list(), columns=['IMPRESSION'])
Y_Train = Y_DTrain


# lets convert the Y_Test to a dataframe
Y_DTest = pd.DataFrame(data=Y_Test.to_list(), columns=['IMPRESSION'])
Y_Test = Y_DTest

# set variables
embedding_dim = 100
pad_length_impression = 40
units = 256
vocab_tar_size = len(impression_tokenizer.word_index) + 1
impression_matrix = embedding_matrix_impression


# model training

# this function will take the dataframe and return the tokenized and padded findings and impression
def tokenize(dataset):
    # tokenizing and padding findings
    #findings_tensor = findings_tokenizer.texts_to_sequences(dataset.FINDINGS)

    #findings_tensor = tensorflow.keras.preprocessing.sequence.pad_sequences(findings_tensor, maxlen = pad_length_findings, padding='post')
    # tokenizing and padding impression

    impression_tensor = impression_tokenizer.texts_to_sequences(
        dataset.IMPRESSION)

    impression_tensor = tensorflow.keras.preprocessing.sequence.pad_sequences(impression_tensor, maxlen=pad_length_impression,
                                                                              padding='post')

    return impression_tensor


# this function will load the cleaned images, findings and impression
def load_dataset(dataset, purpose='testing'):
    # creating cleaned input, output pairs
    if purpose == 'training':
        # oversample_dataset = oversample_data(dataset)
        # impression_tensor = tokenize(oversample_dataset)
        # img_feature_1, img_feature_2 = load_imgs(oversample_dataset)
        impression_tensor = tokenize(dataset)
        img_feature_1, img_feature_2 = load_imgs(dataset)
    else:
        impression_tensor = tokenize(dataset)
        img_feature_1, img_feature_2 = load_imgs(dataset)
    return img_feature_1, img_feature_2, impression_tensor


# this function will convert the image array into numpy
def load_imgs(dataset):
    img_feature_1 = dataset.IMAGE_FEATURE_1.values
    tmp_arr_1 = np.zeros((len(img_feature_1), 1024))
    img_feature_2 = dataset.IMAGE_FEATURE_2.values
    tmp_arr_2 = np.zeros((len(img_feature_2), 1024))
    # print(tmp_arr_train.shape)
    i = 0
    for r in img_feature_1:
        # print(r)
        tmp_arr_1[i] = r
        i += 1

    img_feature_1 = tmp_arr_1

    i = 0
    for r in img_feature_2:
        # print(r)
        tmp_arr_2[i] = r
        i += 1

    img_feature_2 = tmp_arr_2

    return img_feature_1, img_feature_2


# create the train dataframe
data_train = pd.DataFrame()
data_train['UID'] = X_Train.UID.values.tolist()
data_train['IMAGES'] = X_Train.IMAGES.values.tolist()
data_train['IMAGE_FEATURE_1'] = X_Train.IMAGE_FEATURE_1.values.tolist()
data_train['IMAGE_FEATURE_2'] = X_Train.IMAGE_FEATURE_2.values.tolist()
data_train['FINDINGS'] = X_Train.FINDINGS.values.tolist()
data_train['IMPRESSION'] = Y_Train.IMPRESSION.values
img_feature_1_train, img_feature_2_train, impression_tensor_train = load_dataset(
    data_train, purpose='training')
data_train_impression_tensor = impression_tokenizer.texts_to_sequences(
    data_train.IMPRESSION)
data_train['IMPRESSION_TOKENS'] = data_train_impression_tensor
train_len = impression_tensor_train.shape[0]


# create the test dataframe
data_test = pd.DataFrame()
data_test['UID'] = X_Test.UID.values.tolist()
data_test['IMAGES'] = X_Test.IMAGES.values.tolist()
data_test['IMAGE_FEATURE_1'] = X_Test.IMAGE_FEATURE_1.values.tolist()
data_test['IMAGE_FEATURE_2'] = X_Test.IMAGE_FEATURE_2.values.tolist()
data_test['FINDINGS'] = X_Test.FINDINGS.values.tolist()
data_test['IMPRESSION'] = Y_Test.IMPRESSION.values
img_feature_1_test, img_feature_2_test, impression_tensor_test = load_dataset(
    data_test)
data_test_impression_tensor = impression_tokenizer.texts_to_sequences(
    data_test.IMPRESSION)
data_test['IMPRESSION_TOKENS'] = data_test_impression_tensor
test_len = impression_tensor_test.shape[0]

# initializing the optimizer and the loss function
optimizer = tensorflow.keras.optimizers.Adam(0.01)
loss_object = tensorflow.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tensorflow.math.logical_not(tensorflow.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tensorflow.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    loss_ /= pad_length_impression
    return tensorflow.reduce_mean(loss_)


@tensorflow.function
def train_step(encoder1, encoder2, decoder, img_tensor1, img_tensor2, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    # hidden_m = decoder.reset_state(batch_size=target.shape[0])
    # hidden_c = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tensorflow.expand_dims(
        [impression_tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tensorflow.GradientTape() as tape:

        features1 = encoder1(img_tensor1)
        features2 = encoder2(img_tensor2)

        for i in range(1, target.shape[1]):
            # get predictions
            predictions, hidden = decoder(
                dec_input, features1, features2, hidden)

            loss += loss_function(target[:, i], predictions)
            # using teacher forcing
            dec_input = tensorflow.expand_dims(target[:, i], 1)
            # print(dec_input.shape)
    total_loss = loss

    trainable_variables = encoder1.trainable_variables + \
        encoder2.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return total_loss


# function to calculate blue score
def calc_blue(encoder1, encoder2, decoder, img1, img2, target):
    # initializing the hidden state for each batch
    # because the impressions are not related from image to image
    hidden = decoder.reset_state(batch_size=1)

    # reshape image features
    img1 = tensorflow.keras.backend.reshape(img1, shape=(1, -1))
    img2 = tensorflow.keras.backend.reshape(img2, shape=(1, -1))
    # get the target sentence
    target_sent = list()
    for t in target:
        target_sent.append(impression_tokenizer.index_word[t])
    # get image features
    features1 = encoder1(img1)
    features2 = encoder2(img2)
    # initial decoder input
    dec_input = tensorflow.expand_dims(
        [impression_tokenizer.word_index['<start>']], 0)
    # initialize the result array
    result = []
    result.append('<start>')
    # loop for the entire pad lenght
    for i in range(pad_length_impression):
        # predict
        predictions, hidden = decoder(dec_input, features1, features2, hidden)

        # calculate max
        predicted_id = predictions.numpy().argmax()
        # append the predicted word to result array
        result.append(impression_tokenizer.index_word[predicted_id])
        # if '<end>' is reached
        if impression_tokenizer.index_word[predicted_id] == '<end>':
            # calculate bleu score and return
            score = sentence_bleu([target_sent], result)
            return score
        # the next input to the model is predected at this step
        dec_input = tensorflow.expand_dims([predicted_id], 0)

    # calculate score at the end and return it
    score = sentence_bleu([target_sent], result)
    return score


def train_model(encoder1, encoder2, decoder, EPOCHS=200):
    """
    This function will take the model and then train it
    """
    # setting some vartables
    BUFFER_SIZE = len(impression_tensor_train)
    BATCH_SIZE = 128
    steps_per_epoch = len(impression_tensor_train) // BATCH_SIZE
    #embedding_dim = 400
    units = 256
    # vocab_inp_size = len(findings_tokenizer.word_index)+1
    vocab_tar_size = len(impression_tokenizer.word_index) + 1

    dataset_train = tensorflow.data.Dataset.from_tensor_slices(
        (img_feature_1_train, img_feature_2_train, impression_tensor_train)).shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)

    example_img1_batch, example_img2_batch, example_target_batch = next(
        iter(dataset_train))

    training_loss = tensorflow.keras.metrics.Mean(name='training_loss')

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor1_train, img_tensor2_train, target_train)) in enumerate(dataset_train):
            t_loss = train_step(encoder1, encoder2, decoder, img_tensor1_train,
                                img_tensor2_train, target_train)
            #total_loss += t_loss
            training_loss(t_loss)

        print ('Epoch {} Training Loss {:.6f}'.format(
            epoch + 1, training_loss.result()))

        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # calculate blue scores
    # train bleu score
    train_bleu = 0
    # get train df values
    data_train_vals = data_train.values
    for val in data_train_vals:
        bleu = calc_blue(encoder1, encoder2, decoder, val[2], val[3], val[-1])
        train_bleu += bleu
    train_bleu /= train_len

    print ('Train BLEU score {:.6f}'.format(train_bleu))

    # test bleu score
    test_bleu = 0
    # get test df values
    data_test_vals = data_test.values
    for val in data_test_vals:
        bleu = calc_blue(encoder1, encoder2, decoder, val[2], val[3], val[-1])
        test_bleu += bleu
    test_bleu /= test_len

    print ('Test BLEU score {:.6f}'.format(test_bleu))

    return encoder1, encoder2, decoder
