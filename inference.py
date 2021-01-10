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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pickle


import tensorflow
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
import utility

SEED = 4
os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_CUDNN_DETERMINISTIC'] = '4'  # new flag present in tf 2.0+
np.random.seed(SEED)
tensorflow.random.set_seed(SEED)
# loading the pickle files
with open(drive_path + '/impression_tokenizer.pickle', 'rb') as handle:
    impression_tokenizer = pickle.load(handle)

with open(drive_path + '/embedding_matrix_impression.pickle', 'rb') as handle:
    embedding_matrix_impression = pickle.load(handle)

data = pd.read_pickle(drive_path + '/data_final.pkl')
data_pairs = data[['UID', 'IMAGES']]
# set the variables
embedding_dim = 100
pad_length_impression = 40
units = 256
vocab_tar_size = len(impression_tokenizer.word_index) + 1
impression_matrix = embedding_matrix_impression


def final_function(dataset, encoder1, encoder2, decoder):
    # get the data pair values
    z = dataset.values
    # extract text file
    xmlfile = z[0][0][3:] + '.xml'
    # extract xray files
    xrays = z[0][1]
    # get the data of the text file
    df = utility.get_dataframe([xmlfile])
    # extract xray features
    if df.UID.values == z[0][0]:
        df['IMAGES'] = [xrays]
    features = utility.extract_features(xrays)
    df['IMAGE_FEATURE_1'] = features[xrays[0]]
    df['IMAGE_FEATURE_2'] = features[xrays[1]]
    # create the final dataframe
    data_final = pd.DataFrame()
    data_final['UID'] = df.UID.values.tolist()
    data_final['IMAGES'] = df.IMAGES.values.tolist()
    data_final['IMAGE_FEATURE_1'] = df.IMAGE_FEATURE_1.values.tolist()
    data_final['IMAGE_FEATURE_2'] = df.IMAGE_FEATURE_2.values.tolist()
    data_final['IMPRESSION'] = df.IMPRESSION.values.tolist()
    impression_tensor = impression_tokenizer.texts_to_sequences(
        data_final.IMPRESSION)
    data_final['IMPRESSION_TOKENS'] = impression_tensor
    # get the target sentence
    target = data_final.IMPRESSION.values
    target_id = data_final.IMPRESSION_TOKENS.values.tolist()[0]
    target_sent = list()
    for id in target_id:
        target_sent.append(impression_tokenizer.index_word[id])

    # call the greedy search method
    for val in data_final.values:
        print('-' * 50 + 'GREEDY_SEARCH' + '-' * 50)
        result_greedy = utility.greedy_search(
            encoder1, encoder2, decoder, val[2], val[3], target)
        greedy_score = sentence_bleu([target_sent], result_greedy)
        print('Actual Impression ', target)
        print('Generated Impression ', result_greedy)
        print('BLEU Score for Greedy search is ', greedy_score)

    # call the beam search method
    for val in data_final.values:
        print('-' * 50 + 'BEAM_SEARCH' + '-' * 50)
        result_beam = utility.beam_search(
            encoder1, encoder2, decoder, val[2], val[3], target, 5)
        beam_score = sentence_bleu([target_sent], result_beam)
        print('Actual Impression ', target)
        print('Generated Impression ', result_beam)
        print('BLEU Score for Beam search is ', beam_score)

    return result_greedy, result_beam


def generate_sample_impression(encoder1, encoder2, decoder):
    """
    This function will take a random image pair and print results
    """
    # get a random image pair
    data_pair = data_pairs.sample(1)
    z = data_pair.values
    image_pairs = z[0][1]
    # extract the xray files
    xray1 = image_pairs[0] + '.png'
    xray2 = image_pairs[1] + '.png'
    # show the xray pairs
    im1 = Image.open(img_path + '/' + xray1)
    im2 = Image.open(img_path + '/' + xray2)
    print('-' * 50 + 'Front X-Ray' + '-' * 50)
    display(im1)
    print('-' * 50 + 'Lateral X-Ray' + '-' * 50)
    display(im2)
    # call the final function
    result_greedy, result_Beam = final_function(
        data_pair, encoder1, encoder2, decoder)
