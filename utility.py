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

# load pickle files
with open(drive_path + '/impression_tokenizer.pickle', 'rb') as handle:
    impression_tokenizer = pickle.load(handle)

with open(drive_path + '/embedding_matrix_impression.pickle', 'rb') as handle:
    embedding_matrix_impression = pickle.load(handle)

# set variables
embedding_dim = 100
pad_length_impression = 40
units = 256
vocab_tar_size = len(impression_tokenizer.word_index) + 1
impression_matrix = embedding_matrix_impression


def get_dataframe(xml_files):
    """
    This function will read the xml file and return the dataframe
    """
    # now lets read each xml file and create a dataframe fron the data

    # rows th the dataframe
    # 1. UID : The unique file id for each xml file
    # 2. FINDINGS : The findings that the doctor writes after viewing the x-rays of the patient
    # 3. NS_FINDINGS : The number of sentences in the findings section
    # 4. IMPRESSIONS : The final diagnosys that the doctor writes
    # 5. NS_IMPRESSIONS : The number of sentences in the impression section
    # 4. IMAGES : The list of images that are associated with a report
    # 7. NO_IMAGES : The number of images in a report

    # This list will be used to used store the contents of each xml file
    rows_list = []
    for xmlfile in xml_files:
        # file path stores the path of the xml files
        filepath = txt_path + '/' + xmlfile
        # this dict is used to contents of a xml file as {tag: value}
        dict_row = {}
        with open(filepath, "r") as f:
            # reading the file
            contents = f.read()
            # here we use lxml to parse the contents of xml file
            soup = BeautifulSoup(contents, 'lxml')
            # this piece of code is used to find the id associated with each file
            uid_tags = soup.findAll('uid')
            # extract the uid from tag and put the uid in the directory
            # this piece of code will check if there are more than one uids in the xml
            cnt = 0
            for ut in uid_tags:
                # increase cnt value
                cnt += 1
                # check if more than one ids are present
                if cnt > 1:
                    # prit and break
                    print('more than one ids')
                    break
                # get the uid value
                file_id = ut.get('id')
                # put the value in the directory
                dict_row['UID'] = file_id
            # this piece of code extracts the text from the xml file where the tag is abstracttext
            tags = soup.findAll(['abstracttext'])
            for t in tags:
                # extract labels
                label = t.get("label")

                # extract the FINDINGS
                if label == 'FINDINGS':
                    f_text = t.text
                    # put the values in a directory
                    dict_row['FINDINGS'] = f_text
                    # lets add the no. of sentences
                    # first we need to add a fullstop to the end in case the doctor forgot to put it
                    if f_text.endswith('.'):
                        f_sen = f_text.split('.')
                    else:
                        f_text = f_text + '.'
                        f_sen = f_text.split('.')
                    # array to store the sentences
                    sent_arr = []
                    for sent in f_sen:
                        # check if the sentence is blank
                        if sent != '':
                            # append the array
                            sent_arr.append(sent)
                    # find no. of sentences
                    ns_findings = len(sent_arr)
                    # add f_sen to the directory
                    dict_row['NS_FINDINGS'] = ns_findings

                # extract the IMPRESSION
                if label == 'IMPRESSION':
                    i_text = t.text
                    i_text = '<start> ' + i_text + ' <end>'
                    # put the values in a directory
                    dict_row['IMPRESSION'] = i_text
                    # lets add the no. of sentences
                    # first we need to add a fullstop to the end in case the doctor forgot to put it
                    i_text = i_text + '.'
                    i_sen = i_text.split('.')
                    # array to store the sentences
                    sent_arr = []
                    for sent in i_sen:
                        # check if the sentence is blank
                        if sent != '':
                            # append the array
                            sent_arr.append(sent)
                    # find no. of sentences
                    ns_impression = len(sent_arr)
                    # add f_sen to the directory
                    dict_row['NS_IMPRESSION'] = ns_impression

            # This piece of code will extract the image names from the XML files
            itags = soup.findAll(['parentimage'])
            # this array will hold the images in a xml file as there are more than one images in a xml file
            x_ray_images = []
            # for each image in the file we will put it in the array
            for element in itags:
                # extract the image name from the file
                x_ray = element.get('id')
                # append the array
                x_ray_images.append(x_ray)
            # finally put the array in the row dict
            dict_row['IMAGES'] = x_ray_images
            # lets find the no of images in a xml file
            no_images = len(x_ray_images)
            # add this to the dict
            dict_row['NO_IMAGES'] = no_images

        # append the dict to the row (for each xml file we will have a dict_row(dict of data) in the rows_list)
        rows_list.append(dict_row)

    # finally create a dataframe from the rows_list
    df = pd.DataFrame(rows_list)
    df = df[['UID', 'IMPRESSION']]
    return df

    # extract features from each x-ray in the directory


def extract_features(img_pair):
    """
    This function will read the image pair and return the features
    """
    # get the files
    files = img_pair
    # load the model
    # loading and re-structureing the model final shape = (1,1024)
    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape)
    base_model = DenseNet121(
        include_top=False,
        input_tensor=img_input,
        input_shape=input_shape,
        pooling="avg")
    x = base_model.output
    predictions = Dense(14, activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)
    model.load_weights(drive_path + '/CheXNet_weights.h5')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # extract features from each photo
    features = dict()
    features_list = list()
    l = len(files)
    idx = 0
    for name in files:

        # load an image from file
        filename = img_path + '/' + name + '.png'
        # get image id and file type
        image_id = name

        image = load_img(filename, target_size=(224, 224, 3))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the model
        image = preprocess_input(image)
        # get features
        feature = model(image, training=False)
        # store feature
        # store in the np array with the image_id
        #temp = [image_id,feature]
        # features_list.append(temp)
        # store in the dict
        features[image_id] = feature
    # validation

    return features


# this function will do greedy search
def greedy_search(encoder1, encoder2, decoder, img1, img2, target):
    """
    This function will return the greedy search results
    """
    # reset hidden states
    hidden = decoder.reset_state(batch_size=1)
    # reshape img vectors
    img1 = tensorflow.keras.backend.reshape(img1, shape=(1, -1))
    img2 = tensorflow.keras.backend.reshape(img2, shape=(1, -1))
    # get image features
    features1 = encoder1(img1)
    features2 = encoder2(img2)
    # decoder input = start
    dec_input = tensorflow.expand_dims(
        [impression_tokenizer.word_index['<start>']], 0)
    result = []
    result.append('<start>')
    # loop for pad length
    for i in range(pad_length_impression):

        # get predections
        predictions, hidden = decoder(dec_input, features1, features2, hidden)
        # get argmax of predicted id
        predicted_id = predictions.numpy().argmax()
        result.append(impression_tokenizer.index_word[predicted_id])
        # if end is reached return
        if impression_tokenizer.index_word[predicted_id] == '<end>':
            return result
        # predicted output = next input
        dec_input = tensorflow.expand_dims([predicted_id], 0)
    # return
    return result


# this function will do beam search
def beam_search(encoder1, encoder2, decoder, img1, img2, target, beam_width=5):
    """
    This function will return the beam search results
    """
    # reset hidden states
    hidden = decoder.reset_state(batch_size=1)
    # reshape img vectors
    img1 = tensorflow.keras.backend.reshape(img1, shape=(1, -1))
    img2 = tensorflow.keras.backend.reshape(img2, shape=(1, -1))
    # get image features
    features1 = encoder1(img1)
    features2 = encoder2(img2)
    # decoder input = start
    start = [impression_tokenizer.word_index['<start>']]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < pad_length_impression:
        temp = []
        for s in start_word:

            dec_input = pad_sequences([[s[0][-1]]])
            # get the predections
            preds, hidden = decoder(dec_input, features1, features2, hidden)

            # Getting the top <beam_width>(n) predictions
            top_words = np.argsort(preds).flatten()
            word_preds = top_words[-beam_width:]

            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_width:]

    # update start word
    start_word = start_word[-1][0]
    # intermediate caption
    intermediate_caption = [impression_tokenizer.index_word[i]
                            for i in start_word]
    # generate final captions
    final_caption = []
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    # return final captions
    final_caption = final_caption[1:]
    final_caption.insert(0, '<start>')
    if len(final_caption) <= 39:
        final_caption.append('<end>')
    return final_caption
