# app.py

from flask import Flask, request, make_response,jsonify

from tweet_preprocessing import preprocessing_function as prep_fnc
from keras.models import model_from_json
import nltk
import numpy as np
import fasttext
from keras.models import model_from_json

app = Flask(__name__)


## utils ####

MAX_WORDS = 55
EMBEDDINGS_SIZE = 300

def toEmbedingsSequence(preprocessed_tweet, ft_model):
  embeddings_sequence = np.zeros((1, MAX_WORDS, EMBEDDINGS_SIZE))
  for i, w in enumerate(preprocessed_tweet[:MAX_WORDS]):