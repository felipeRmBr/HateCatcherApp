import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

from keras.models import model_from_json
import numpy as np
import fasttext

def tweetPreprocessing(raw_tweet, format):
  """
  @user_name      ->  nombre
  https//:sdasf   ->  link
  #some_hashtag   ->  tema

  """

  import re
  raw_tweet += "  "

  # the format option only affects the way how hastags and url's are handled
  # format_1:   #some_topic -> ' _htg '  ;  https://some_link.com -> ' _url ' 
  # format_2:   #some_topic -> ' '  ;  https://some_link.com -> ' ' 
  if format == 1:
    hashtag_replace = ' _htg '
    url_replace = ' _url '
  elif format == 2:
    hashtag_replace = '  '
    url_replace = '  '

  # loweercasing
  preprocessed_tweet = raw_tweet.lower()
  # laughing variants 
  preprocessed_tweet = re.sub(r"\S+j[aeiou]j\S+", ' risa ', preprocessed_tweet)    # jajajaja$
  preprocessed_tweet = re.sub(r"\S+jsj\S+", ' risa ', preprocessed_tweet)          # jsjsjsj  
  preprocessed_tweet = re.sub(r"\S+hah\S+", ' risa ', preprocessed_tweet)          # hahahaha$
  preprocessed_tweet = re.sub(r"\S+ksk\S+", ' risa ', preprocessed_tweet)          # ksksksksk
  # inclusive language
  preprocessed_tweet = re.sub('(?<=\S)@(?=\S)', 'o', preprocessed_tweet)           # alumn@s $
  # usernames
  preprocessed_tweet = re.sub(r"@\S+", " nombre ", preprocessed_tweet)             # @usernam$
  # hashtags
  preprocessed_tweet = re.sub(r"#\S+", hashtag_replace, preprocessed_tweet)        # #some_to$
  # url's
  preprocessed_tweet = re.sub(r"https?://\S+", url_replace, preprocessed_tweet)    # https://$
  # delete numeric and non alphabetic simbols
  preprocessed_tweet = re.sub(r"[^a-z\sáéíóúñ?!]", '', preprocessed_tweet)         # 'juan ra$
  #repeated characters
  preprocessed_tweet=re.sub(r'([^rlce])(?=\1)', '', preprocessed_tweet)            # 'repettt$
  preprocessed_tweet=re.sub(r'[e]{3,}', 'e', preprocessed_tweet)                   # 'jodeeee$
  preprocessed_tweet=re.sub(r'[r]{3,}', 'rr', preprocessed_tweet)                  # 'carrrrr$
  preprocessed_tweet=re.sub(r'[l]{3,}', 'll', preprocessed_tweet)                  # 'llllama$
  preprocessed_tweet=re.sub(r'[c]{3,}', 'cc', preprocessed_tweet)                  # 'cocccio$

  token_list = word_tokenize(preprocessed_tweet)  

  return token_list


## utils ####

MAX_WORDS = 55
EMBEDDINGS_SIZE = 300

def toEmbedingsSequence(preprocessed_tweet, ft_model):
  embeddings_sequence = np.zeros((1, MAX_WORDS, EMBEDDINGS_SIZE))
  for i, w in enumerate(preprocessed_tweet[:MAX_WORDS]):
    word_vector = ft_model.get_word_vector(w).reshape(1,-1)
    embeddings_sequence[0][i] = word_vector
  return embeddings_sequence

##### GET THE PRETRAINED MODEL READY #####

def loadPretrainedModel(model_id):
  # load json and create model
  json_file = open(f'./{model_id}.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  pretrained_model = model_from_json(loaded_model_json)
  # load weights into new model
  pretrained_model.load_weights(f'./{model_id}.hdf5')

  print("Loaded model from disk")

  return pretrained_model

def getFastTextModel():
  ##### Prepare the FAST-TEXT model #####
  print('Preparing FastText model')
  ft_model = ft_model = fasttext.load_model('../FastText_3/embeddings-l-model.bin')

  return ft_model

def loadPretrainedModel_2(model_id):
  # load json and create model
  json_file = open('CNN_model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  pretrained_model = model_from_json(loaded_model_json)
  # load weights into new model
  pretrained_model.load_weights('./K-0_HTA.best.hdf5')
  print("Loaded model from disk")

  ##### Prepare the FAST-TEXT model #####
  print('Preparing FastText model')
  ft_model = ft_model = fasttext.load_model('../FastText_3/embeddings-l-model.bin')
  a = np.array([1,2,3,4,5])
