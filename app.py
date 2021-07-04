# app.py
from flask import Flask, request, make_response,jsonify, render_template
from utils import tweetPreprocessing, toEmbedingsSequence, loadPretrainedModel, getFastTextModel

import sys
import numpy as np
import pickle

operation_mode = str(sys.argv[0])

app = Flask(__name__)

if operation_mode == "full":
    ft_model = getFastTextModel()

# CNN
CNN_MODEL_ID = "GWVBtYHV"
CNN_CLASSIFIER = loadPretrainedModel(CNN_MODEL_ID)

# CNN ENSEMBLE
ENSEMBLE_IDS = ["xCdcvedN", "YBJbUNaX", "WFNrfnfd", "GWVBtYHV", "PZKzUveQ", "HMLMYxmV", "EhlKsruy"]
ENSEMBLE_CLASSIFIERS = [loadPretrainedModel(MODEL_ID) for MODEL_ID in ENSEMBLE_IDS]

# SVC
# load the text_model 
with open(f'./models/djwYeF.tm', 'rb') as file_handler:
  text_model = pickle.load(file_handler)

# load the classifier
with open(f'./models/KAcOYq.svc', 'rb') as file_handler:
  svc = pickle.load(file_handler)

print("THE CLASSIFIERS WERE LOADED TO MEMORY")
print("THE SERVER IS READY...\n")

print("OPERATION_MODE: ", operation_mode)

### GET THINGS READY
@app.route("/")
def home():
    word_vector = ft_model.get_word_vector('hola')
    print('embedding sample: ', word_vector[:10])
    return 'FastText Ready'

### INICIO
@app.route("/hate_catcher")
def hateCatcherHome():
    return render_template('index.html')

### GET VECTOR (TESTING SERVICE)
@app.route("/get_vector")
def getVector():
    word = request.args.get('word')
    word_vector = ft_model.get_word_vector(word)
    print('retrieved embedding : ', word_vector[:10])

    return 'Request is complete'

### TWEET PRESPROCESSING (TESTING SERVICE)
@app.route("/tp")
def tp():
    str = request.args.get('tweet')
    tokens_list = prep_fnc(str,2)
    print("Tokens list : ", tokens_list)

    return 'Preprocessing complete...'

### PREDICT
@app.route("/predict")
def predict():

    # headers cant receive emojis or new line characters
    # that's the reason why we use request.args 
    mesg_str = request.args.get('tweet')
    # mesg_str = request.headers.get('message')
    chosen_classifier = request.headers.get('classifier')
    working_mode = request.headers.get('mode')

    if chosen_classifier == "CNN":
        if operation_mode == "full":
            tokens_list = tweetPreprocessing(mesg_str,2)
            encoded_tweet = toEmbedingsSequence(tokens_list, ft_model)
            encoded_tweet.reshape((1,55,300))

            pred = CNN_CLASSIFIER.predict(encoded_tweet)
            label = pred.argmax()
            confidence = pred[0][label]
        else:
            label = request.headers.get('test_label')
            confidence = 0.75

            print("THIS IS TESTING MODE...")

    elif chosen_classifier == "Ensmble-CNN":
        if operation_mode == "full":
            tokens_list = tweetPreprocessing(mesg_str,2)
            encoded_tweet = toEmbedingsSequence(tokens_list, ft_model)
            encoded_tweet.reshape((1,55,300))
            
            classes_probs_sum = np.zeros((1,5))

            for CLASSIFIERS in ENSEMBLE_CLASSIFIERS:
                # make predictions on X_test samples
                classes_probs = CLASSIFIERS.predict(encoded_tweet)
                
                classes_probs_sum += classes_probs

                label = classes_probs_sum.argmax()
                confidence = classes_probs_sum[0][label]/7
        
            print(tokens_list)
            print(label)

        else:
            label = request.headers.get('test_label')
            confidence = 0.75

            print("THIS IS TESTING MODE...")

        print(tokens_list)
        print(label)

    elif chosen_classifier == "SVC":
        label = svc.predict(text_model.transform([mesg_str])[0])
        confidence = 0

        print(label)
    
    
    result = {'label':str(label), 'confianza':str(confidence)}

    res = make_response(jsonify(result), 200)
    res.headers['Content-Type']= 'application/json'
    #res = make_response(f'Label: {label}', 200)
    #res.headers['Content-Type'] = 'text/plain'
    res.headers['Access-Control-Allow-Origin']="*"
    return res

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2500, debug=False)
