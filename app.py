# app.py
from flask import Flask, request, make_response,jsonify, render_template
from utils import tweetPreprocessing, toEmbedingsSequence, loadPretrainedModel, getFastTextModel

import numpy as np

app = Flask(__name__)

#MODEL_ID = "buWKkGpy"
#classifier_1 = loadPretrainedModel(MODEL_ID)

# ft_model = getFastTextModel()

CNN_MODEL_ID = "GWVBtYHV"
CNN_CLASSIFIER = loadPretrainedModel(CNN_MODEL_ID)

ENSEMBLE_IDS = ["xCdcvedN", "YBJbUNaX", "WFNrfnfd", "GWVBtYHV", "PZKzUveQ", "HMLMYxmV", "EhlKsruy"]
ENSEMBLE_CLASSIFIERS = [loadPretrainedModel(MODEL_ID) for MODEL_ID in ENSEMBLE_IDS]

print("THE CLASSIFIERS WERE LOADED TO MEMORY")
print("THE SERVER IS READY...\n")

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
    # so args works better in this case
    mesg_str = request.args.get('tweet')
    # mesg_str = request.headers.get('message')
    chosen_classifier = request.headers.get('classifier')
    working_mode = request.headers.get('mode')

    tokens_list = tweetPreprocessing(mesg_str,2)

    if working_mode == "operation":
        encoded_tweet = toEmbedingsSequence(tokens_list, ft_model)
        encoded_tweet.reshape((1,55,300))

        if chosen_classifier == "CNN":
            pred = CNN_CLASSIFIER.predict(encoded_tweet)
            label = pred.argmax()
            confidence = pred[0][label]

        elif chosen_classifier == "Ensmble-CNN":
            classes_probs_sum = np.zeros((1,5))

            for CLASSIFIERS in ENSEMBLE_CLASSIFIERS:
                # make predictions on X_test samples
                classes_probs = CLASSIFIERS.predict(encoded_tweet)
                
                classes_probs_sum += classes_probs

                label = classes_probs_sum.argmax()
                confidence = classes_probs_sum[0][label]/7

    else:
        label = 2
        confidence = 0.75

    print(tokens_list)
    print(label)
    print("THIS IS TESTING MODE...")

    result = {'label':str(label), 'confianza':str(confidence)}

    res = make_response(jsonify(result), 200)
    res.headers['Content-Type']= 'application/json'
    #res = make_response(f'Label: {label}', 200)
    #res.headers['Content-Type'] = 'text/plain'
    res.headers['Access-Control-Allow-Origin']="*"
    return res

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2500, debug=False)
