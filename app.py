# app.py

from flask import Flask, request, make_response,jsonify, render_template
from utils import tweetPreprocessing, toEmbedingsSequence, loadPretrainedModel, getFastTextModel

app = Flask(__name__)

MODEL_ID = "buWKkGpy"

ft_model = getFastTextModel()
classifier_1 = loadPretrainedModel(MODEL_ID)

print("THE SERVER IS READY...\n")

@app.route("/")
def home():
    word_vector = ft_model.get_word_vector('hola')
    print('embedding sample: ', word_vector[:10])
    return 'FastText Ready'

@app.route("/hate_catcher")
def hateCatcherHome():
    return render_template('index.html')

@app.route("/get_vector")
def getVector():
    word = request.args.get('word')
    word_vector = ft_model.get_word_vector(word)
    print('retrieved embedding : ', word_vector[:10])

    return 'Request is complete'

@app.route("/tp")
def tp():
    str = request.args.get('tweet')
    tokens_list = prep_fnc(str,2)
    print("Tokens list : ", tokens_list)

    return 'Preprocessing complete...'


@app.route("/predict")
def predict():
    mesg_str = request.args.get('tweet')
    tokens_list = tweetPreprocessing(mesg_str)
    encoded_tweet = toEmbedingsSequence(tokens_list, ft_model)
    encoded_tweet.reshape((1,55,300))
    pred = classifier_1.predict(encoded_tweet)

    label = pred.argmax()

    print(tokens_list)

    print(pred)

    result = {'label':str(label), 'confianza':str(pred[0][label])}

    res = make_response(jsonify(result), 200)
    res.headers['Content-Type']= 'application/json'
    #res = make_response(f'Label: {label}', 200)
    #res.headers['Content-Type'] = 'text/plain'
    res.headers['Access-Control-Allow-Origin']="*"
    return res

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
