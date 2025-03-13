from flask import Flask, render_template, request, redirect
import pandas as pd
import speech_recognition as sr
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('model.h5')
cv = pickle.load(open('transform.pkl', 'rb'))

# Initialize Translator globally
translator = GoogleTranslator(source='auto', target='en')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Translate message
        message = translator.translate(message)  # Directly store translated text
        
        # Preprocess the text
        data = [message]
        vect = cv.texts_to_sequences(data)
        vect = pad_sequences(vect, maxlen=71)  # Fix padding size to match model
        
        # Make prediction
        my_prediction = np.argmax(model.predict(np.array(vect)), axis=-1)

        return render_template('result.html', prediction=my_prediction, message=message)


@app.route("/index1", methods=["GET", "POST"])
def index1():
    return render_template('index1.html')


@app.route("/predict1", methods=["GET", "POST"])
def predict1():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
                transcript = recognizer.recognize_google(data, key=None)

        print(transcript)
        
        # Translate transcript
        transcript = translator.translate(transcript)

        # Preprocess text
        data = [transcript]
        vect = cv.texts_to_sequences(data)
        vect = pad_sequences(vect, maxlen=71)  # Fix padding size

        # Make prediction
        my_prediction = np.argmax(model.predict(np.array(vect)), axis=-1)

        return render_template('index1.html', prediction=my_prediction, transcript=transcript)


@app.route('/index2')
def index2():
    return render_template('index2.html')


@app.route('/result2', methods=['POST', 'GET'])
def result2():
    if request.method == 'POST':
        result = request.form
        data = result.getlist('Name')

        print("Raw Data:", data[0])

        # Translate input text
        transcript = translator.translate(data[0])

        # Preprocess text
        data = [transcript]
        vect = cv.texts_to_sequences(data)
        vect = pad_sequences(vect, maxlen=71)  # Fix padding size

        # Make prediction
        my_prediction = np.argmax(model.predict(np.array(vect)), axis=-1)

        return render_template('result2.html', prediction=my_prediction, result=result)


if __name__ == '__main__':
    app.run(debug=True)
