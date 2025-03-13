import speech_recognition as sr

# Create a speech recognition object
r = sr.Recognizer()

# Open the microphone stream
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)  # Record audio from the microphone

try:
    # Recognize speech using Google Speech Recognition
    text = r.recognize_google(audio, language="en-US")
    print("You said: " + text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand your audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# from random import random
# from xml.parsers.expat import model
# from flask import Flask,render_template,url_for,request
# from flask import Flask, render_template, request, redirect
# import pandas as pd 

# from googletrans import Translator
# import warnings

# import speech_recognition as sr

# import pickle
# import numpy as np
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model

# model = load_model('model.h5')

# cv=pickle.load(open('transform.pkl','rb'))
# translator = Translator()

# recognizer = sr.Recognizer()
# audioFile = sr.AudioFile("sample/03-01-01-01-02-01-01.wav")
# with audioFile as source:
#     data = recognizer.record(source)
#     transcript = recognizer.recognize_google(data)
# print(transcript)
# translations = translator.translate(transcript, dest='en')
# transcript =  translations.text
# data = [transcript]
# vect = cv.texts_to_sequences(data)
# print(vect)