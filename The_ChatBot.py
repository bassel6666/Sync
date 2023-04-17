import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

limitizer = WordNetLemmatizer()

intents = json.loads(open('Intent.json').read())  # load the intents.json data and convert it into a python dictionary object.
wordss = pickle.load(open('words.pkl', 'rb'))  # load the words.pkl file and convert it into a python dictionary object.
classes = pickle.load(open('classes.pkl', 'rb'))  # load the classes.pkl file and convert it into a python dictionary object.
model = load_model('Bassel_ChatBot.h5')  # load the chat model.h5 file.


def clean_up_sentence(sentence):  # function to convert a sentence into a series of lower case letters and remove any extra characters like dots or whitespaces.
    sentence_words = nltk.word_tokenize(sentence)  # tokenize the sentence into words.
    sentence_words = [limitizer.lemmatize(word.lower()) for word in sentence_words]  # lemmatize
    return sentence_words  # return the list of words.


def bag_of_words(sentence):  # function to define the bag of words. This function takes two arguments. the sentence and the word_features. The sentence is the list of words and the word_features is a list of words that are the most important words. The function returns a binary vector of 0 or 1. 1 means that the word is in the sentence or in the word_features. If the word is not in the word_features list, the function returns 0. This function is used to determine the intent of a message.
    sentence_words = clean_up_sentence(sentence)  # tokenize the sentence into words.
    bag = [0] * len(wordss)  # create an empty bag. We need to fill it with 0 values. The length of the bag
    for s in sentence_words: 
        for i, word in enumerate(wordss): 	# iterate over the words in the word_features list.
            if word == s:
                bag[i] = 1  # set the value to 1 if the word is in the sentence or in the word_features.
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)  # create a bag of words. We need to convert the sentence into a bag of words. The function returns
    res = model.predict(np.array([bow.squeeze()]))[0]  # make a prediction. The function returns a vector of 0 and 1. The first
    ERROR_THRESHOLD = 0.25  # if the absolute difference between the predictions is greater than this value, then it is an incorrect classification.
    result = [[i, r] for i, r in enumerate(res) if abs(r - 1) > ERROR_THRESHOLD]  # filter the predictions.

    result.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in result:  # sort the results. The first element is the index, the second element is the classification.
        return_list.append({"Intent": classes[r[0]], "probability": str(r[1])})  # add the index and classification to the list.

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['Intent']  # Fix typo: 'intent' instead of 'intents'
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['intent'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('I\'m a bot created by Bassel')

while True:
    message=input()
    ints=predict_class(message)  # call the function to make a prediction.
    res=get_response(ints, intents)  # call the function to get a response. The function returns a string.
    print(res)  # print the response.
