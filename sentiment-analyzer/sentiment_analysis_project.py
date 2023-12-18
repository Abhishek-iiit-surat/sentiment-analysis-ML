import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
# from wordcloud import WordCloud,STOPWORDS
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string,unicodedata
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet,stopwords
import re
import warnings
from googletrans import Translator
import joblib
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]
loaded_tfidf = joblib.load('tfidf_vectorizer.joblib')
import nltk
nltk.download('stopwords')

nltk.download('wordnet')

stop_words = stopwords.words('english')
new_stopwords = ["one","film","mario","la","blah","saturday","monday","sunday","morning","evening","friday","would","shall","could","might"]
stop_words.extend(new_stopwords)
stop_words.remove("not")
stop_words.remove("no")
stop_words.remove("No")
stop_words=set(stop_words)

#Removing special character
def remove_special(content):
       return re.sub(r'[^a-zA-Z0-9\s]', '', content)



# remove url from the data
def remove_url(content):
      return re.sub(r'http\S+','',content)

# removing stopwords from text

def remove_stopwords(content):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)


def translate_hindi_words_to_english(sentence):
    translator = Translator()
    words = sentence.split()
    hindi_unicode_ranges = [
        (0x0900, 0x097F),
    ]

    translated_words = []

    for word in words:
        # Check in the Hindi Unicode ranges
        

            
        if any(start <= ord(char) <= end for start, end in hindi_unicode_ranges for char in word):
            # If the word is in Hindi, translate it
            translation = translator.translate(word, src='hi', dest='en')
            translated_words.append(translation.text)
        else:
            # keep it as is
            if word=='gud'or word=='gd' or word=='guud':
                translated_words.append("good")
            elif word=='nt'or word=='nut':
                translated_words.append("not")
            else:
                translated_words.append(word)
                
    translated_sentence = ' '.join(translated_words)

    return translated_sentence


def spell_correct_sentence(sentence):
    spell = SpellChecker()
    words = sentence.split()
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]
    corrected_sentence = ' '.join(corrected_words)
    return corrected_sentence


# Expanding of english contractions
def contraction_expansion(content):
  content=re.sub(r"wouldn\'t","would not",content)
  content=re.sub(r"won\'t","will not",content)
  content=re.sub(r"can\'t","can not",content)
  content=re.sub(r"don\'t","do not",content)
  content=re.sub(r"shouldn\'t","should not",content)
  content=re.sub(r"needn\'t","need not",content)
  content=re.sub(r"weren\'t","were not",content)
  content=re.sub(r"mightn\'t","might not",content)
  content=re.sub(r"didn\'t","would not",content)
  content=re.sub(r"n\'t"," not",content)
  content=re.sub(r"r\'re"," are",content)
  content=re.sub(r"\'s"," is",content)
  content=re.sub(r"\'d"," would",content)
  content=re.sub(r"\'ll"," will",content)
  content=re.sub(r"\'t"," not",content)
  content=re.sub(r"\'ve"," have",content)
  content=re.sub(r"\'m"," am",content)
  return content

# data preprocessing

def data_cleaning(content):
    # Assuming these functions are correctly implemented
    content=translate_hindi_words_to_english(content)
    print(content)
    content=spell_correct_sentence(content)
    content = contraction_expansion(content)
    content = remove_special(content)
    content = remove_url(content)
    content = remove_stopwords(content)
    print(content)
    tfidf_matrix = loaded_tfidf.transform([content])
    return tfidf_matrix


# Example usage
