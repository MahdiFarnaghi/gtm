import os
from dotenv import load_dotenv
import nltk
from gttm.nlp import VectorizerUtil_FastText

languages = os.getenv('LANGUAGES')
if languages != '':
    languages = str(languages).split(',')

vectorizer = VectorizerUtil_FastText()
[vectorizer.get_model(lang) for lang in languages]

nltk.download('wordnet')

