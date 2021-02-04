import numpy as np
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datetime import datetime, timedelta
import requests
import tempfile
from pathlib import Path
import traceback
import fasttext
import fasttext.util

from dotenv import load_dotenv

load_dotenv()

FASTTEXT_FOLDER_PATH = os.getenv('FASTTEXT_FOLDER_PATH')

# https://fasttext.cc/


class VectorizerUtil_FastText:

    def __init__(self):
        self.models = {}

    def get_model_file_name(self, lang):
        return F"cc.{lang}.300.bin"

    # def get_model_url(self, lang):
    #     return F"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz"

    def vectorize(self, sentences, lang) -> np.array:
        model = self.get_model(lang)
        return np.array([model.get_sentence_vector(sent) for sent in sentences])

    def get_model(self, lang):
        if not lang in self.models.keys():
            fasttext.util.download_model(lang, if_exists='ignore')
            self.models[lang] = fasttext.load_model(
                self.get_model_file_name(lang))
        return self.models[lang]
