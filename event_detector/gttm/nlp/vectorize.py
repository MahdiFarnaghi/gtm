from gensim.models import KeyedVectors, FastText
import numpy as np
import os
import sys
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datetime import datetime, timedelta
import requests
import tempfile
from pathlib import Path
import traceback
from tqdm import tqdm
import fasttext
import fasttext.util

from dotenv import load_dotenv

load_dotenv()

FASTTEXT_FOLDER_PATH = os.getenv('FASTTEXT_FOLDER_PATH')

# https://fasttext.cc/

# TODO: If gensim is not used in other parts of the project, it should be removed from the requirements.yml


class VectorizerUtil_FastText:

    def __init__(self):
        self.models = {}

    def get_model_file_name(self, lang):
        return F"cc.{lang}.300.bin"

    # def get_model_url(self, lang):
    #     return F"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz"

    def vectorize(self, sentences, lang):
        model = self.get_model(lang)
        return np.array([model.get_sentence_vector(sent) for sent in sentences])

    def get_model(self, lang):
        if not lang in self.models.keys():
            fasttext.util.download_model(lang, if_exists='ignore')
            self.models[lang] = fasttext.load_model(
                self.get_model_file_name(lang))
        return self.models[lang]
