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

from dotenv import load_dotenv

load_dotenv()

FASTTEXT_FOLDER_PATH = os.getenv('FASTTEXT_FOLDER_PATH')

# https://fasttext.cc/


# TODO: Load Multi Linual Models. Check: https://fasttext.cc/docs/en/crawl-vectors.html

class VectorizerUtil_FastText:

    def __init__(self):
        self.models = {}

    def get_model_path(self, lang):
        base_path = 'data/processing/fasttext'
        if FASTTEXT_FOLDER_PATH:
            base_path = FASTTEXT_FOLDER_PATH
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, F"cc.{lang}.300.bin.gz") 
    
    def get_model_url(self, lang):
        return F"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz"

    def download_pretrained_model(self, lang):
        download_url = self.get_model_url(lang)
        model_path = self.get_model_path(lang)
        if os.path.exists(model_path):
            return True
        
        num_attempts = 10
        for _ in range(0, num_attempts):
            try:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(download_url, allow_redirects=True)
                open(model_path, 'wb').write(r.content)
                return True
            except: 
                traceback.print_exc(file=sys.stdout)
                pass
        
        raise Exception(F'Could not download pretrained models from {download_url}')
    
    def vectorize(self, sentences, lang):
        model = self.get_model(lang)
        tokenized_sentences = [[word for word in sent.split(' ')] for sent in
                               sentences]

        return [np.mean(model.wv[tokenized_sentence], axis=0) for tokenized_sentence in
                    tokenized_sentences]        
    
    def get_model(self, lang):
        if not lang in self.models.keys():
            if self.download_pretrained_model(lang):
                filepath = self.get_model_path(lang)
                if os.path.isfile(os.path.realpath(filepath)):
                    try:                            
                        print("\tLoaing FastText {} ({})".format(
                            lang,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        s_time = datetime.now()
                        self.models[lang] = FastText.load_fasttext_format(filepath)
                        print('\tInit FastText')
                        self.models[lang].init_sims(replace=True)                    
                        print("\tInitialization finished. ({})".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        dur = datetime.now() - s_time
                        print("\tIt took {} seconds".format(dur.seconds))    
                    except:
                        traceback.print_exc(file=sys.stdout)                    
                else:
                    raise FileNotFoundError('{} was not found.'.format(filepath))
        return self.models[lang]