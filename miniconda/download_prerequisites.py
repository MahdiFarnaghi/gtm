import nltk
import fasttext.util

languages = "en, pt"
if languages != '':
    languages = str(languages).split(',')

print('FASTTEXT LANGUAGE MODELS.')
for lang in languages:
    print(f"\tDownloading {lang}")
    fasttext.util.download_model(str.strip(lang), if_exists='ignore')    
print('LOADING FASTTEXT LANGUAGE MODELS was finished.')

print('LOADING NLTK')
nltk.download('wordnet', '/data/nltk')
print('LOADING NLTK was finished!')