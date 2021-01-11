from pathlib import Path

import pandas as pd
import os
from wordcloud import WordCloud


def generate_wordcloud(series: pd.Series, stop_words=None, save_wc=False, max_words=20,
                       filepath_wc=""):
    wc = WordCloud(background_color="white", max_words=max_words,
                   stopwords=stop_words, contour_width=3, contour_color='steelblue', width=4000, height=2000)
    text = series.str.cat(sep=' ')

    # generate word cloud
    wordcloud = wc.generate(text)

    if save_wc and filepath_wc != "":
        if not Path(filepath_wc).parent.exists():
            os.makedirs(Path(filepath_wc).parent.absolute())
        wc.to_file(filepath_wc)
    return wordcloud


def generate_wordcloud_of_topic(topic, save_wc=False, filepath_wc=""):
    wc = WordCloud(background_color="white", max_words=50,
                   stopwords=None, contour_width=3, contour_color='steelblue', width=4000, height=2000)
    wordcloud = wc.generate_from_frequencies(topic)
    if save_wc and filepath_wc != "":
        if not Path(filepath_wc).parent.exists():
            os.makedirs(Path(filepath_wc).parent.absolute())
        wc.to_file(filepath_wc)
    return wordcloud
