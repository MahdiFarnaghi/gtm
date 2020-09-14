FROM python:3.7
RUN mkdir -p /code
RUN mkdir -p /data/tweets
COPY twitter_listener /code/twitter_listener
WORKDIR /code/twitter_listener
RUN echo $(pwd) 
RUN echo $(ls) 

RUN pip install -r requirements.txt
