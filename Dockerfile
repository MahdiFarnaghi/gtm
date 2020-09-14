FROM python:3.7.9-apline3.12
RUN mkdir -p /code
RUN mkdir -p /data/tweets
COPY twitter_listener /code/
WORKDIR /code/twitter_listener
RUN pip install -r requirements.txt