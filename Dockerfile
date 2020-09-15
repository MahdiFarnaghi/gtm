FROM python:3.7
RUN mkdir -p /code
RUN mkdir -p /data/tweets
COPY twitter_listener /code/twitter_listener
COPY ["./twitter_listener/.env", "/code/twitter_listener"]
WORKDIR /code/twitter_listener
RUN echo $(pwd) 
RUN echo $(ls) 
RUN if test -f "/code/twitter_listener/.env"; then echo ".env exists."; else echo ".env not found."; fi;
RUN pip install -r requirements.txt
