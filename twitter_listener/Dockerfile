FROM python:3.7
RUN mkdir -p /code
RUN mkdir -p /data/tweets
COPY twitter_listener /code/twitter_listener
WORKDIR /code/twitter_listener
# RUN echo $(pwd) 
# RUN echo $(ls) 
# RUN if test -f "/code/twitter_listener/.env"; then echo ".env exists."; else echo ".env not found."; fi;
RUN pip install -r requirements.txt
RUN python download_prerequisites.py

# ENTRYPOINT ["python"]
# CMD ["retrieve_data.py"]

# ENTRYPOINT ["python"]
# CMD ["forever.py", "retrieve_data.py"]

# ENTRYPOINT python ./forever.py ./retrieve_data.py
# ENTRYPOINT python code/twitter_listener/forever.py code/twitter_listener/retrieve_data.py
# CMD ["forever.py", "retrieve_data.py"]
# ENTRYPOINT ["python", "/code/twitter_listener/forever.py /code/twitter_listener/retrieve_data.py"]
# ENTRYPOINT ["python", "test.py"]

