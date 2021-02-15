FROM continuumio/miniconda:4.7.12

RUN apt-get update \
    # Install C++ tools
    && apt-get -y install \
        build-essential \
        cmake

RUN mkdir -p /data/nltk
RUN mkdir -p /code
RUN mkdir -p /data/processing
COPY event_detector /code/event_detector
WORKDIR /code/event_detector
RUN conda update -n base -c defaults conda
RUN conda env create --file environment.yml

ENV PATH /opt/conda/envs/gtm/bin:$PATH
RUN /bin/bash -c "source activate gtm"

RUN python download_prerequisites.py

ENTRYPOINT ["python"]
CMD ["-u", "event_detector.py"]
