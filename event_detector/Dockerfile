FROM continuumio/miniconda:4.7.12

RUN apt-get update \
    # Install C++ tools
    && apt-get -y install \
        build-essential \
        cmake

RUN mkdir -p /code
RUN mkdir -p /data/processing
COPY event_detector /code/event_detector
WORKDIR /code/event_detector
RUN conda update -n base -c defaults conda
RUN conda env create --file environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "gtm", "/bin/bash", "-c"]

RUN python download_prerequisites.py

# ENTRYPOINT conda run -n gtm python event_detector.py

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gtm", "python", "event_detector.py"]