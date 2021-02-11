FROM continuumio/miniconda:4.7.12

RUN apt-get update \
    # Install C++ tools
    && apt-get -y install \
        build-essential \
        cmake

RUN mkdir -p /code

COPY environment.yml /code
COPY download_prerequisites.py /code
WORKDIR /code
RUN conda update -n base -c defaults conda
RUN conda env create --file environment.yml

ENV PATH /opt/conda/envs/gtm/bin:$PATH
RUN /bin/bash -c "source activate gtm"

RUN python -u download_prerequisites.py