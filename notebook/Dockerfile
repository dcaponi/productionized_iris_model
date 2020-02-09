FROM jupyter/scipy-notebook

RUN conda install --quiet --yes \
    'psycopg2'

COPY ./TrainModel.ipynb /app/TrainModel.ipynb

WORKDIR /app
