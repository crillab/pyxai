FROM python:3.8


RUN pip install jupyter
RUN pip install pyxai


ADD . /data/

WORKDIR /data

VOLUME /data

EXPOSE 8888

CMD jupyter notebook --allow-root --no-browser --ip=0.0.0.0