FROM ubuntu:22.04

RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y

RUN pip install jupyter
RUN pip install pyxai

ADD . /data/

WORKDIR /data

VOLUME /data

EXPOSE 8888

CMD jupyter notebook --allow-root --no-browser --ip=0.0.0.0