FROM ubuntu:22.04

RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install ffmpeg libsm6 libxext6 qt6-base-dev libxcb-cursor0 libxcb-xinerama0 nodejs -y
RUN apt install wget -y


RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b
ENV PATH="/root/anaconda3/bin:$PATH"

RUN conda install -c conda-forge nodejs
RUN conda install -c conda-forge/label/gcc7 nodejs
RUN conda install -c conda-forge/label/cf201901 nodejs
RUN conda install -c conda-forge/label/cf202003 nodejs

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN pip install -U jupyterlab
RUN pip install -U pyxai-experimental
RUN pip install -U ipython
RUN pip install -U widgetsnbextension
RUN pip install -U nodejs
RUN pip install -U ipywidgets
RUN pip install -U jupyter-lsp

ADD . /data/

WORKDIR /data

VOLUME /data

EXPOSE 8888

CMD jupyter lab --allow-root --no-browser --ip=0.0.0.0