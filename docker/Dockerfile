FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y install git vim

RUN pip install pymongo

ENV smac_ver 1
ENV SC2PATH /source/3rdparty/StarCraftII

ARG UID
RUN useradd -u $UID --create-home benlis
USER benlis
WORKDIR /home/benlis/pymarl

RUN echo 'set editing-mode vi' >> /home/benlis/.inputrc
RUN echo 'set keymap vi' >> /home/benlis/.inputrc

ADD install_dependencies.sh /source/
RUN bash -c /source/install_dependencies.sh

WORKDIR /source/
