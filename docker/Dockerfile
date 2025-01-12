FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ENV DISPLAY=:0

RUN apt update && apt upgrade
RUN apt install -y python3
RUN apt install -y pip
RUN apt install -y python3-venv
RUN apt install -y git

WORKDIR /home/ubuntu

#RUN python3 -m venv .venv
#RUN source .venv/bin/activate
RUN python3 -m pip install --upgrade scipy
RUN python3 -m pip install --upgrade numpy
RUN python3 -m pip install --upgrade build
RUN python3 -m pip install --upgrade matplotlib
RUN python3 -m pip install --upgrade progressbar
RUN python3 -m pip install --upgrade pandas
RUN python3 -m pip install --upgrade PyQt5
RUN git clone https://github.com/pvangoor/pylie.git
RUN git clone https://github.com/aau-cns/ABC-EqF.git

WORKDIR /home/ubuntu/pylie

RUN pip install --upgrade pip
RUN python3 -m build && python3 -m pip install dist/pylie-0.1.0.tar.gz

WORKDIR /home/ubuntu
