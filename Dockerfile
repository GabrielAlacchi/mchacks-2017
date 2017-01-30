FROM node:6.9

EXPOSE 3001
ENV NODE_ENV 'production'

RUN apt-get update && apt-get install -y python \
                                         python-pip \
                                         build-essential

RUN pip install --upgrade pip
RUN pip install tensorflow \
                Pillow \
                numpy \
                scipy \
                simplejson

COPY *.js package.json /paint-me/
COPY app_api /paint-me/app_api/
COPY app_ui /paint-me/app_ui/
COPY public /paint-me/public/
COPY bin /paint-me/bin/
COPY neural-style /paint-me/neural-style

# COPY . /paint-me
WORKDIR /paint-me

VOLUME /paint-me/public/uploaded

RUN npm install
ENTRYPOINT npm start
