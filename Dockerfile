FROM node:6.9

# Expose http and https ports
EXPOSE 80
EXPOSE 443

# Non production ports
EXPOSE 8080
EXPOSE 8443

# Default node env to production
ENV NODE_ENV 'production'

# Will use these files if they're mounted as a volume for https
ENV CRT_FILE /etc/ssl/certs/paint-me_me.crt
ENV KEY_FILE /etc/ssl/private/paint-me.key
ENV CA_BUNDLE /etc/ssl/certs/paint-me_me.ca-bundle
ENV AUTH_FILE /etc/ssl/firebase_auth.txt

# Install dependencies
RUN apt-get update && apt-get install -y python \
                                         python-pip \
                                         build-essential

RUN pip install --upgrade pip
RUN pip install tensorflow \
                Pillow \
                numpy \
                scipy \
                simplejson

# Copy the project files.
COPY *.js package.json /paint-me/
COPY app_api /paint-me/app_api/
COPY app_ui /paint-me/app_ui/
COPY public /paint-me/public/
COPY bin /paint-me/bin/
COPY neural-style /paint-me/neural-style

# Set the working directory
WORKDIR /paint-me

VOLUME /paint-me/public/uploaded/

RUN npm install
ENTRYPOINT npm start
