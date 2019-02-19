#!/bin/bash

# Statics
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  sudo apt update
  sudo apt-get install python3-pip python3-dev
fi

# Install requirement python libraries for service_base
pip3 install pandas numpy sklearn gensim pillow scipy DateTime 
pip3 instal nltk grpcio grpcio-tools
cp -r utils/* flask_server/
cp -r utils/* model_server/
cp -r utils/* store_server/
cp -r utils/* vec_server/