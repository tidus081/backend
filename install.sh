#!/bin/bash

# Statics
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  sudo apt update
  sudo apt-get install python3-pip python3-dev
fi

# Install requirement python libraries for service_base
pip3 install -r requirements.txt
cp -r ../utils/. ../flask_server/utils
cp -r ../utils/. ../model_server/utils
cp -r ../utils/. ../store_server/utils
cp -r ../utils/. ../vec_server/utils