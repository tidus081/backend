#!/bin/bash

# Statics
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  sudo apt-get install python-pip python-dev
fi

# Install requirement python libraries for service_base
pip3 install -r requirements.txt
