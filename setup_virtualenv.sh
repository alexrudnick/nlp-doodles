#!/bin/bash

virtualenv -p /usr/bin/python3 venv
. venv/bin/activate

pip install numpy
pip install scipy
pip install scikit-learn
pip install nltk
