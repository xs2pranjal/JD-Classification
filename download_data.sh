#!/bin/bash

curl -L http://www-nlp.stanford.edu/data/glove.6B.zip -o data/glove.6B.zip
unzip data/glove.6B.zip -d data
rm data/glove.6B.zip
rm data/glove.6B.50d.txt
rm data/glove.6B.200d.txt
rm data/glove.6B.300d.txt
