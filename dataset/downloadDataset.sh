#!/bin/bash
sudo apt-get install unzip
sudo apt-get install wget
wget http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip
unzip "Activity recognition exp.zip"
mv Activity\ recognition\ exp/* .
rm "Activity recognition exp.zip"