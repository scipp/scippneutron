#!/bin/bash

set -ex

# Fetch data files from local dmsc web server
mkdir -p  data
cd data
wget -nv -r -np -nH --cut-dirs=3 -R index.html* https://public.esss.dk/users/neil.vaytet/scippneutron
# Create Mantid properties file so that we can give it the location of the data files
mkdir -p  $HOME/.mantid
echo -e "usagereports.enabled=0\ndatasearch.directories=$(pwd)/data" > $HOME/.mantid/Mantid.user.properties
