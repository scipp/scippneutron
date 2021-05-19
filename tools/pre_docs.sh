#!/bin/bash

set -ex

# Fetch data files from local dmsc web server.
# Note that the final '/' is important, if not wget ends up downloading all
# the other folders on the same level.
wget -nv -r -np -nH --cut-dirs=2 -R index.html* https://public.esss.dk/users/neil.vaytet/scippneutron/
mv scippneutron data;
# Create Mantid properties file so that we can give it the location of the data files
mkdir -p  $HOME/.mantid
echo -e "usagereports.enabled=0\ndatasearch.directories=$(pwd)/data" > $HOME/.mantid/Mantid.user.properties
