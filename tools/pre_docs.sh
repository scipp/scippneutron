#!/bin/bash

set -ex

python data/fetch_neutron_data.py
mkdir -p  $HOME/.mantid
echo -e "usagereports.enabled=0\ndatasearch.directories=$(pwd)/data" > $HOME/.mantid/Mantid.user.properties
