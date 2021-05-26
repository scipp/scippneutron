#!/bin/bash

set -ex

if test -z "${DOCS_PREFIX}"
then
  DOCS_PREFIX="$(pwd)"
fi

# Fetch data files from local dmsc web server.
# Note that the final '/' is important, if not wget ends up downloading all
# the other folders on the same level.
wget -c -N -nv -r -np -nH --cut-dirs=2 -R index.html* https://public.esss.dk/groups/scipp/scippneutron/
mv scippneutron ${DOCS_PREFIX}/data
# Create Mantid properties file so that we can give it the location of the data files
mkdir -p  ${HOME}/.mantid
PROPS_FILE=${HOME}/.mantid/Mantid.user.properties
# Make a backup of the user file in case it already exists
if test -f ${PROPS_FILE}
then
  cp ${PROPS_FILE} ${PROPS_FILE}.backup
  REVERT_FILE=true
else
  REVERT_FILE=false
fi
# Make custom properties file so files can be found during documentation build
echo -e "usagereports.enabled=0\ndatasearch.directories=${DOCS_PREFIX}/data" > ${HOME}/.mantid/Mantid.user.properties
# If a backup was made, revert to original file
if [ "$REVERT_FILE" = true ]
then
  cp ${PROPS_FILE}.backup ${PROPS_FILE}
fi
