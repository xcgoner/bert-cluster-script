#!/bin/bash

if [ $# -ne 2 ]; then
    echo 'usage: ';
    echo 'bash nlp-install.sh mx-0525-raw-large-cased py27';
    echo "number of arguments received=$#";
    exit -1;
fi

export EXP=$1
export PY_VERSION=$2

if [ $PY_VERSION = 'py27' ]; then
  export PY='python27';
  export PIP='pip-2.7';
fi
if [ $PY_VERSION = 'py36' ]; then
  export PY='python36';
  export PIP='pip-3.6';
fi

cd /fsx/$EXP;
$PY setup.py develop --user;
