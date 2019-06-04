#!/bin/bash

if [ $# -le 1 ]; then
    echo 'usage: ';
    echo 'bash npz.sh 40 py36';
    echo "number of arguments received=$#";
    exit -1;
fi

export DUPE=$1
export PY_VERSION=$2

if [ $PY_VERSION = 'py27' ]; then
  export PY='python27';
elif [ $PY_VERSION = 'py36' ]; then
  export PY='python36';
fi


for (( i=0; i<$DUPE; i++ ))
do
    $PY create_pretraining_data.py \
    --input_file /fsx/datasets/webtext/webtext-split/*.train,/fsx/datasets/enwiki/enwiki-feb-doc-split/*.train,/fsx/datasets/book-corpus/book-corpus-large-split/*.train \
    --output_dir /fsx/datasets/generated/web-book-enwiki/web-book-enwiki-sp-uncased/part-$i/ \
    --sentencepiece /fsx/datasets/vocab/webtext-book-wiki/webtext_book_wiki_uncased_bpe_30k_sampled_20M.model \
    --num_outputs 99999 --num_workers 72 --random_seed $i
done


