if [ $# -le 13 ]; then
    echo 'usage: ';
    echo 'bash run.sh base  book-wiki-cased       ~/hosts 64 bertadam npz 8192 2 0.0002828 0.08 125000  py27 experiment-name';
    echo 'bash run.sh large book-wiki-uncased     ~/hosts 64 bertadam raw 2048 1 0.0001    0.01 1000000 py27 experiment-name';
    echo 'bash run.sh base  web-book-wiki-uncased ~/hosts 64 bertadam raw 8192 1 0.0002    0.04 750000  py36 experiment-name /fsx/datasets/vocab/webtext-book-wiki/webtext_book_wiki_uncased_bpe_30k_sampled_20M.model';
    echo "number of arguments received=$#";
    exit -1;
fi

export MODEL=$1;
export DATASET=$2;
export HOSTS=$3
export NGPU=$4
export OPTIMIZER=$5
export FORMAT=$6
export BS=$7
export ACCUMULATE=$8
export LOCALSGD=$9
export LR=${10}
export WARMUP=${11}
export NSTEP="${12}"
export PY_VERSION="${13}"
export EXP="${14}"
export SPM="${15}"

if [ $PY_VERSION = 'py27' ]; then
  export PY='python27';
elif [ $PY_VERSION = 'py36' ]; then
  export PY='python36';
elif [ $PY_VERSION = 'py3' ]; then
  export PY='python3';
fi
echo "==================== arguments ==================== ";
echo "model=$MODEL, dataset=$DATASET, hosts=$HOSTS, num_gpus=$NGPU, format=$FORMAT";
echo "optimizer=$OPTIMIZER, batch_size=$BS, accumulate=$ACCUMULATE, learning_rate=$LR, warmup=$WARMUP"
echo "num_steps=$NSTEP, python version=$PY_VERSION, sentencepiece=$SPM";

DATA_DIR='/fsx/datasets/generated'
RAW_DATA_DIR='/fsx/datasets/'

BOOK_CASED_DIR="$DATA_DIR/generated-book-large-cased-py3-512"
WIKI_CASED_DIR="$DATA_DIR/generated-enwiki-feb-cased-512"

BOOK_UNCASED_DIR="$DATA_DIR/generated-book-large-uncased-py3-512"
WIKI_UNCASED_DIR="$DATA_DIR/generated-enwiki-feb-uncased-py3-512"

BOOK_WIKI_CASED="$BOOK_CASED_DIR/train/part-*/*.npz,$WIKI_CASED_DIR/train/part-*/*.npz";
BOOK_WIKI_CASED_EVAL="$BOOK_CASED_DIR/dev/part-*/*.npz,$WIKI_CASED_DIR/dev/part-*/*.npz";

if [ "$MODEL" = 'small' ]; then
  BOOK_WIKI_UNCASED="$BOOK_UNCASED_DIR/train/part-2*/*.npz,$BOOK_UNCASED_DIR/train/part-3*/*.npz,$WIKI_UNCASED_DIR/train/part-2*/*.npz,$WIKI_UNCASED_DIR/train/part-3*/*.npz";
else
  BOOK_WIKI_UNCASED="$BOOK_UNCASED_DIR/train/part-*/*.npz,$WIKI_UNCASED_DIR/train/part-*/*.npz";
fi


BOOK_WIKI_UNCASED_EVAL="$BOOK_UNCASED_DIR/dev/part-*/*.npz,$WIKI_UNCASED_DIR/dev/part-*/*.npz";
# BOOK_WIKI_UNCASED="/fsx/datasets/generated/xc*/train/part*/*.npz";
# BOOK_WIKI_UNCASED_EVAL="/fsx/datasets/generated/xc*/dev/part*/*.npz";

WEB_BOOK_WIKI_UNCASED_EVAL="$DATA_DIR/dev/web-book-wiki-sp/*.npz";

if [ "$FORMAT" = 'raw' ]; then
  DATA_DIR='/fsx/datasets'
  BOOK_WIKI_CASED="$RAW_DATA_DIR/enwiki/enwiki-feb-doc-split/*.train,$RAW_DATA_DIR/book-corpus/book-corpus-large-split/*.train";
  BOOK_WIKI_UNCASED="$RAW_DATA_DIR/enwiki/enwiki-feb-doc-split/*.train,$RAW_DATA_DIR/book-corpus/book-corpus-large-split/*.train";
  WEB_BOOK_WIKI_UNCASED="$BOOK_WIKI_UNCASED,$RAW_DATA_DIR/webtext/webtext-split/*.train";
fi

DATASET_NAME="book_corpus_wiki_en_uncased"
if [ "$DATASET" = 'book-wiki-cased' ]; then
  DATA_TRAIN="$BOOK_WIKI_CASED"
  DATA_EVAL="$BOOK_WIKI_CASED_EVAL"
  DATASET_NAME="book_corpus_wiki_en_cased"
elif [ "$DATASET" = 'book-wiki-uncased' ]; then
  DATA_TRAIN="$BOOK_WIKI_UNCASED"
  DATA_EVAL="$BOOK_WIKI_UNCASED_EVAL"
elif [ "$DATASET" = 'web-book-wiki-uncased' ]; then
  DATA_TRAIN="$WEB_BOOK_WIKI_UNCASED"
  DATA_EVAL="$WEB_BOOK_WIKI_UNCASED_EVAL"
fi

if [ "$MODEL" = 'small' ]; then
  BERT_MODEL="bert_6_768_12"
elif [ "$MODEL" = 'base' ]; then
  BERT_MODEL="bert_12_768_12"
elif [ "$MODEL" = 'large' ]; then
  BERT_MODEL="bert_24_1024_16"
fi

HVD_PREFIX=" --hostfile hosts -mca pml ob1 \
             -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket \
             -x NCCL_MIN_NRINGS=16 -x NCCL_DEBUG=WARNING -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
             --tag-output ";

EXTRA_FLAG=""
if [ "$FORMAT" = 'raw' ]; then
  EXTRA_FLAG=" --raw --max_seq_length 512 --short_seq_prob 0.1 --masked_lm_prob 0.15 --max_predictions_per_seq 80 "
  if [ "$SPM" != '' ]; then
    EXTRA_FLAG="$EXTRA_FLAG --sentencepiece $SPM"
  fi
fi

echo "killing running python processes ... ";
hudl -h $HOSTS "sudo pkill -9 $PY";

echo "creating host file ... ";

rm -f hosts;
while read line; do
  echo "$line slots=8" >> hosts
done < $HOSTS

CKPT_DIR="/fsx/experiment/$EXP"
sudo mkdir -p $CKPT_DIR
sudo chmod a+w $CKPT_DIR

HPARAMS=" --batch_size $BS --accumulate $ACCUMULATE --lr $LR --local_sgd $LOCALSGD"

echo -e "\n====================== command: ====================== "
CMD=" $PY run_pretraining_local_hvd.py $HPARAMS --data $DATA_TRAIN \
      --data_eval $DATA_EVAL --optimizer $OPTIMIZER --warmup_ratio $WARMUP --num_steps $NSTEP --log_interval=250 --start_step=00000 \
      --ckpt_dir $CKPT_DIR/ckpt --ckpt_interval 25000 --num_buckets 10 --dataset_name $DATASET_NAME \
      --dtype float16 --use_avg_len --model $BERT_MODEL $EXTRA_FLAG "
echo -e "$CMD \n =====================================================\n"

echo -e "\n=================== mpirun command: ================== "
MPICMD="mpirun -np $NGPU $HVD_PREFIX -x MXNET_SAFE_ACCUMULATION=1 $CMD"
echo -e "$MPICMD \n =====================================================\n"

mpirun -np $NGPU --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no' \
       $HVD_PREFIX -x MXNET_SAFE_ACCUMULATION=1 $CMD 2>&1 | tee $CKPT_DIR/result.log

#       --sentencepiece '/fsx/datasets/vocab/book-large-enwiki-cased/bpe.model' 
#mpirun -np 64 --hostfile hosts --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no" -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket -x NCCL_MIN_NRINGS=16 -x NCCL_DEBUG=INFO -x MXNET_SAFE_ACCUMULATION=1 --tag-output python run_pretraining_hvd.py --batch_size 2048 --accumulate 1 --lr 0.0001 --data "$DATA" --data_eval "$DATAEVAL" --warmup_ratio 0.01 --num_steps 1000000 --log_interval=25 --ckpt_dir 's3://bert-pretraining/exp/0523-cased-large/ckpt/' --ckpt_interval 25000 --num_buckets 10 --dtype float16 --use_avg_len --dataset_name book_corpus_wiki_en_uncased --model bert_24_1024_16 #2>&1 | tee -a ~/0523-cased-large.log2

#MXNET_SAFE_ACCUMULATION=1 PYTHONPATH=~/gluon-nlp/src/ python run_pretraining_hvd.py --batch_size 16 --accumulate 1 --lr 1e-4 --data "./part-0160.train, ./part-0160.train.trunc" --data_eval "./part-0/part-000.npz" --warmup_ratio 0.01 --num_steps 1000000 --log_interval=5 --ckpt_dir './ckpt-test' --ckpt_interval 25 --num_buckets 10 --dtype float16 --dummy_data_len 0 --raw --max_seq_length 512 --short_seq_prob 0.1  --masked_lm_prob 0.15 --max_predictions_per_seq 80 --num_data_workers 8 --sentencepiece 'vocab/book-large-enwiki-cased/bpe.mode

#mpirun -np 64 --hostfile hosts --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no" -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO python run_pretraining_hvd.py --batch_size 12288 --accumulate 1 --lr 0.000225 --data "$DATA" --warmup_ratio 0.06 --num_steps 166666 --log_interval=250 --ckpt_dir './ckpt-large' --ckpt_interval 20000 --num_buckets 10 --dtype float16 --use_avg_len 2>&1 | tee -a ~/6x.log
# DATA='/fsx/datasets/generated/*-cased-*/train/part-*/*.npz'
# DATAEVAL='/fsx/datasets/generated/generated-enwiki-feb-cased-512/dev/part-0/*.npz,/fsx/datasets/generated/generated-book-large-cased-py3-512/dev/part-0/*.npz'
#python create_pretraining_data.py --input_file '/home/ec2-user/dataset/*split/*.dev' --output_dir /fsx/datasets/generated/dev/book-wiki-cased-sp/ --dataset_name book_corpus_wiki_en_uncased --max_seq_length 512 --max_predictions_per_seq 80 --dupe_factor 1 --masked_lm_prob 0.15 --short_seq_prob 0.1 --num_workers 1 --sentencepiece '/fsx/datasets/vocab/book-large-enwiki-cased/bpe.model' --verbose --cased
