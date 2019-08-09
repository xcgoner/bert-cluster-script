# task: MNLI
TASK=MNLI
# typically 16, 32
# BS=16
# typically 3, 4
# EPOCH=4
# typically 1e-5, 2e-5, 5e-5
# LR=5e-5

ESP="small-bert-lamb"

GPU=0
SEED=0
NAME=test-run

# we need to manually use aws cli to copy the model parameter to s3
# TODO(xcong) automatically distribute the model parameter
BUCKET=s3://gluon-nlp-staging/batch/temp
PARAMFILE=bert-local-hiersgd-8x
LOG_PATH=~/log/$TASK-$ESP-$PARAMFILE

mkdir -p ~/log
mkdir -p $LOG_PATH

for EPOCH in 3 4
do
    for LR in 1e-5 2e-5 5e-5
    do
        for BS in 16 32
        do
            CONFIG=$TASK-$ESP-$BS-$EPOCH-$LR-$SEED-len-512
            FULL_NAME=$NAME-$CONFIG-$PARAMFILE
            LOG_FILE=$FULL_NAME/finetune_mnli_result.log

            aws s3 cp $BUCKET/$LOG_FILE $LOG_PATH/$FULL_NAME.log
            echo $CONFIG
            cat $LOG_PATH/$FULL_NAME.log | grep 'Best'
        done
    done
done

# cat $LOG_PATH/$NAME-$TASK-$ESP-* | grep 'Best'