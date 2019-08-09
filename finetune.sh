# task: MNLI
TASK=MNLI
# typically 16, 32
# BS=16
# typically 3, 4
EPOCH=4
# typically 1e-5, 2e-5, 5e-5
# LR=5e-5

ESP="small-bert-lamb"

GPU=0
SEED=0
NAME=test-run

# we need to manually use aws cli to copy the model parameter to s3
# TODO(xcong) automatically distribute the model parameter
BUCKET=s3://gluon-nlp-staging/batch/xcong
PARAMFILE=small-bert-lamb-model.params
CKPT=$BUCKET/$PARAMFILE
SCRIPT=finetune_classifier.py

for LR in 1e-5 2e-5 5e-5
do
    for BS in 16 32
    do
        CONFIG=$TASK-$ESP-$BS-$EPOCH-$LR-$SEED-len-512
        FULL_NAME=$NAME-$CONFIG
        # for logging, in gluon-nlp/ci/batch/docker/gluon_nlp_job.sh, change /bin/bash -c "$COMMAND" to /bin/bash -c "$COMMAND" 2>&1 result.log
        CMD="python $SCRIPT --task_name $TASK:q!
         --log_interval 100 --batch_size $BS --epochs $EPOCH --gpu $GPU --lr $LR --seed $SEED --bert_model bert_6_768_12 --pretrained_bert_parameters $CKPT --max_len 512"
        python submit-job.py --source-ref round-padding --remote https://github.com/xcgoner/gluon-nlp --name $FULL_NAME --save-path batch/temp/$FULL_NAME --conda-env gpu/py3-master --command "$CMD"
    done
done