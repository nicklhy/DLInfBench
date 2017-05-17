NETWORK_LIST="alexnet resnet50 resnet101 resnet152"
GPU=0
BATCH_SIZE_LIST="1 2 4 8 16 32 64 128"
N_EPOCH=20
DLLIB_LIST="caffe2 caffe mxnet pytorch"

for DLLIB in ${DLLIB_LIST}
do
    for NETWORK in ${NETWORK_LIST}
    do
        for BATCH_SIZE in ${BATCH_SIZE_LIST}
        do
            python inference_${DLLIB}.py --network ${NETWORK} --batch-size ${BATCH_SIZE} --n-sample 1000 --n-epoch ${N_EPOCH} --gpu ${GPU}
        done
    done
done
