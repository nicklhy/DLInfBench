NETWORK_LIST="alexnet vgg16 vgg19 inception-bn inception-v3 resnet50 resnet101 resnet152"
GPU=0
BATCH_SIZE_LIST="1 2 4 8 16 32 64 128"
N_EPOCH=20
DLLIB_LIST="caffe caffe2 mxnet pytorch"

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

for NETWORK in ${NETWORK_LIST}
do
    python plot_speed.py --network ${NETWORK}
done
