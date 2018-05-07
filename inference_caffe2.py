import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
import caffe2
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, models
from caffe2.python.models import resnet
from caffe2.python.cnn import CNNModelHelper
import numpy as np
import argparse
import time

DLLIB = 'caffe2'

def create_alexnet(model,
                   data,
                   num_labels=1000,
                   label=None,
                   no_loss=False):
    model.Conv(data, 'conv1', 3, 96, weight_init=("MSRAFill", {}), kernel=11, stride=4)
    model.Relu('conv1', 'conv1')
    model.LRN('conv1', 'norm1', size=5, alpha=0.0001, beta=0.75)
    model.MaxPool('norm1', 'pool1', kernel=3, stride=2)

    model.Conv('pool1', 'conv2', 96, 256, weight_init=("MSRAFill", {}), kernel=5, group=2, pad=2)
    model.Relu('conv2', 'conv2')
    model.LRN('conv2', 'norm2', size=5, alpha=0.0001, beta=0.75)
    model.MaxPool('norm2', 'pool2', kernel=3, stride=2)

    model.Conv('pool2', 'conv3', 256, 384, weight_init=("MSRAFill", {}), kernel=3, pad=1)
    model.Relu('conv3', 'conv3')

    model.Conv('conv3', 'conv4', 384, 384, weight_init=("MSRAFill", {}), kernel=3, pad=1, group=2)
    model.Relu('conv4', 'conv4')

    model.Conv('conv4', 'conv5', 384, 256, weight_init=("MSRAFill", {}), kernel=3, pad=1, group=2)
    model.Relu('conv5', 'conv5')
    model.MaxPool('conv5', 'pool5', kernel=3, stride=2)

    #  shape of pool5 is (batch_size, 256, 6, 6)
    model.FC('pool5', 'fc6', 256*6*6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'fc6', dropout_ratio=0.5)

    model.FC('fc6', 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'fc7', dropout_ratio=0.5)

    last_out = model.FC('fc7', 'fc8', 4096, num_labels)

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss([last_out, label], ["softmax", "loss"])
        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax(last_out, "softmax")

def create_resnet(
    model,
    data,
    num_layers=50,
    num_input_channels=3,
    num_labels=1000,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    conv1_kernel=7,
    conv1_stride=2,
    final_avg_kernel=7,
):
    # conv1 + maxpool
    model.Conv(data, 'conv1', num_input_channels, 64, weight_init=("MSRAFill", {}),
               kernel=conv1_kernel, stride=conv1_stride, pad=3, no_bias=no_bias)

    model.SpatialBN('conv1', 'conv1_spatbn_relu', 64,
                    epsilon=1e-3, momentum=0.1, is_test=is_test)
    model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')
    model.MaxPool('conv1_spatbn_relu', 'pool1', kernel=3, stride=2)

    # Residual blocks...
    builder = resnet.ResNetBuilder(model, 'pool1', no_bias=no_bias,
                                   is_test=is_test, spatial_bn_mom=0.1)

    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 50:
        units = [3, 4, 6, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))

    if num_layers >= 50:
        filter_list = [64, 256, 512, 1024, 2048]
    else:
        filter_list = [64, 64, 128, 256, 512]

    for i in range(len(units)):
        builder.add_bottleneck(filter_list[i], filter_list[i+1]/4, filter_list[i+1], down_sampling=(i!=0))
        for j in range(units[i]-1):
            builder.add_bottleneck(filter_list[i+1], filter_list[i+1]/4, filter_list[i+1])

    # Final layers
    final_avg = model.AveragePool(
        builder.prev_blob, 'final_avg', kernel=final_avg_kernel, stride=1,
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = model.FC(final_avg, 'last_out_L{}'.format(num_labels),
                        2048, num_labels)

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax(last_out, "softmax")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test CNN inference speed')
    parser.add_argument('--network', type=str, default='resnet50',
                        choices=['alexnet', 'resnet50', 'resnet101', 'resnet152'],
                        help='network name')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16'],
                        help='data type')
    parser.add_argument('--params', type=str, help='model parameters')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--im-size', type=int, help='image size')
    parser.add_argument('--n-sample', type=int, default=1000, help='number of samples')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--n-epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--warm-up-num', type=int, default=10, help='number of iterations for warming up')
    parser.add_argument('--verbose', type=lambda x: x.lower() in ('yes', 'true', 't', '1'), default=True,
                        help='verbose information')
    args = parser.parse_args()

    print('===================== benchmark for %s %s =====================' % (DLLIB, args.network))
    print('n_sample=%d, batch size=%d, num epoch=%d' %  (args.n_sample, args.batch_size, args.n_epoch))

    im_size = 224
    if args.im_size is not None:
        im_size = args.im_size
    elif args.network == 'alexnet':
        im_size = 227
    elif args.network == 'inception-v3':
        im_size = 299

    #  loading model
    t1 = time.time()
    device_opts = caffe2_pb2.DeviceOption()
    device_opts.device_type = caffe2_pb2.CUDA
    device_opts.cuda_gpu_id = args.gpu

    if args.network.lower().startswith('vgg'):
        net_path = os.path.join(ROOT_DIR, 'models', 'caffe', args.network+'_pred_net.pb')
        if not os.path.exists(net_path):
            print('%s doesn\'t exists!' % args.network)
            sys.exit(1)

        if args.params is None:
            print('Currently, we do not support building %s with random values, you have to choose a pre-trained weights file.' % args.network)
            sys.exit(1)
        elif not os.path.exists(args.params):
            print('%s does not exists!' % args.params)
            sys.exit(1)

        init_def = caffe2_pb2.NetDef()
        with open(args.params, 'r') as f:
            init_def.ParseFromString(f.read())
            init_def.device_option.CopyFrom(device_opts)
            workspace.RunNetOnce(init_def)

        net_def = caffe2_pb2.NetDef()
        with open(net_path, 'r') as f:
            net_def.ParseFromString(f.read())
            net_def.device_option.CopyFrom(device_opts)
            for op in net_def.op:
                op.engine = 'CUDNN'
            workspace.CreateNet(net_def)
    elif args.network.startswith('resnet') or args.network == 'alexnet':
        if args.network.startswith('resnet'):
            model = CNNModelHelper(
                order='NCHW',
                name=args.network,
                use_cudnn=True,
                cudnn_exhaustive_search=True
            )
            num_layers = int(args.network[6:])
            softmax = create_resnet(model, 'data', num_layers=num_layers, num_input_channels=3, num_labels=1000, label=None, no_bias=True, no_loss=True)
        elif args.network == 'alexnet':
            model = CNNModelHelper(
                order='NCHW',
                name=args.network,
                #  use_cudnn=True,
                #  cudnn_exhaustive_search=True
                use_cudnn=False
            )
            print('WARNING: This alexnet implementation can not use CUDNN for some LRN layer related reason. If you can solve this problem, a PR is welcomed.')
            softmax = create_alexnet(model, 'data', num_labels=1000, label=None, no_loss=True)
        else:
            raise NotImplementedError
        net_def = model.net.Proto()
        net_def.device_option.CopyFrom(device_opts)
        model.param_init_net.RunAllOnGPU(gpu_id=args.gpu, use_cudnn=True)

        workspace.CreateBlob('data')
        #  workspace.CreateBlob('label')
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(net_def)
    else:
        raise NotImplementedError('%s is not supported yet' % args.network)

    t2 = time.time()
    print('Finish loading model in %.4fs' % (t2-t1))

    t1 = time.time()
    data_list = [np.random.uniform(-1, 1, (args.batch_size, 3, im_size, im_size)).astype(np.float32) for i in range(int(np.ceil(1.0*args.n_sample/args.batch_size)))]
    t2 = time.time()
    print('Generate %d random images in %.4fs!' % (args.n_sample, t2-t1))

    # warm-up to burn your GPU to working temperature (usually around 80C) to get stable numbers
    k = 0
    while k < args.warm_up_num:
        for batch in data_list:
            if k >= args.warm_up_num:
                break
            k += 1
            workspace.FeedBlob('data', batch, device_opts)
            workspace.RunNet(net_def.name, 1)
    print('Warm-up for %d iterations' % args.warm_up_num)

    t_list = []
    t_start = time.time()
    for i in range(args.n_epoch):
        t1 = time.time()

        for j, batch in enumerate(data_list):
            workspace.FeedBlob('data', batch, device_opts)
            workspace.RunNet(net_def.name, 1)

        t2 = time.time()
        t_list.append(t2-t1)
        if args.verbose:
            print('Epoch %d, finish %d images in %.4fs, speed = %.4f image/s' % (i, args.n_sample, t2-t1, args.n_sample/(t2-t1)))

    t_end = time.time()

    t_list = np.array(t_list)
    if args.n_epoch > 2:
        argmax = t_list.argmax()
        argmin = t_list.argmin()
        t_list[argmax] = 0
        t_list[argmin] = 0
        t_avg = np.sum(t_list)/(args.n_epoch-2)
    else:
        t_avg = np.sum(t_list)/args.n_epoch
    print('Finish %d images for %d times in %.4fs, speed = %.4f image/s (%.4f ms/image)' % (args.n_sample, args.n_epoch, t_end-t_start, args.n_sample/t_avg, t_avg*1000.0/args.n_sample))

    print('===================== benchmark finished =====================')

    from utils import get_gpu_memory
    gpu_mem = get_gpu_memory()

    #  save results
    res_dir = 'cache/results'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    res_file_path = os.path.join(res_dir, '%s_%s_%s_%d.txt' % (DLLIB,
                                                               args.network,
                                                               args.dtype,
                                                               args.batch_size))
    with open(res_file_path, 'w') as fd:
        fd.write('%s %s %s %d %f %d\n' % (DLLIB, args.network, args.dtype,
                                     args.batch_size, args.n_sample/t_avg, gpu_mem))
