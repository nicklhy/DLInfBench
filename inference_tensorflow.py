import os
import sys
#  import the deep learning framework you want to use
#  ...
import numpy as np
import argparse
import time
import tensorflow as tf

from models.tensorflow import nets_factory

slim = tf.contrib.slim
#  change DLLIB to the deep learning framework's name
#  ...
DLLIB = 'tensorflow'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test CNN inference speed')
    parser.add_argument('--network', type=str, default='resnet50',
                        choices=['alexnet-v2', 'inception-v1', 'inception-bn', 'inception-v3',
                                 'inception-v4', 'inception-resnet-v2', 'resnet50', 'resnet101', 'resnet152',
                                 'resnet200', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200',
                                 'vgga', 'vgg16', 'vgg19'],
                        help = 'network name')
    parser.add_argument('--params', type=str, help='model parameters')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--im-size', type=int, help='image size')
    parser.add_argument('--n-sample', type=int, default=1000, help='number of samples')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--n-epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--warm-up-num', type=int, default=10, help='number of iterations for warming up')
    parser.add_argument('--verbose', type=lambda x: x.lower() in ("yes", 'true', 't', '1'), default=True,
                        help='verbose information')
    args = parser.parse_args()

    network_fn = nets_factory.get_network_fn(args.network, num_classes=1000, weight_decay=0.00004, is_training=False)

    print('===================== benchmark for %s %s =====================' % (DLLIB, args.network))
    print('n_sample=%d, batch size=%d, num epoch=%d' %  (args.n_sample, args.batch_size, args.n_epoch))

    im_size = 224
    if args.im_size is not None:
        im_size = args.im_size
    elif hasattr(network_fn, 'default_image_size'):
        im_size = network_fn.default_image_size

    #  loading model
    t1 = time.time()

    #  set up the network model for inference task
    #  ...
    inputs = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, im_size, im_size, 3])
    logits, end_points = network_fn(inputs)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=tf.GPUOptions(allow_growth=True,
                                                      visible_device_list=args.gpu))
    sess = tf.InteractiveSession(config=config)

    if args.params is not None:
        #  load pre-trained weights
        variables_to_restore = slim.get_model_variables()
        restore_op = slim.assign_from_checkpoint_fn(args.params,
                                       variables_to_restore)
        restore_op(sess)
        print('Load parameters from %s' % (args.params))
    else:
        #  initialize randomly
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('Initialize randomly')

    #  ...

    t2 = time.time()
    print('Finish loading model in %.4fs' % (t2-t1))

    t1 = time.time()
    data_list = [np.random.uniform(-1, 1, (args.batch_size, im_size, im_size, 3)).astype(np.float32)
                 for i in range(int(np.ceil(1.0*args.n_sample/args.batch_size)))]

    #  transform the data from numpy.ndarrays to the needed format(i.e. torch.Tensor)
    #  ...

    t2 = time.time()
    print('Generate %d random images in %.4fs!' % (args.n_sample, t2-t1))

    # warm-up to burn your GPU to working temperature (usually around 80C) to get stable numbers
    k = 0
    while k < args.warm_up_num:
        for batch in data_list:
            if k >= args.warm_up_num:
                break
            k += 1
            output = sess.run([logits], feed_dict={inputs:batch})
    print('Warm-up for 10 iterations')

    t_list = []
    t_start = time.time()
    for i in range(args.n_epoch):
        t1 = time.time()

        #  iterate and forward all data samples for one epoch
        #  ...
        for j, batch in enumerate(data_list):
            output = sess.run([logits], feed_dict={inputs:batch})

        t2 = time.time()
        t_list.append(t2-t1)
        if args.verbose:
            print('Epoch %d, finish %d images in %.4fs, speed = %.4f image/s' % (i, args.n_sample, t2-t1, args.n_sample/(t2-t1)))

    t_end = time.time()

    t_list = np.array(t_list)
    if args.n_epoch > 2:
        #  delete the slowest and fastest results
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

    res_file_path = os.path.join(res_dir, '%s_%s_%d.txt' % (DLLIB, args.network, args.batch_size))
    with open(res_file_path, 'w') as fd:
        fd.write('%s %s %d %f %d' % (DLLIB, args.network, args.batch_size, args.n_sample/t_avg, gpu_mem))
