import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
import tensorrt as trt
import numpy as np
import argparse
import time

DLLIB = 'tensorrt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test CNN inference speed')
    parser.add_argument('--network', type=str, default='resnet50',
                        choices=['alexnet', 'inception-bn', 'inception-v3', 'resnet50', 'resnet101', 'resnet152',  'vgg16',  'vgg19'],
                        help='network')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16'],
                        help='data type')
    parser.add_argument('--params', type=str, help='model parameters')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--im-size', type=int, help='image size')
    parser.add_argument('--n-sample', type=int, default=1024, help='number of samples')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--n-epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--warm-up-num', type=int, default=10, help='number of iterations for warming up')
    parser.add_argument('--verbose', type=lambda x: x.lower() in ("yes", 'true', 't', '1'), default=True,
                        help='verbose information')
    parser.add_argument('--python', type=str, default='python3',
                        help='python executable command/path')
    parser.add_argument('--onnx2trt', type=str, default='onnx2trt',
                        help='onnx2trt executable command/path')
    args = parser.parse_args()

    print('===================== benchmark for %s %s (%s) =====================' % (DLLIB, args.network, args.dtype))
    print('n_sample=%d, batch_size=%d, n_epoch=%d' %  (args.n_sample, args.batch_size, args.n_epoch))

    im_size = 224
    if args.im_size is not None:
        im_size = args.im_size
    elif args.network == 'alexnet':
        im_size = 227
    elif args.network == 'inception-v3':
        im_size = 299

    #  loading model
    t1 = time.time()
    print('Convert mxnet to onnx model ...')
    net_path = os.path.join(ROOT_DIR, 'models',
                            'mxnet', args.network+'.json')
    if not os.path.exists(net_path):
        print('%s doesn\'t exists!' % args.network)
        sys.exit(1)
    onnx_file = os.path.join(ROOT_DIR, 'models',
                             'onnx', args.network+'_from_mx.onnx')
    ret = os.system('%s mx2onnx.py --net-json %s --batch-size %d --output %s'
                    % (args.python, net_path, args.batch_size, onnx_file))
    if ret != 0:
        raise RuntimeError('Failed to convert mxnet to onnx!')
    trt_file = os.path.join(ROOT_DIR, 'models',
                            'trt', args.network+'_from_mx_bs'+str(args.batch_size)+'.plan')
    if not os.path.exists(os.path.join(ROOT_DIR, 'models', 'trt')):
        os.makedirs(os.path.join(ROOT_DIR, 'models', 'trt'))

    if args.dtype == 'float16':
        arg_dtype = 16
    elif args.dtype == 'float32':
        arg_dtype = 32
    else:
        raise ValueError('Unknown data type: %s' % args.dtype)
    ret = os.system('%s %s -o %s -d %d' % (args.onnx2trt, onnx_file,
                                           trt_file, arg_dtype))
    if ret != 0:
        raise RuntimeError('Failed to convert onnx to tensorrt plan file!')

    if args.dtype == 'float32':
        trt_dtype = trt.infer.DataType.FLOAT
    elif args.dtype == 'float16':
        trt_dtype = trt.infer.DataType.Half
    else:
        raise ValueError('Unknown dtype: %s' % args.dtype)
    engine = trt.lite.Engine(PLAN=trt_file,
                             data_type=trt_dtype,
                             logger_severity=trt.infer.LogSeverity.ERROR,
                             max_batch_size=args.batch_size)
    t2 = time.time()
    print('Finish loading model %s in %.4fs' % (trt_file, t2-t1))

    t1 = time.time()
    data_list = [np.random.uniform(-1, 1, (args.batch_size, 3, im_size, im_size)).astype(np.float32) for i in range(int(np.ceil(1.0*args.n_sample/args.batch_size)))]
    t2 = time.time()
    print('Generate %d random images in %.4fs' % (args.n_sample, t2-t1))

    # warm-up to burn your GPU to working temperature (usually around 80C) to get stable numbers
    k = 0
    while k < args.warm_up_num:
        for batch in data_list:
            if k >= args.warm_up_num:
                break
            k += 1
            out = engine.infer(batch)[-1]
    print('Warm-up for %d iterations' % args.warm_up_num)

    t_list = []
    t_start = time.time()
    for i in range(args.n_epoch):
        t1 = time.time()

        for batch in data_list:
            out = engine.infer(batch)[-1]

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
