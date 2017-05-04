import sys
import torch
import torchvision
import numpy as np
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test CNN inference speed')
    parser.add_argument('--network', type=str, default='resnet50', help = 'network name')
    parser.add_argument('--params', type=str, help='model parameters')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--im-size', type=int, help='image size')
    parser.add_argument('--n-sample', type=int, default=1000, help='number of samples')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--n-epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--verbose', type=lambda x: x.lower() in ("yes", 'true', 't', '1'), default=True,
                        help='verbose information')
    args = parser.parse_args()

    print('===================== benchmark for %s =====================' % args.network)
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
    if args.network not in torchvision.models.__dict__:
        print('%s doesn\'t exists!' % args.network)
        sys.exit(1)
    net = torchvision.models.__dict__[args.network]()
    if args.params is not None:
        net.load_state_dict(torch.load(args.params))
        print('Load parameters from %s' % (args.params))
    else:
        print('Initialize randomly')

    net.cuda(device_id=args.gpu)
    net.eval()
    t2 = time.time()
    print('Finish loading model in %.4fs' % (t2-t1))

    t1 = time.time()
    data = np.random.uniform(-1, 1, (args.n_sample, 3, im_size, im_size))
    label = np.random.randint(0, 100, (args.n_sample,))
    tensor_imgs = torch.Tensor(data.astype(np.float32))
    tensor_labels = torch.Tensor(label.astype(np.float32))
    dataset = torch.utils.data.dataset.TensorDataset(tensor_imgs, tensor_labels)
    data_loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    t2 = time.time()
    print('Generate %d random images in %.4fs!' % (args.n_sample, t2-t1))

    t_list = []
    t_start = time.time()
    for i in range(args.n_epoch):
        t1 = time.time()

        for batch_id, (batch_input, batch_target) in enumerate(data_loader):
            batch_input_var = torch.autograd.Variable(batch_input, volatile=True).cuda(device_id=args.gpu)
            output =  net(batch_input_var)

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
