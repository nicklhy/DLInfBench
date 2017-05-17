import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot the speed benchmark results')
    parser.add_argument('--network', type=str, default='resnet50', help='network name')
    parser.add_argument('--res-dir', type=str, default='cache/results', help='network name')
    args = parser.parse_args()

    results = {}
    file_list = glob.glob('%s/*_%s_*.txt' % (args.res_dir, args.network))
    for fpath in file_list:
        fname = os.path.split(fpath)[-1]
        dllib, network, batch_size = os.path.splitext(fname)[0].split('_')
        speed = -1
        gpu_mem = -1
        batch_size = int(batch_size)
        with open(fpath, 'r') as fd:
            _dllib, _network, _batch_size, speed, gpu_mem = fd.readline().strip().split()
            assert(_dllib==dllib)
            assert(_network==network)
            assert(_network==args.network)
            assert(int(_batch_size)==batch_size)
            gpu_mem = int(gpu_mem)
            speed = float(speed)
        if dllib not in results:
            results[dllib] = {
                'batch_size': [],
                'speed': [],
                'gpu memory': [],
            }
        if speed == -1 or batch_size == -1:
            #  skip if failed
            continue
        results[dllib]['batch_size'].append(batch_size)
        results[dllib]['gpu memory'].append(gpu_mem)
        results[dllib]['speed'].append(speed)

    for dllib in results.keys():
        ind_sort = np.argsort(results[dllib]['batch_size'])
        for k in results[dllib].keys():
            #  sort by batch size
            results[dllib][k] = [results[dllib][k][i] for i in ind_sort]

    print('Read results: %s' % results)

    for target in ['speed', 'gpu memory']:
        plt.clf()
        plt.figure(figsize=(12, 8))
        plt.title('%s benchmark' % target)
        plt.ylabel('speed(images/s)')
        plt.xlabel('batch size')
        xticks = []
        for dllib in results:
            plt.plot(results[dllib]['batch_size'], results[dllib][target], label=dllib, marker='x')
            if len(results[dllib]['batch_size']) > len(xticks):
                xticks = results[dllib]['batch_size']
        plt.legend(loc=2)
        plt.xticks(xticks)
        res_path = os.path.join(args.res_dir, '%s_%s.png' % (args.network, target.replace(' ', '_')))
        print('Save %s benchmark results to: %s' % (target, res_path))
        plt.savefig(res_path)
