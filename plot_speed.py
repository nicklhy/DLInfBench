import os
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot the speed benchmark results')
    parser.add_argument('--network', type=str, default='resnet50', help='network name')
    parser.add_argument('--res-dir', type=str, default='cache/results',
                        help='result file dir')
    args = parser.parse_args()

    results = {}
    file_list = glob.glob('%s/*_%s_*.txt' % (args.res_dir, args.network))
    if len(file_list) == 0:
        print('No results for %s' % args.network)
        sys.exit(1)
    for fpath in file_list:
        fname = os.path.split(fpath)[-1]
        t = os.path.splitext(fname)[0].split('_')
        if len(t) == 4:
            dllib, network, dtype, batch_size = t
        elif len(t) == 3:
            dllib, network, batch_size = t
            dtype = 'float32'
        else:
            raise ValueError('Unknown file')
        speed = -1
        gpu_mem = -1
        batch_size = int(batch_size)
        with open(fpath, 'r') as fd:
            t = fd.readline().strip().split()
            if len(t) == 6:
                _dllib, _network, _dtype, _batch_size, speed, gpu_mem = t
            elif len(t) == 5:
                _dllib, _network, _batch_size, speed, gpu_mem = t
                _dtype = 'float32'
            else:
                raise ValueError('Unknown result content: %s' % ' '.join(t))

            assert(_dllib==dllib)
            assert(_network==network)
            assert(_network==args.network)
            assert(_dtype==dtype)
            assert(int(_batch_size)==batch_size)
            gpu_mem = int(gpu_mem)
            speed = float(speed)
        name = dllib+'_'+dtype
        if name not in results:
            results[name] = {
                'batch_size': [],
                'speed': [],
                'gpu memory': [],
            }
        if speed == -1 or batch_size == -1:
            #  skip if failed
            continue
        results[name]['batch_size'].append(batch_size)
        results[name]['gpu memory'].append(gpu_mem)
        results[name]['speed'].append(speed)

    for name in results.keys():
        ind_sort = np.argsort(results[name]['batch_size'])
        for k in results[name].keys():
            #  sort by batch size
            results[name][k] = [results[name][k][i] for i in ind_sort]

    print('Read results: %s' % results)

    for target in ['Speed', 'GPU Memory']:
        plt.clf()
        plt.figure(figsize=(12, 8))
        plt.title('%s %s Benchmark' % (args.network.capitalize(), target))
        if target == 'Speed':
            plt.ylabel('Speed(images/s)')
        elif target == 'GPU Memory':
            plt.ylabel('GPU Memory(MB)')
        plt.xlabel('Batch Size')
        xticks = []
        for name in results:
            plt.plot(
                results[name]['batch_size'],
                results[name][target.lower()],
                label=name,
                marker='x'
            )
            for x, y in zip(results[name]['batch_size'],
                            results[name][target.lower()]):
                plt.text(x, y, '%.2f' % y, fontsize=6)
            if len(results[name]['batch_size']) > len(xticks):
                xticks = results[name]['batch_size']
        plt.legend(loc=4)
        plt.xticks(xticks)
        res_path = os.path.join(args.res_dir, '%s_%s.png' % (args.network,
                                                                target.lower().replace(' ', '_')))
        print('Save %s benchmark results to: %s' % (target.lower(), res_path))
        plt.savefig(res_path)
