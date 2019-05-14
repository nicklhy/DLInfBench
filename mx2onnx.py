#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import logging
import argparse
import mxnet as mx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert a mxnet model to onnx format.'
    )
    parser.add_argument('--net-json', type=str, required=True,
                        help='network symbol file')
    parser.add_argument('--layers', type=str,
                        help='network output layers')
    parser.add_argument('--params', type=str,
                        help='network params file (optional)')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='input data type')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--n-channels', type=int, default=3,
                        help='number of channels')
    parser.add_argument('--height', type=int, default=224,
                        help='input height')
    parser.add_argument('--width', type=int, default=224,
                        help='input width')
    parser.add_argument('--input-name', type=str, default='data',
                        help='input name')
    parser.add_argument('--label-name', type=str, default='softmax_label',
                        help='input label name')
    parser.add_argument('--output', type=str, required=True,
                        help='output uff file')
    args = parser.parse_args()

    #  set logging
    head = '[%(asctime)s] {%(filename)s:%(lineno)d} [%(levelname)s] - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    sym = mx.sym.load(args.net_json)
    if args.layers is not None:
        internals = sym.get_internals()
        sym = mx.sym.Group([internals[x] for x in args.layers.split(',')])

    params = {}
    if args.params is not None:
        save_dict = mx.nd.load(args.params)
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                params[name] = v
            elif tp == 'aux':
                params[name] = v
        logging.info('Initialize parameters from %s!' % args.params)
    else:
        mod = mx.mod.Module(
            symbol=sym,
            data_names=[args.input_name],
            label_names=[args.label_name] if args.label_name else None,
            context=mx.cpu()
        )
        mod.bind(
            data_shapes=[(args.input_name, (args.batch_size,
                                            args.n_channels,
                                            args.height,
                                            args.width))],
            label_shapes=[(args.label_name, (args.batch_size,))] if args.label_name else None
        )
        mod.init_params(initializer=mx.init.Normal(0.01))
        logging.info('Initialize parameters randomly!')
        arg_params, aux_params = mod.get_params()
        params = arg_params
        params.update(aux_params)

    dir_name = os.path.dirname(args.output)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    input_shape = [
        (args.batch_size,
         args.n_channels,
         args.height,
         args.width),
    ]
    if args.label_name:
        input_shape.append((args.batch_size,))
    mx.contrib.onnx.export_model(
        sym,
        params,
        input_shape=input_shape,
        input_type=args.dtype,
        onnx_file_path=args.output,
        verbose=False
    )
