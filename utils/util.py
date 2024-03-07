import math
import random
from os import environ
from platform import system

import cv2
import numpy
import torch
from PIL import Image


def print_benchmark(model, shape):
    import os
    import onnx
    from caffe2.proto import caffe2_pb2
    from caffe2.python.onnx.backend import Caffe2Backend
    from caffe2.python import core, model_helper, workspace

    inputs = torch.randn(shape, requires_grad=True)
    model(inputs)

    # export torch to onnx
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}

    torch.onnx.export(model, inputs, './weights/model.onnx', True, False,
                      input_names=["input0"],
                      output_names=["output0"],
                      keep_initializers_as_inputs=True,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                      dynamic_axes=dynamic_axes,
                      opset_version=10)

    onnx.checker.check_model(onnx.load('./weights/model.onnx'))

    # export onnx to caffe2
    onnx_model = onnx.load('./weights/model.onnx')

    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)

    # print benchmark
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    init_net_proto.ParseFromString(caffe2_init.SerializeToString())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    predict_net_proto.ParseFromString(caffe2_predict.SerializeToString())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=shape, mean=0.0, std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)
    # remove onnx model
    os.remove('./weights/model.onnx')


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def plot_lr(args, optimizer, scheduler):
    import copy
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        scheduler.step(epoch, optimizer)
        y.append(optimizer.param_groups[0]['lr'])

    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('epoch')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def weight_decay(model, decay=1E-4):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.},
            {'params': p2, 'weight_decay': decay}]


@torch.no_grad()
def accuracy(output, target, top_k):
    output = output.topk(max(top_k), 1, True, True)[1].t()
    output = output.eq(target.view(1, -1).expand_as(output))

    results = []
    for k in top_k:
        correct = output[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct.mul_(100.0 / target.size(0)))
    return results


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        size = self.size
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([size, size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        if (size[0] / size[1]) < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (size[0] / size[1]) > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num
