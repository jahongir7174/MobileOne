import copy
import math

import torch
from torch.nn.functional import pad


class SE(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(ch, ch // 16, 1),
                                      torch.nn.ReLU(),
                                      torch.nn.Conv2d(ch // 16, ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, d=1, g=1,
                 inference_mode=False, use_se=False, num_conv=1):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.kernel_size = k
        self.stride = s
        self.groups = g
        self.num_conv = num_conv
        self.inference_mode = inference_mode

        # Check if SE is enabled
        if use_se:
            self.se = SE(out_ch)
        else:
            self.se = torch.nn.Identity()
        self.activation = torch.nn.ReLU()

        if inference_mode:
            self.fused_conv = torch.nn.Conv2d(in_channels=in_ch,
                                              out_channels=out_ch,
                                              kernel_size=k,
                                              stride=s,
                                              padding=p,
                                              dilation=d,
                                              groups=g,
                                              bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = torch.nn.BatchNorm2d(num_features=in_ch) \
                if out_ch == in_ch and s == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv):
                rbr_conv.append(self.conv_bn(kernel_size=k, padding=p))
            self.rbr_conv = torch.nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if k > 1:
                self.rbr_scale = self.conv_bn(kernel_size=1, padding=0)

    def forward(self, x):
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.fused_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def re_parameterize(self):
        if self.inference_mode:
            return
        kernel, bias = self.kernel_bias()
        self.fused_conv = torch.nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                          out_channels=self.rbr_conv[0].conv.out_channels,
                                          kernel_size=self.rbr_conv[0].conv.kernel_size,
                                          stride=self.rbr_conv[0].conv.stride,
                                          padding=self.rbr_conv[0].conv.padding,
                                          dilation=self.rbr_conv[0].conv.dilation,
                                          groups=self.rbr_conv[0].conv.groups,
                                          bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def kernel_bias(self):
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self.fuse_bn(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            p = self.kernel_size // 2
            kernel_scale = pad(kernel_scale, [p, p, p, p])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self.fuse_bn(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for i in range(self.num_conv):
            _kernel, _bias = self.fuse_bn(self.rbr_conv[i])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def fuse_bn(self, module):
        if isinstance(module, torch.nn.Sequential):
            kernel = module.conv.weight
            running_mean = module.bn.running_mean
            running_var = module.bn.running_var
            gamma = module.bn.weight
            beta = module.bn.bias
            eps = module.bn.eps
        else:
            assert isinstance(module, torch.nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=module.weight.dtype,
                                           device=module.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = module.running_mean
            running_var = module.running_var
            gamma = module.weight
            beta = module.bias
            eps = module.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def conv_bn(self, kernel_size: int, padding: int):
        module = torch.nn.Sequential()
        module.add_module('conv', torch.nn.Conv2d(in_channels=self.in_channels,
                                                  out_channels=self.out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=self.stride,
                                                  padding=padding,
                                                  groups=self.groups,
                                                  bias=False))
        module.add_module('bn', torch.nn.BatchNorm2d(num_features=self.out_channels))
        return module


class MobileOne(torch.nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_multipliers=None,
                 inference_mode=False,
                 use_se=False,
                 num_conv=1):
        super().__init__()

        assert len(width_multipliers) == 4
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv = num_conv
        self.inference_mode = inference_mode

        # Build stages
        self.p1 = Residual(in_ch=3, out_ch=self.in_planes,
                           k=3, s=2, p=1,
                           inference_mode=self.inference_mode)

        self.p2 = self.make_stage(int(64 * width_multipliers[0]), 2,
                                  num_se_blocks=0)
        self.p3 = self.make_stage(int(128 * width_multipliers[1]), 8,
                                  num_se_blocks=0)
        self.p4 = self.make_stage(int(256 * width_multipliers[2]), 10,
                                  num_se_blocks=5 if use_se else 0)
        self.p5 = self.make_stage(int(512 * width_multipliers[3]), 1,
                                  num_se_blocks=1 if use_se else 0)
        self.fc = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(int(512 * width_multipliers[3]), num_classes))

    def make_stage(self, planes, num_blocks, num_se_blocks):
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for i, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot exceed number of layers.")
            if i >= (num_blocks - num_se_blocks):
                use_se = True

            # Depth-wise conv
            blocks.append(Residual(in_ch=self.in_planes,
                                   out_ch=self.in_planes,
                                   k=3,
                                   s=stride,
                                   p=1,
                                   g=self.in_planes,
                                   inference_mode=self.inference_mode,
                                   use_se=use_se,
                                   num_conv=self.num_conv))
            # Point-wise conv
            blocks.append(Residual(in_ch=self.in_planes,
                                   out_ch=planes,
                                   k=1,
                                   s=1,
                                   p=0,
                                   g=1,
                                   inference_mode=self.inference_mode,
                                   use_se=use_se,
                                   num_conv=self.num_conv))
            self.in_planes = planes
        return torch.nn.Sequential(*blocks)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        return self.fc(x)


def mobile_one_s0(num_classes: int = 1000, inference_mode: bool = False):
    return MobileOne(num_classes, (0.75, 1.0, 1.0, 2.0), inference_mode, False, 4)


def mobile_one_s1(num_classes: int = 1000, inference_mode: bool = False):
    return MobileOne(num_classes, (1.50, 1.5, 2.0, 2.5), inference_mode, False, 1)


def mobile_one_s2(num_classes: int = 1000, inference_mode: bool = False):
    return MobileOne(num_classes, (1.50, 2.0, 2.5, 4.0), inference_mode, False, 1)


def mobile_one_s3(num_classes: int = 1000, inference_mode: bool = False):
    return MobileOne(num_classes, (2.00, 2.5, 3.0, 4.0), inference_mode, False, 1)


def mobile_one_s4(num_classes: int = 1000, inference_mode: bool = False):
    return MobileOne(num_classes, (3.00, 3.5, 3.5, 4.0), inference_mode, True, 1)


def re_parameterize_model(model):
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 're_parameterize'):
            module.re_parameterize()
    return model


class EMA:
    def __init__(self, model, decay=0.9995):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, args, model):
        if args.distributed:
            model = model.module

        m_std = model.state_dict().values()
        e_std = self.model.state_dict().values()

        for m, e in zip(m_std, e_std):
            e.copy_(self.decay * e + (1. - self.decay) * m)


class CosineLR:
    def __init__(self, args, optimizer):
        self.optimizer = optimizer

        self.epochs = args.epochs
        self.values = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.warmup_epochs = 5
        self.warmup_values = [(v - 1e-4) / self.warmup_epochs for v in self.values]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1e-4

    def step(self, epoch):
        epochs = self.epochs

        if epoch < self.warmup_epochs:
            values = [1e-4 + epoch * value for value in self.warmup_values]
        else:
            epoch = epoch - self.warmup_epochs
            if epoch < epochs:
                alpha = math.pi * (epoch - (epochs * (epoch // epochs))) / epochs
                values = [1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(alpha)) for lr in self.values]
            else:
                values = [1e-5 for _ in self.values]

        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        prob = self.softmax(outputs)
        loss = -(prob.gather(dim=-1, index=targets.unsqueeze(1))).squeeze(1)

        return ((1.0 - self.epsilon) * loss - self.epsilon * prob.mean(dim=-1)).mean()
