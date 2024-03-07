import copy
import math

import torch


class SE(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(ch, ch // 16, kernel_size=1),
                                      torch.nn.ReLU(),
                                      torch.nn.Conv2d(ch // 16, ch, kernel_size=1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, se=False, num_conv=1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Identity()

        # Check if SE is enabled
        if se:
            self.se = SE(out_ch)
        else:
            self.se = torch.nn.Identity()

        # Re-parameterizable conv branches
        conv1 = list()
        for _ in range(num_conv):
            conv1.append(Conv(in_ch, out_ch, k=k, s=s, p=p, g=g))
        self.conv1 = torch.nn.ModuleList(conv1)

        # Re-parameterizable scale branch
        self.conv2 = Conv(in_ch, out_ch, k=1, s=s, p=0, g=g) if k > 1 else None

        # Re-parameterizable skip branch
        self.identity = torch.nn.BatchNorm2d(in_ch) if in_ch == out_ch and s == 1 else None

    def forward(self, x):
        # Multi-branched train-time forward pass.

        # Re-parameterizable conv branches output
        y1 = 0
        for conv1 in self.conv1:
            y1 += conv1(x)

        # Re-parameterizable scale branch output
        y2 = 0
        if self.conv2 is not None:
            y2 = self.conv2(x)

        # Re-parameterizable skip branch output
        y3 = 0
        if self.identity is not None:
            y3 = self.identity(x)

        return self.relu(self.se(y1 + y2 + y3))

    def fuse_forward(self, x):
        # Inference mode forward pass.
        return self.relu(self.se(self.conv(x)))

    def fuse(self):
        # get weights and bias of skip branch
        k1 = 0
        b1 = 0
        if self.identity is not None:
            k1, b1 = self.__fuse(self.identity)

        # get weights and bias of conv branches
        k2 = 0
        b2 = 0
        for conv1 in self.conv1:
            k, b = self.__fuse(conv1)
            k2 += k
            b2 += b

        # get weights and bias of scale branch
        k3 = 0
        b3 = 0
        if self.conv2 is not None:
            k3, b3 = self.__fuse(self.conv2)
            # Pad scale branch kernel to match conv branch kernel size.
            p = self.conv1[0].conv.kernel_size[0] // 2
            k3 = torch.nn.functional.pad(k3, pad=[p, p, p, p])

        self.conv = torch.nn.Conv2d(in_channels=self.conv1[0].conv.in_channels,
                                    out_channels=self.conv1[0].conv.out_channels,
                                    kernel_size=self.conv1[0].conv.kernel_size,
                                    stride=self.conv1[0].conv.stride,
                                    padding=self.conv1[0].conv.padding,
                                    dilation=self.conv1[0].conv.dilation,
                                    groups=self.conv1[0].conv.groups,
                                    bias=True)
        self.conv.weight.data = k1 + k2 + k3
        self.conv.bias.data = b1 + b2 + b3

        # Delete un-used branches
        for p in self.parameters():
            p.detach_()
        if hasattr(self, 'identity'):
            self.__delattr__('identity')
        if hasattr(self, 'conv1'):
            self.__delattr__('conv1')
        if hasattr(self, 'conv2'):
            self.__delattr__('conv2')

        self.forward = self.fuse_forward

    def __fuse(self, m):
        if isinstance(m, Conv):
            kernel = m.conv.weight
            running_mean = m.norm.running_mean
            running_var = m.norm.running_var
            gamma = m.norm.weight
            beta = m.norm.bias
            eps = m.norm.eps
        else:
            assert isinstance(m, torch.nn.BatchNorm2d)
            if not hasattr(self, 'norm'):
                in_channels = self.conv1[0].conv.in_channels
                k = self.conv1[0].conv.kernel_size[0]
                g = self.conv1[0].conv.groups
                kernel_value = torch.zeros((in_channels, in_channels // g, k, k),
                                           dtype=m.weight.dtype,
                                           device=m.weight.device)
                for i in range(in_channels):
                    kernel_value[i, i % in_channels // g, k // 2, k // 2] = 1
                self.norm = kernel_value
            kernel = self.norm
            running_mean = m.running_mean
            running_var = m.running_var
            gamma = m.weight
            beta = m.bias
            eps = m.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class MobileOne(torch.nn.Module):
    def __init__(self, width, depth, se, num_conv, num_classes=1000):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1
        for i in range(depth[0]):
            if i == 0:
                self.p1.append(Residual(width[0], width[1], k=3, s=2, p=1))
            else:
                # Depth-wise conv
                self.p1.append(Residual(width[1], width[1],
                                        k=3, s=1, p=1, g=width[2], num_conv=num_conv))
                # Point-wise conv
                self.p1.append(Residual(width[1], width[1],
                                        k=1, s=1, p=0, g=1, num_conv=num_conv))
        # p2
        for i in range(depth[1]):
            if i == 0:
                # Depth-wise conv
                self.p2.append(Residual(width[1], width[1],
                                        k=3, s=2, p=1, g=width[1], num_conv=num_conv))
                # Point-wise conv
                self.p2.append(Residual(width[1], width[2],
                                        k=1, s=1, p=0, g=1, num_conv=num_conv))
            else:
                # Depth-wise conv
                self.p2.append(Residual(width[2], width[2],
                                        k=3, s=1, p=1, g=width[2], num_conv=num_conv))
                # Point-wise conv
                self.p2.append(Residual(width[2], width[2],
                                        k=1, s=1, p=0, g=1, num_conv=num_conv))
        # p3
        for i in range(depth[2]):
            if i == 0:
                # Depth-wise conv
                self.p3.append(Residual(width[2], width[2],
                                        k=3, s=2, p=1, g=width[2], num_conv=num_conv))
                # Point-wise conv
                self.p3.append(Residual(width[2], width[3],
                                        k=1, s=1, p=0, g=1, num_conv=num_conv))
            else:
                # Depth-wise conv
                self.p3.append(Residual(width[3], width[3],
                                        k=3, s=1, p=1, g=width[3], num_conv=num_conv))
                # Point-wise conv
                self.p3.append(Residual(width[3], width[3],
                                        k=1, s=1, p=0, g=1, num_conv=num_conv))
        # p4
        for i in range(depth[3]):
            use_se = True if i >= 5 and se else False
            if i == 0:
                # Depth-wise conv
                self.p4.append(Residual(width[3], width[3],
                                        k=3, s=2, p=1, g=width[3], se=use_se, num_conv=num_conv))
                # Point-wise conv
                self.p4.append(Residual(width[3], width[4],
                                        k=1, s=1, p=0, g=1, se=use_se, num_conv=num_conv))
            else:
                # Depth-wise conv
                self.p4.append(Residual(width[4], width[4],
                                        k=3, s=1, p=1, g=width[4], se=use_se, num_conv=num_conv))
                # Point-wise conv
                self.p4.append(Residual(width[4], width[4],
                                        k=1, s=1, p=0, g=1, se=use_se, num_conv=num_conv))
        # p5
        for i in range(depth[4]):
            use_se = True if i >= 0 and se else False
            if i == 0:
                # Depth-wise conv
                self.p5.append(Residual(width[4], width[4],
                                        k=3, s=2, p=1, g=width[4], se=use_se, num_conv=num_conv))
                # Point-wise conv
                self.p5.append(Residual(width[4], width[5],
                                        k=1, s=1, p=0, g=1, se=use_se, num_conv=num_conv))
            else:
                # Depth-wise conv
                self.p5.append(Residual(width[5], width[5],
                                        k=3, s=1, p=1, g=width[5], se=use_se, num_conv=num_conv))
                # Point-wise conv
                self.p5.append(Residual(width[5], width[5],
                                        k=1, s=1, p=0, g=1, se=use_se, num_conv=num_conv))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.fc = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(width[5], num_classes))

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        return self.fc(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Residual:
                m.fuse()
        return self


def mobile_one_s0():
    return MobileOne(width=(3, 48, 48, 128, 256, 1024), depth=(1, 2, 8, 10, 1), se=False, num_conv=4)


def mobile_one_s1():
    return MobileOne(width=(3, 64, 96, 192, 512, 1280), depth=(1, 2, 8, 10, 1), se=False, num_conv=1)


def mobile_one_s2():
    return MobileOne(width=(3, 64, 96, 256, 640, 2048), depth=(1, 2, 8, 10, 1), se=False, num_conv=1)


def mobile_one_s3():
    return MobileOne(width=(3, 64, 128, 320, 768, 2048), depth=(1, 2, 8, 10, 1), se=False, num_conv=1)


def mobile_one_s4():
    return MobileOne(width=(3, 64, 192, 448, 896, 2048), depth=(1, 2, 8, 10, 1), se=True, num_conv=1)


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
    def __init__(self, args, lr):
        self.lr = []
        self.weigh_decay = []
        for epoch in range(args.epochs):
            alpha = math.pi * epoch / args.epochs
            self.lr.append(1E-4 + 0.5 * (lr - 1E-4) * (1 + math.cos(alpha)))
            self.weigh_decay.append(1E-5 + 0.5 * (1E-4 - 1E-5) * (1 + math.cos(alpha)))

    def step(self, epoch, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr[epoch]
            param_group['weight_decay'] = self.weigh_decay[epoch]


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        prob = self.softmax(outputs)
        loss = -(prob.gather(dim=-1, index=targets.unsqueeze(1))).squeeze(1)

        return ((1.0 - self.epsilon) * loss - self.epsilon * prob.mean(dim=-1)).mean()
