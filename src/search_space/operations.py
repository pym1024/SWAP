import torch
import torch.nn as nn

OPS = {
    'none': lambda C_in, C_out, stride, affine: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine: POOLING(C_in, C_out, stride, 'avg', affine),
    'max_pool_3x3': lambda C_in, C_out, stride, affine: POOLING(C_in, C_out, stride, 'max', affine),
    'skip_connect': lambda C_in, C_out, stride, affine: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine),
    'sep_conv_3x3': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine),
    'dil_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine),
    'dil_conv_5x5': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, track_running_stats=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine, track_running_stats=track_running_stats),
            
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride=2, affine=True, track_running_stats=True):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in   = C_in
        self.C_out  = C_out
        self.relu   = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append( nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in   = C_in
        self.C_out  = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1: return x.mul(0.)
            else               : return x[:,:,::self.stride,::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
        if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, inputs):
        if self.preprocess: x = self.preprocess(inputs)
        else              : x = inputs
        return self.op(x)

