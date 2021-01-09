import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pad import *

class TransformerRNN(torch.nn.Module):
    def __init__(self, pad_type="reflect-start", upsample="deconv"):
        super(TransformerRNN, self).__init__()

        self.pad_type = pad_type

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, pad_type=pad_type)
        self.in1 = nn.InstanceNorm2d(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=4, stride=2, pad_type=pad_type)
        self.in2 = nn.InstanceNorm2d(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=4, stride=2, pad_type=pad_type)
        self.in3 = nn.InstanceNorm2d(128)

        # Residual layers
        self.res1 = ResidualBlock(128, pad_type)
        self.res2 = ResidualBlock(128, pad_type)
        self.res3 = ResidualBlock(128, pad_type)
        self.res4 = ResidualBlock(128, pad_type)
        self.res5 = ResidualBlock(128, pad_type)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, 4, 2, pad_type=pad_type, upsample=upsample)
        self.in4 = nn.InstanceNorm2d(64)
        self.deconv2 = UpsampleConvLayer(64, 32, 4, 2, pad_type=pad_type, upsample=upsample)
        self.in5 = nn.InstanceNorm2d(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1, pad_type=pad_type)

        # Non-linearities
        self.relu = nn.ReLU()

        self.setup_pad_input(256)

    def pad(self, x):
        if self.pad_type == "reflect-start":
            return self.pad_input(x)
        else:
            return x

    def setup_pad_input(self, size=256):
        x = torch.zeros(1, 3, size, size).float()
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        diff_h = x.size(2) - y.size(2)
        diff_w = x.size(3) - y.size(3)
        ph, pw = diff_h // 2, diff_w // 2
        self.pad_input = nn.ReflectionPad2d((ph, ph, pw, pw))
        print(str(x.size()) + " => " + str(y.size()))

    def forward(self, X, prev=None):
        """
        Split the batch dim according to time and do rnn unroll.
        """
        if prev is None: prev = torch.zeros_like(X[0:1])
        out = []
        for i in range(X.shape[0]):
            y = torch.cat([prev, X[i:i+1]], 1)
            y = self.pad(y)
            y = self.relu(self.in1(self.conv1(y)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y)
            y = self.res2(y)
            y = self.res3(y)
            y = self.res4(y)
            y = self.res5(y)
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            y = self.deconv3(y)
            # for evaluation, rnn dynamic BP graph is not maintained.
            if not self.training: y = y.detach()
            out.append(y)
        # the batch dim is recursive number
        return torch.cat(out)

class TransformerNet(torch.nn.Module):
    def __init__(self, pad_type="reflect-start", upsample="deconv"):
        super(TransformerNet, self).__init__()

        self.pad_type = pad_type

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1, pad_type=pad_type)
        self.in1 = nn.InstanceNorm2d(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=4, stride=2, pad_type=pad_type)
        self.in2 = nn.InstanceNorm2d(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=4, stride=2, pad_type=pad_type)
        self.in3 = nn.InstanceNorm2d(128)

        # Residual layers
        self.res1 = ResidualBlock(128, pad_type)
        self.res2 = ResidualBlock(128, pad_type)
        self.res3 = ResidualBlock(128, pad_type)
        self.res4 = ResidualBlock(128, pad_type)
        self.res5 = ResidualBlock(128, pad_type)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, 4, 2, pad_type=pad_type, upsample=upsample)
        self.in4 = nn.InstanceNorm2d(64)
        self.deconv2 = UpsampleConvLayer(64, 32, 4, 2, pad_type=pad_type, upsample=upsample)
        self.in5 = nn.InstanceNorm2d(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1, pad_type=pad_type)

        # Non-linearities
        self.relu = nn.ReLU()

        self.setup_pad_input(256)

    def record(self, x, name):
        self.record_tensor.append(x)
        self.record_name.append(name)

    def reset_pad_type(self, pad_type):
        self.pad_type = pad_type
        self.conv1.set_pad_method(pad_type)
        self.conv2.set_pad_method(pad_type)
        self.conv3.set_pad_method(pad_type)
        self.res1.pad_type = pad_type
        self.res2.pad_type = pad_type
        self.res3.pad_type = pad_type
        self.res4.pad_type = pad_type
        self.res5.pad_type = pad_type
        self.res1.conv1.set_pad_method(pad_type)
        self.res2.conv1.set_pad_method(pad_type)
        self.res3.conv1.set_pad_method(pad_type)
        self.res4.conv1.set_pad_method(pad_type)
        self.res5.conv1.set_pad_method(pad_type)
        self.res1.conv2.set_pad_method(pad_type)
        self.res2.conv2.set_pad_method(pad_type)
        self.res3.conv2.set_pad_method(pad_type)
        self.res4.conv2.set_pad_method(pad_type)
        self.res5.conv2.set_pad_method(pad_type)
        self.deconv1.pad_type = pad_type
        self.deconv2.pad_type = pad_type
        self.deconv3.set_pad_method(pad_type)

    def debug(self, X):
        self.record_tensor = []
        self.record_name = []
        y = X
        y = self.pad(y)
        y = self.conv1(y)
        self.record(y, "conv1")
        y = self.relu(self.in1(y))
        y = self.conv2(y)
        self.record(y, "conv2")
        y = self.relu(self.in2(y))
        y = self.conv3(y)
        self.record(y, "conv3")
        y = self.relu(self.in3(y))
        y = self.res1(y)
        self.record(y, "res1")
        y = self.res2(y)
        self.record(y, "res2")
        y = self.res3(y)
        self.record(y, "res3")
        y = self.res4(y)
        self.record(y, "res4")
        y = self.res5(y)
        self.record(y, "res5")
        y = self.deconv1(y)
        self.record(y, "deconv1")
        y = self.relu(self.in4(y))
        y = self.deconv2(y)
        self.record(y, "deconv2")
        y = self.relu(self.in5(y))
        y = self.deconv3(y)
        return y

    def print_shape(self):
        for n,t in zip(self.record_name, self.record_tensor):
            print("=> %s\t:%s" % (n, str(t.shape)))

    def setup_pad_input(self, size=256):
        x = torch.zeros(1, 3, size, size).float()
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        diff_h = x.size(2) - y.size(2)
        diff_w = x.size(3) - y.size(3)
        ph, pw = diff_h // 2, diff_w // 2
        self.pad_input = nn.ReflectionPad2d((ph, ph, pw, pw))
        self.diff_h, self.diff_w = diff_h, diff_w
        print(str(x.size()) + " => " + str(y.size()))

    def pad(self, x):
        if "reflect-start" in self.pad_type:
            return self.pad_input(x)
        elif "resize-start" in self.pad_type:
            n, c, h, w = x.shape
            return F.upsample_bilinear(x, (self.diff_h + h, self.diff_w + w))
        else:
            return x

    def forward(self, X):
        y = X
        y = self.pad(y)
        y = self.relu(self.in1(self.conv1(y)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_type):
        super(ConvLayer, self).__init__()
        self.pad_type = pad_type
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.set_pad_method(pad_type)

    def set_pad_method(self, pad_type):
        padding = (self.kernel_size - self.stride, self.kernel_size - self.stride)
        if pad_type == "none" or "start" in pad_type:
            self.pad = None
        else:
            self.pad = Padding2d(padding, pad_type)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, pad_type):
        super(ResidualBlock, self).__init__()
        self.pad_type = pad_type
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, pad_type=pad_type)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, pad_type=pad_type)
        self.in2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        if self.pad_type == "none" or "start" in self.pad_type:
            residual = residual[:, :, 2:-2, 2:-2]
        x = self.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        # no padding in the middle
        return x + residual

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_type="none", upsample="deconv"):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.kernel_size = kernel_size

        if upsample == "nearest":
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        elif upsample == "deconv":
            self.deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

        pad_size = int(np.floor(kernel_size / 2))
        self.pad_type = pad_type
        #self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

    def forward(self, x):
        if self.upsample == "nearest":
            x = self.upsample_layer(x)
            x = self.reflection_pad(x)
            x = self.conv2d(x)
        elif self.upsample == "deconv":
            x = self.deconv2d(x)
            # crop output
            if self.kernel_size == 3:
                x = x[:, :, 1:, 1:]
            elif self.kernel_size == 4:
                x = x[:, :, 1:-1, 1:-1]
        return x

"""
class nn.InstanceNorm2d(torch.nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(nn.InstanceNorm2d, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
"""