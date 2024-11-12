import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras import layers, models

import megengine as mge
import numpy as np

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        print('torch', mu.shape, var.shape)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class LayerNorm2D(tf.keras.layers.Layer):
    def __init__(self, channels, epsilon=1e-6, **kwargs):
        super(LayerNorm2D, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = self.add_weight(shape=(channels,), initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(channels,), initializer='zeros', trainable=True)

    def call(self, x):
        # Compute mean and variance
        mean, variance = tf.nn.moments(x, axes=3, keepdims=True)
        print('tf', mean.shape, variance.shape)
        # Normalize
        x_norm = (x - mean) / tf.sqrt(variance + self.epsilon)
        # Scale and shift
        return self.gamma[None, None, None, :] * x_norm + self.beta[None, None, None, :]

class SimpleGateTF(layers.Layer):
    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=3)
        return x1 * x2

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class DepthwiseSeparableConvTF(layers.Layer):
    def __init__(self, nin, nout, kernel_size=3, padding='valid', stride=1, use_bias=False):
        super(DepthwiseSeparableConvTF, self).__init__()
        self.depthwise = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias)
        self.pointwise = layers.Conv2D(nout, kernel_size=1, padding='valid', use_bias=use_bias)

    def call(self, x):
        x = self.depthwise(x)
        print('dp_tf:', np.array(x).mean())
        x = self.pointwise(x)
        print('pw_tf:', np.array(x).mean())
        return x
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 0, stride = 1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        print('dp_pt:', x.mean())
        x = self.pointwise(x)
        print('pw_pt:', x.mean())
        return x
      
class UpsampleWithFlopsTF(layers.Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(UpsampleWithFlopsTF, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.flops = 0

    def call(self, inputs):
        self.flops += tf.size(inputs)
        return tf.image.resize(inputs, size=self.size, method=self.mode)  
    
class UpsampleWithFlops(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
        self.__flops__ = 0

    def forward(self, input):
        self.__flops__ += input.numel()
        return super(UpsampleWithFlops, self).forward(input)
    
input = np.random.randint(0, 100, (1, 544, 960, 4)).astype(np.float32)
input_tf = input
arr_tf = LayerNorm2D(4)(input)
arr_tf = np.array(arr_tf)

input_torch = torch.Tensor(np.transpose(input, (0, 3, 1, 2)))
input_pt = input_torch
# print(arr.shape)
arr_torch = LayerNorm2d(4)(input_torch)
arr_torch = arr_torch.detach().numpy()
print(arr_tf.mean(), arr_tf.shape)
print(arr_torch.mean(), arr_torch.shape)

d2s_tf = tf.nn.depth_to_space(input, block_size=2)
d2s_pt = nn.PixelShuffle(2)(input_torch)
print('d2s: ', np.array(d2s_tf).shape, d2s_pt.numpy().shape)
print('d2s: ', np.array(d2s_tf)[0,500,1000,0], d2s_pt.numpy()[0,0,500,1000])

sg_tf = SimpleGateTF()(input)
sg_pt = SimpleGate()(input_torch)
print('sg: ', np.array(sg_tf).shape, sg_pt.numpy().shape)
print('sg: ', np.array(sg_tf)[0,300,600,0], sg_pt.numpy()[0,0,300,600])
print('sg: ', np.array(sg_tf).mean(), sg_pt.numpy().mean())


dconv_tf = DepthwiseSeparableConvTF(4, 4, 3, stride=3)
dpconv_tf = dconv_tf(input)
print('depconv_tf: ', dconv_tf.depthwise.get_weights()[0].mean(), dconv_tf.depthwise.get_weights()[0].shape)
print('dpointconv_tf: ', dconv_tf.pointwise.get_weights()[0].mean(), dconv_tf.pointwise.get_weights()[0].shape)

dconv_pt = DepthwiseSeparableConv(4, 4, 3, stride=3)
dconv_pt.depthwise.weight.data = torch.Tensor(np.transpose(dconv_tf.depthwise.get_weights()[0] ,(2,3,0,1)))
dconv_pt.pointwise.weight.data = torch.Tensor(np.transpose(dconv_tf.pointwise.get_weights()[0], (3,2,0,1)))
dpconv_pt = dconv_pt(input_torch)
print('depconv_pt: ', dconv_pt.depthwise.weight.data.clone().mean(), dconv_pt.depthwise.weight.data.clone().shape)
print('dpointconv_pt: ', dconv_pt.pointwise.weight.data.clone().mean(), dconv_pt.pointwise.weight.data.clone().shape)
dpconv_tf = np.array(dpconv_tf)
dp_conv_pt = dpconv_pt.detach().numpy()
print('dpconv: ', np.array(dpconv_tf).shape, dp_conv_pt.shape)
print('dpconv: ', np.array(dpconv_tf)[0,100,200,0], dp_conv_pt[0,0,100,200])
print('dpconv: ', np.array(dpconv_tf).mean(), dp_conv_pt.mean())

h = 2000
w = 4000
upsample_tf = UpsampleWithFlopsTF(size=(h,w), mode='nearest')(input_tf)
upsample_pt = UpsampleWithFlops(size=(h,w), mode='nearest')(input_pt)
print('upsample: ', np.array(upsample_tf).shape, upsample_pt.shape)
print('upsample: ', np.array(upsample_tf)[0,100,200,0], upsample_pt[0,0,100,200])
print('upsample: ', np.array(upsample_tf).mean(), upsample_pt.mean())

gelu_tf = tf.nn.gelu(input_tf)
gelu_pt = F.gelu(input_pt)
print('gelu: ', np.array(gelu_tf).shape, gelu_pt.shape)
print('gelu: ', np.array(gelu_tf)[0,100,200,0], gelu_pt[0,0,100,200])
print('gelu: ', np.array(gelu_tf).mean(), gelu_pt.mean())


ap_tf = layers.GlobalAveragePooling2D()(input_tf)
ap_tf = layers.Reshape((1, 1, ap_tf.shape[-1]))(ap_tf)
ap_pt = nn.AdaptiveAvgPool2d(1)(input_pt)
print('ap: ', np.array(ap_tf).shape, ap_pt.shape)
print('ap: ', np.array(ap_tf), ap_pt)
print('ap: ', np.array(ap_tf).mean(), ap_pt.mean())