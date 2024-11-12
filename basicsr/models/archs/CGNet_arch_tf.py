import tensorflow as tf
from tensorflow.keras import layers, models

# Define the custom operations and layers in TensorFlow

class LayerNorm2D(tf.keras.layers.Layer):
    def __init__(self, channels, epsilon=1e-6, **kwargs):
        super(LayerNorm2D, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = self.add_weight(shape=(channels,), initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(channels,), initializer='zeros', trainable=True)

    def call(self, x):
        # Compute mean and variance
        mean, variance = tf.nn.moments(x, axes=3, keepdims=True)
        # Normalize
        x_norm = (x - mean) / tf.sqrt(variance + self.epsilon)
        # Scale and shift
        return self.gamma[None, None, None, :] * x_norm + self.beta[None, None, None, :]

class PixelShuffleLayer(layers.Layer):
    def __init__(self, upscale_factor):
        super(PixelShuffleLayer, self).__init__()
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.upscale_factor)

class SimpleGate(layers.Layer):
    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=3)
        return x1 * x2
    
class DepthwiseSeparableConv(layers.Layer):
    def __init__(self, nin, nout, kernel_size=3, padding='valid', stride=1, use_bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias)
        self.pointwise = layers.Conv2D(nout, kernel_size=1, padding='valid', use_bias=use_bias)

    def call(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class UpsampleWithFlops(layers.Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(UpsampleWithFlops, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.flops = 0

    def call(self, inputs):
        self.flops += tf.size(inputs)
        return tf.image.resize(inputs, size=self.size, method=self.mode)

class GlobalContextExtractor(layers.Layer):
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding='valid', use_bias=False):
        super(GlobalContextExtractor, self).__init__()
        self.convs = [
            DepthwiseSeparableConv(c, c, kernel_size, padding, stride, use_bias)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ]

    def call(self, x):
        outputs = []
        for conv in self.convs:
            x = tf.nn.gelu(conv(x))
            outputs.append(x)
        return outputs

class CascadedGazeBlock(layers.Layer):
    def __init__(self, c, GCE_Conv=2, DW_Expand=2, FFN_Expand=2, dropout_rate=0):
        super(CascadedGazeBlock, self).__init__()
        self.dw_channel = c * DW_Expand
        self.GCE_Conv = GCE_Conv
        self.conv1 = layers.Conv2D(self.dw_channel, kernel_size=1, padding='valid', use_bias=True)
        self.conv2 = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=True)
        # self.conv2 = DepthwiseSeparableConv(self.dw_channel, self.dw_channel, kernel_size=3, padding='same', stride=1, use_bias=True)

        if self.GCE_Conv == 3:
            self.GCE = GlobalContextExtractor(c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4])
            self.project_out = layers.Conv2D(c, kernel_size=1)
            self.sca = models.Sequential([
                layers.GlobalAveragePooling2D(),
                layers.Reshape((1, 1, int(self.dw_channel * 2.5))),
                layers.Conv2D(self.dw_channel * 2.5, kernel_size=1, padding='valid')
            ])
        else:
            self.GCE = GlobalContextExtractor(c, kernel_sizes=[3, 3], strides=[2, 3])
            self.project_out = layers.Conv2D(c, kernel_size=1)
            self.sca = models.Sequential([
                layers.GlobalAveragePooling2D(),
                layers.Reshape((1, 1, self.dw_channel * 2)),
                layers.Conv2D(self.dw_channel * 2, kernel_size=1, padding='valid')
            ])

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = layers.Conv2D(ffn_channel, kernel_size=1, padding='valid', use_bias=True)
        self.conv5 = layers.Conv2D(c, kernel_size=1, padding='valid', use_bias=True)

        self.norm1 = LayerNorm2D(c)
        self.norm2 = LayerNorm2D(c)

        self.dropout1 = layers.Dropout(dropout_rate) if dropout_rate > 0. else layers.Layer()
        self.dropout2 = layers.Dropout(dropout_rate) if dropout_rate > 0. else layers.Layer()

        self.beta = self.add_weight(shape=(1, 1, 1, c), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(1, 1, 1, c), initializer='zeros', trainable=True)

    def call(self, x):

        b, h, w, c = x.shape
        self.upsample = UpsampleWithFlops(size=(h,w), mode='nearest')
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.nn.gelu(x)

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=3)
        if self.GCE_Conv == 3:
            x1, x2, x3 = self.GCE(x1 + x2)
            x = tf.concat([x, self.upsample(x1), self.upsample(x2), self.upsample(x3)], axis=3)
        else:
            x1, x2 = self.GCE(x1 + x2)
            x = tf.concat([x, self.upsample(x1), self.upsample(x2)], axis=3)
        x = self.sca(x) * x
        x = self.project_out(x)

        x = self.dropout1(x)
        y = x + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class NAFBlock0(layers.Layer):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dropout_rate=0.0):
        super(NAFBlock0, self).__init__()
        dw_channel = c * DW_Expand
        self.conv1 = layers.Conv2D(dw_channel, kernel_size=1, padding='valid', use_bias=True)
        self.conv2 = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=True)
        self.conv3 = layers.Conv2D(c, kernel_size=1, padding='valid', use_bias=True)

        self.sca = models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, dw_channel // 2)),
            layers.Conv2D(dw_channel // 2, kernel_size=1, padding='valid')
        ])

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = layers.Conv2D(ffn_channel, kernel_size=1, padding='valid', use_bias=True)
        self.conv5 = layers.Conv2D(c, kernel_size=1, padding='valid', use_bias=True)

        self.norm1 = LayerNorm2D(c)
        self.norm2 = LayerNorm2D(c)

        self.dropout1 = layers.Dropout(dropout_rate) if dropout_rate > 0. else layers.Layer()
        self.dropout2 = layers.Dropout(dropout_rate) if dropout_rate > 0. else layers.Layer()

        self.beta = self.add_weight(shape=(1, 1, 1, c), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(1, 1, 1, c), initializer='zeros', trainable=True)

    def call(self, x):
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class CascadedGaze(models.Model):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], GCE_CONVS_nums=[]):
        super(CascadedGaze, self).__init__()
        self.intro = layers.Conv2D(width, kernel_size=3, padding='same', use_bias=True)
        self.ending = layers.Conv2D(img_channel, kernel_size=3, padding='same', use_bias=True)

        self.encoders = []
        self.decoders = []
        self.middle_blks = []
        self.ups = []
        self.downs = []

        chan = width
        for i in range(len(enc_blk_nums)):
            num = enc_blk_nums[i]
            GCE_Convs = GCE_CONVS_nums[i]
            self.encoders.append(
                [CascadedGazeBlock(chan, GCE_Conv=GCE_Convs) for _ in range(num)]
            )
            self.downs.append(
                layers.Conv2D(chan * 2, kernel_size=2, strides=2, padding='valid')
            )
            chan = chan * 2

        self.middle_blks = [NAFBlock0(chan) for _ in range(middle_blk_num)]

        for i in range(len(dec_blk_nums)):
            num = dec_blk_nums[i]
            self.ups.append(
                models.Sequential([
                    layers.Conv2D(chan * 2, kernel_size=1, padding='valid', use_bias=False),
                    PixelShuffleLayer(upscale_factor=2)
                ])
            )
            chan = chan // 2
            self.decoders.append(
                [NAFBlock0(chan) for _ in range(num)]
            )

        self.padder_size = 2 ** len(self.encoders)

    def call(self, x):
        B, H, W, C = x.shape
        x = self.check_image_size(x)
        inp = x
        x = self.intro(x)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            for enc in encoder:
                x = enc(x)
            encs.append(x)
            x = down(x)

        for blk in self.middle_blks:
            x = blk(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            for dec in decoder:
                x = dec(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :H, :W, :]

    def check_image_size(self, x):
        _, h, w, _ = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = tf.pad(x, [[0, 0], [0, mod_pad_h], [0, mod_pad_w], [0, 0]])
        return x

def get_loss_l1(pred: tf.Tensor, label: tf.Tensor, norm_k: tf.Tensor) -> tf.Tensor:
    B = tf.shape(pred)[0]
    L1 = tf.reduce_mean(tf.abs(pred - label), axis=[1, 2, 3])
    L1 = L1 / tf.reshape(norm_k, [B])
    return tf.reduce_mean(L1)

# Example usage
if __name__ == '__main__':
    img_channel = 4
    width = 8
    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 10
    dec_blks = [2, 2, 2, 2]
    GCE_CONVS_nums = [3, 3, 2, 2]

    model = CascadedGaze(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, GCE_CONVS_nums=GCE_CONVS_nums)

    inp_shape = (1, 540, 960, img_channel)
    data = tf.random.normal(inp_shape)
    def get_flops(model, inputs):
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        real_model = tf.function(model).get_concrete_function(inputs)
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
        flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, 
                                              run_meta=run_meta, 
                                              cmd="scope", 
                                              options=opts)
        return flops.total_float_ops
    
    # outputs = model(inputs)
    
    inputs = tf.TensorSpec((1, 544, 960, 4), tf.float32, name="input")
    flops = get_flops(model, inputs)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    # Forward pass
    # output = model(data)
    # print(output.shape)