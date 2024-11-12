import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, Conv2DTranspose, Add, Input
from tensorflow.keras.models import Model
import tf2onnx


def conv2d_layer(in_channels, out_channels, kernel_size, stride, padding, is_seperable=False, has_relu=False):
    layers = []
    if is_seperable:
        layers.append(DepthwiseConv2D(kernel_size, strides=stride, padding=padding, depth_multiplier=1, use_bias=False))
        layers.append(Conv2D(out_channels, kernel_size=1, strides=1, padding='valid', use_bias=True))
    else:
        layers.append(Conv2D(out_channels, kernel_size, strides=stride, padding=padding, use_bias=True))
    if has_relu:
        layers.append(ReLU())
    return tf.keras.Sequential(layers)

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(EncoderBlock, self).__init__()
        self.conv1 = conv2d_layer(in_channels, mid_channels, 5, stride, 'same', is_seperable=True, has_relu=True)
        self.conv2 = conv2d_layer(mid_channels, out_channels, 5, 1, 'same', is_seperable=True, has_relu=False)
        self.proj = (tf.keras.layers.Lambda(lambda x: x) if stride == 1 and in_channels == out_channels
                     else conv2d_layer(in_channels, out_channels, 3, stride, 'same', is_seperable=True, has_relu=False))
        self.relu = ReLU()

    def call(self, x):
        proj = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = Add()([x, proj])
        return self.relu(x)

def encoder_stage(in_channels, out_channels, num_blocks):
    layers = [EncoderBlock(in_channels, out_channels // 4, out_channels, stride=2)]
    for _ in range(num_blocks - 1):
        layers.append(EncoderBlock(out_channels, out_channels // 4, out_channels, stride=1))
    return tf.keras.Sequential(layers)

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DecoderBlock, self).__init__()
        padding = 'same'
        self.conv0 = conv2d_layer(in_channels, out_channels, kernel_size, 1, padding, is_seperable=True, has_relu=True)
        self.conv1 = conv2d_layer(out_channels, out_channels, kernel_size, 1, padding, is_seperable=True, has_relu=False)

    def call(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = Add()([x, inp])
        return x

class DecoderStage(tf.keras.layers.Layer):
    def __init__(self, in_channels, skip_in_channels, out_channels):
        super(DecoderStage, self).__init__()
        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3)
        self.upsample = Conv2DTranspose(out_channels, kernel_size=2, strides=2, padding='valid')
        self.proj_conv = conv2d_layer(skip_in_channels, out_channels, 3, 1, 'same', is_seperable=True, has_relu=True)

    def call(self, inp, skip):
        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return Add()([x, y])

class Network(tf.keras.Model):
    def __init__(self, model_mul=4):
        super(Network, self).__init__()
        self.conv0 = conv2d_layer(4, 16//model_mul, 3, 1, 'same', is_seperable=False, has_relu=True)
        self.enc1 = encoder_stage(16//model_mul, 64//model_mul, 2)
        self.enc2 = encoder_stage(64//model_mul, 128//model_mul, 2)
        self.enc3 = encoder_stage(128//model_mul, 256//model_mul, 4)
        self.enc4 = encoder_stage(256//model_mul, 512//model_mul, 4)
        
        self.encdec = conv2d_layer(512//model_mul, 64//model_mul, 3, 1, 'same', is_seperable=True, has_relu=True)
        self.dec1 = DecoderStage(64//model_mul, 256//model_mul, 64//model_mul)
        self.dec2 = DecoderStage(64//model_mul, 128//model_mul, 32//model_mul)
        self.dec3 = DecoderStage(32//model_mul, 64//model_mul, 32//model_mul)
        self.dec4 = DecoderStage(32//model_mul, 16//model_mul, 16//model_mul)
        
        self.out0 = DecoderBlock(16//model_mul, 16//model_mul, kernel_size=3)
        self.out1 = conv2d_layer(16//model_mul, 4, 3, 1, 'same', is_seperable=False, has_relu=False)

    def call(self, inp):
        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)
        conv5 = self.encdec(conv4)
        up3 = self.dec1(conv5, conv3)
        up2 = self.dec2(up3, conv2)
        up1 = self.dec3(up2, conv1)
        x = self.dec4(up1, conv0)
        x = self.out0(x)
        x = self.out1(x)
        pred = inp + x
        return pred
    
def get_loss_l1(pred: tf.Tensor, label: tf.Tensor, norm_k: tf.Tensor) -> tf.Tensor:
    # print(f'get loss l1: {pred.dtype} {label.dtype} {pred.shape} {label.shape}')
    B = tf.shape(pred)[0]
    L1 = tf.reduce_mean(tf.abs(pred - label), axis=[1, 2, 3])
    L1 = L1 / tf.reshape(norm_k, [B])
    return tf.reduce_mean(L1)

if __name__ == "__main__":
    # Initialize the model
    model = Network(4)

    # Build the model by passing a dummy input
    dummy_input = Input(shape=(544, 960, 4))
    _ = model(dummy_input)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # # Summary of the model
    # model.summary()
    # # Convert the model to ONNX
    inputs = (tf.TensorSpec((1, 544, 960, 4), tf.float32, name="input"),)
    output_path = "checkpoints/modelx4_20240903.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=inputs, output_path=output_path)


    # inputs = (tf.TensorSpec((1, 544, 960, 4), tf.float32, name="input"),)
    # output_path = "checkpoints/modelx8_upsampling_conv1x1.onnx"
    # model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)


    # tf.compat.v1.disable_eager_execution()

    # inputs = tf.random.normal([1, 544, 960, 4])

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
    
    flops = get_flops(model, inputs)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")



    # net = Network()
    # # img = mge.tensor(np.random.randn(1, 4, 64, 64).astype(np.float32))
    # img = torch.randn(1, 4, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
    # net.eval()
    # dummy_input = torch.randn((1, 4, 544, 960), requires_grad=True)
    # torch.onnx.export(net,         # model being run 
    #     dummy_input,       # model input (or a tuple for multiple inputs) 
    #     "pmrid.onnx",       # where to save the model  
    #     export_params=True,  # store the trained parameter weights inside the model file 
    #     opset_version=10,    # the ONNX version to export the model to 
    #     do_constant_folding=True,  # whether to execute constant folding for optimization 
    #     input_names = ['modelInput'],   # the model's input names 
    #     output_names = ['modelOutput']) # the model's output names 
    #     # dynamic_axes={'modelInput' : {0 : 1},    # variable length axes 
    #     #                     'modelOutput' : {0 : 1}}) 
    # out = net(img)
    # # flops, macs, params = calculate_flops(model=net, 
    # #                                   input_shape=(1, 4, 544, 960),
    # #                                   output_as_string=True,
    # #                                   output_precision=4)
    # # print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    # torch.save(net, 'pmrid.pth')
    # scripted_model = torch.jit.script(net)
    # scripted_model.save('pmrid.pt')
    # # import IPython; IPython.embed()

# vim: ts=4 sw=4 sts=4 expandtab
