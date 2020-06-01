from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.python.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D, AveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras import backend as K
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
import numpy as np

# from keras.engine import InputSpec
from tensorflow.python.keras import backend as K
# from keras.applications import imagenet_utils
from tensorflow.python.keras.utils import conv_utils

# from keras.utils.data_utils import get_file

WEIGHTS_PATH_X = "deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.upsample_size = conv_utils.normalize_tuple(
                output_size, 2, 'size')
            self.upsampling = None
        else:
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                     input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.upsample_size[0]
            width = self.upsample_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                               inputs.shape[2] * self.upsampling[1]),
                                                      align_corners=True)
        else:
            return tf.compat.v1.image.resize_bilinear(inputs, (self.upsample_size[0],
                                                               self.upsample_size[1]),
                                                      align_corners=True)

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SepConv_BN(Layer):
    def __init__(self, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        super(SepConv_BN, self).__init__()

        self.stride = stride
        self.depth_activation = depth_activation

        if self.stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            self.zp = ZeroPadding2D((pad_beg, pad_end))
            depth_padding = 'valid'

        if not self.depth_activation:
            self.da = Activation('relu')
        self.dw_conv2d = DepthwiseConv2D((kernel_size, kernel_size), strides=(self.stride, self.stride),
                                         dilation_rate=(rate, rate),
                                         padding=depth_padding, use_bias=False, name=prefix + '_depthwise')
        self.dw_bn = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)
        if self.depth_activation:
            self.dp_2 = Activation('relu')
        self.pointwise = Conv2D(filters, (1, 1), padding='same',
                                use_bias=False, name=prefix + '_pointwise')
        self.pointwise_bn = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)
        if self.depth_activation:
            self.da_2 = Activation('relu')

    def call(self, x):
        # print(self.stride)
        if self.stride != 1:
            x = self.zp(x)
        if not self.depth_activation:
            x = self.da(x)
        x = self.dw_conv2d(x)
        x = self.dw_bn(x)

        if self.depth_activation:
            x = self.dp_2(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)

        if self.depth_activation:
            x = self.da_2(x)
        return x


class conv2d_same(Layer):
    def __init__(self, filters, prefix, stride=1, kernel_size=3, rate=1):
        super(conv2d_same, self).__init__()
        self.stride = stride
        if self.stride == 1:
            self.conv2a = Conv2D(filters,
                                 (kernel_size, kernel_size),
                                 strides=(self.stride, self.stride),
                                 padding='same', use_bias=False,
                                 dilation_rate=(rate, rate),
                                 name=prefix)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            self.zp = ZeroPadding2D((pad_beg, pad_end))
            self.conv2a = Conv2D(filters,
                                 (kernel_size, kernel_size),
                                 strides=(stride, stride),
                                 padding='valid', use_bias=False,
                                 dilation_rate=(rate, rate),
                                 name=prefix)

    def call(self, x):
        if self.stride == 1:
            x = self.conv2a(x)
        else:
            x = self.zp(x)
            x = self.conv2a(x)

        return x


class xception_block(Layer):
    def __init__(self, depth_list, prefix, skip_connection_type, stride,
                 rate=1, depth_activation=False, return_skip=False):
        super(xception_block, self).__init__()

        self.skip_connection_type = skip_connection_type
        self.return_skip = return_skip

        self.residual = []
        for i in range(3):
            self.residual.append(SepConv_BN(
                depth_list[i],
                prefix + '_separable_conv{}'.format(i + 1),
                stride=stride if i == 2 else 1,
                rate=rate,
                depth_activation=depth_activation))
            if i == 1:
                self.skip = self.residual[i]

        if self.skip_connection_type == 'conv':
            self.shortcut = conv2d_same(depth_list[-1], prefix + '_shortcut',
                                        kernel_size=1,
                                        stride=stride)
            self.shortcut_bn = BatchNormalization(name=prefix + '_shortcut_BN')

    def call(self, x):
        residual = x
        for i in range(3):
            residual = self.residual[i](x)
            if i == 1:
                self.skip = self.residual[i](x)
        print("Res before CONV = ", residual)
        if self.skip_connection_type == 'conv':
            shortcut1 = self.shortcut(x)
            shortcut2 = self.shortcut_bn(shortcut1)
            self.outputs = layers.add([residual, shortcut2])
        elif self.skip_connection_type == 'sum':
            self.outputs = layers.add([residual, x])
        elif self.skip_connection_type == 'none':
            print("output = ", residual)
            self.outputs = residual

        if self.return_skip:
            print("skip = ", self.outputs, self.skip)
            return self.outputs, self.skip
        else:
            return self.outputs


class Deeplabv3_plus(Model):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights`
    """

    def __init__(self, weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, OS=8):
        super(Deeplabv3_plus, self).__init__()

        if input_tensor is None:
            self.img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                self.img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                self.img_input = input_tensor

        self.input_tensor = input_tensor
        # OS == 8 params
        self.entry_block3_stride = 1
        self.middle_block_rate = 2  # ! Not mentioned in paper, but required
        self.exit_block_rates = (2, 4)
        self.atrous_rates = (12, 24, 36)

        # Layers
        self.entry_flow_conv1_1 = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same')
        self.entry_flow_conv1_1_BN = BatchNormalization()
        self.relu1 = Activation('relu')

        self.entry_flow_conv1_2 = conv2d_same(64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        self.entry_flow_conv1_2_BN = BatchNormalization()
        self.relu2 = Activation('relu')

        self.entry_flow_block1 = xception_block([128, 128, 128], 'entry_flow_block1',
                                                skip_connection_type='conv', stride=2,
                                                depth_activation=False)

        self.entry_flow_block2 = xception_block([256, 256, 256], 'entry_flow_block2',
                                                skip_connection_type='conv', stride=2,
                                                depth_activation=False, return_skip=True)

        self.entry_flow_block3 = xception_block([728, 728, 728], 'entry_flow_block3',
                                                skip_connection_type='conv', stride=self.entry_block3_stride,
                                                depth_activation=False)

        self.middle_flow_unit = []
        for i in range(16):
            self.middle_flow_unit.append(xception_block([728, 728, 728], 'middle_flow_unit' + str(i),
                                                        skip_connection_type='sum', stride=1,
                                                        rate=self.middle_block_rate,
                                                        depth_activation=False))

        self.exit_flow_block1 = xception_block([728, 1024, 1024], 'middle_flow_unit' + str(i),
                                               skip_connection_type='conv', stride=1, rate=self.exit_block_rates[0],
                                               depth_activation=False)
        self.exit_flow_block2 = xception_block([1536, 1536, 2048], 'middle_flow_unit' + str(i),
                                               skip_connection_type='none', stride=1, rate=self.exit_block_rates[1],
                                               depth_activation=True)
        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling
        # simple 1x1
        self.aspp0 = Conv2D(256, (1, 1), padding='same', use_bias=False)
        self.aspp0_BN = BatchNormalization(epsilon=1e-5)
        self.aspp0_activation = Activation('relu')

        # rate = 6 (12)
        self.aspp1 = SepConv_BN(256, 'aspp1',
                                rate=self.atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        self.aspp2 = SepConv_BN(256, 'aspp2',
                                rate=self.atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        self.aspp3 = SepConv_BN(256, 'aspp3',
                                rate=self.atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # Image Feature branch
        self.out_shape = int(np.ceil(input_shape[0] / OS))
        self.b4_ap = AveragePooling2D(pool_size=(self.out_shape, self.out_shape))
        self.image_pooling = Conv2D(256, (1, 1), padding='same',
                                    use_bias=False)
        self.image_pooling_BN = BatchNormalization(epsilon=1e-5)
        self.relu3 = Activation('relu')
        self.b4_bu = BilinearUpsampling((self.out_shape, self.out_shape))

        # concatenate ASPP branches & project
        self.concat1 = Concatenate()
        self.concat_projection = Conv2D(256, (1, 1), padding='same',
                                        use_bias=False)
        self.concat_projection_BN = BatchNormalization(epsilon=1e-5)
        self.relu4 = Activation('relu')
        self.droput = Dropout(0.1)
        self.activation = Activation("softmax")

        # DeepLab v.3+ decoder

        # Feature projection
        # x4 (x2) block
        self.bu = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                                  int(np.ceil(input_shape[1] / 4))))
        self.feature_projection0 = Conv2D(48, (1, 1), padding='same',
                                          use_bias=False)
        self.feature_projection0_BN = BatchNormalization(epsilon=1e-5)
        self.relu5 = Activation('relu')
        self.concat2 = Concatenate()  # move to call
        self.decoder_conv0 = SepConv_BN(256, 'decoder_conv0',
                                        depth_activation=True, epsilon=1e-5)
        self.decoder_conv1 = SepConv_BN(256, 'decoder_conv1',
                                        depth_activation=True, epsilon=1e-5)

        self.logits1 = Conv2D(classes, (1, 1), padding='same')
        self.logits2 = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))

    def call(self, inputs, training=True):

        if self.input_tensor is not None:
            inputs = get_source_inputs(self.input_tensor)
        else:
            inputs = self.img_input
        if training:
            x = self.entry_flow_conv1_1(inputs)
            x = self.entry_flow_conv1_1_BN(x)
            x = self.relu1(x)

            x = self.entry_flow_conv1_2(x)
            x = self.entry_flow_conv1_2_BN(x)
            x = self.relu2(x)
            x = self.entry_flow_block1(x)
            print("Block 1 x = ", x)
            print("#######################################################################################")
            x, skip1 = self.entry_flow_block2(x)
            print("Block 2 x = ", x)
            print("#######################################################################################")
            x = self.entry_flow_block3(x)
            print("Block 3 x = ", x)
            print("#######################################################################################")
            for i in range(16):
                x = self.middle_flow_unit[i](x)
            print("middle flow complete")
            print("######################################################################################")
            x = self.exit_flow_block1(x)
            print("Exit Flow 1")
            print("#######################################################################################")
            print("X before exit flow = ", x)
            x = self.exit_flow_block2(x)
            print("Exit Flow 2")
            print("#######################################################################################")
            # End: Feature extractor

            # Start: Atrous Pooling
            b0 = self.aspp0(x)
            b0 = self.aspp0_BN(b0)
            b0 = self.aspp0_activation(b0)

            b1 = self.aspp1(x)

            b2 = self.aspp2(x)

            b3 = self.aspp3(x)

            b4 = self.b4_ap(x)
            b4 = self.image_pooling(b4)
            b4 = self.image_pooling_BN(b4)
            b4 = self.relu3(b4)
            b4 = self.b4_bu(b4)

            # ASPP and Project
            x = self.concat1([b4, b0, b1, b2, b3])
            x = self.concat_projection(x)
            x = self.concat_projection_BN(x)
            x = self.relu4(x)
            x = self.droput(x)

            # Feature Projection
            x = self.bu(x)
            dec_skip1 = self.feature_projection0(skip1)
            dec_skip1 = self.feature_projection0_BN(x)
            dec_skip1 = self.relu5(dec_skip1)
            x = self.concat2([x, dec_skip1])
            x = self.decoder_conv0(x)
            x = self.decoder_conv1(x)

            x = self.logits1(x)
            x = self.logits2(x)

            # # Ensure that the model takes into account
            # # any potential predecessors of `input_tensor`.
            # if self.input_tensor is not None:
            #     inputs = get_source_inputs(self.input_tensor)
            # else:
            #     inputs = self.img_input

            # print("inputs = ", inputs)
            # print("x = ", x)
            # model = Model(inputs, x)
            #
            # print("weights_path", WEIGHTS_PATH_X)
            # model.load_weights(WEIGHTS_PATH_X)
        else:
            x = inputs
        return x
