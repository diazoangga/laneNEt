"""
Implement VGG16 based fcn net for semantic segmentation
"""
import collections
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU, Add, Conv2DTranspose, Dropout, MaxPool2D, Softmax, Layer, Input
from tensorflow.keras import Model
#from local_utils.config_utils import parse_config_utils

class VGG16Conv(Model):
    def __init__(self, kernel_size, out_channel, name,
                stride=1, pad='SAME', need_layer_norm=True,
                use_bias=True, k_init='glorot_uniform', b_init='zeros'):
        super(VGG16Conv, self).__init__()
        self.k_size = kernel_size
        self.out_channel = out_channel
        self.stride = stride
        self.padding = pad.upper()
        self.need_layer_norm = need_layer_norm
        self.use_bias = use_bias
        self.k_init = k_init
        self.b_init = b_init

        self.conv = Conv2D(self.out_channel, self.k_size, strides=self.stride,
                            padding=self.padding, use_bias=self.use_bias, kernel_initializer=self.k_init,
                            bias_initializer=self.b_init, name=name+'conv')
        self.relu = ReLU(name=name+'relu')
        self.bn = BatchNormalization(name=name+'bn')
    
    def call(self, input_tensor, training=False):
        conv = self.conv(input_tensor, training=training)
        if self.need_layer_norm:
            bnorm = self.bn(conv, training=training)
            out = self.relu(bnorm)
        else:
            out = self.relu(conv)
        
        return out

class VGG16FCNEncode(Model):
    def __init__(self):
        super(VGG16FCNEncode, self).__init__()
        self.encode_result = collections.OrderedDict()
        # encode stage 1
        self.conv_1_1 = VGG16Conv(
            3, 64, name='conv1_1',
            need_layer_norm=True
        )
        self.conv_1_2 = VGG16Conv(
            3, 64, name='conv1_2',
            need_layer_norm=True
        )
        
        # encode stage 2
        self.pool1 = MaxPool2D(2,
            strides=2, name='pool1'
        )
        self.conv_2_1 = VGG16Conv(
            3, 128, name='conv2_1',
            need_layer_norm=True
        )
        self.conv_2_2 = VGG16Conv(
            3, 128, name='conv2_2',
            need_layer_norm=True
        )

        # encode stage 3
        self.pool2 = MaxPool2D(2,
            strides=2, name='pool2'
        )
        self.conv_3_1 = VGG16Conv(
            3, 256, name='conv3_1',
            need_layer_norm=True
        )
        self.conv_3_2 = VGG16Conv(
            3, 256, name='conv3_2',
            need_layer_norm=True
        )
        self.conv_3_3 = VGG16Conv(
            3, 256, name='conv3_3',
            need_layer_norm=True
        )

        # encode stage 4
        self.pool3 = MaxPool2D(2,
            strides=2, name='pool3'
        )
        self.conv_4_1 = VGG16Conv(
            3, 512, name='conv4_1',
            need_layer_norm=True
        )
        self.conv_4_2 = VGG16Conv(
            3, 512, name='conv4_2',
            need_layer_norm=True
        )
        self.conv_4_3 = VGG16Conv(
            3, 512, name='conv4_3',
            need_layer_norm=True
        )

        # encode stage 5 for binary segmentation
        self.pool4 = MaxPool2D(2,
            strides=2, name='pool4'
        )
        self.conv_5_1_binary = VGG16Conv(
            3, 512, name='conv5_1_binary',
            need_layer_norm=True
        )
        self.conv_5_2_binary = VGG16Conv(
            3, 512, name='conv5_2_binary',
            need_layer_norm=True
        )
        self.conv_5_3_binary = VGG16Conv(
            3, 512, name='conv5_3_binary',
            need_layer_norm=True
        )

        # encode stage 5 for instance segmentation
        self.conv_5_1_instance = VGG16Conv(
            3, 512, name='conv5_1_instance',
            need_layer_norm=True
        )
        self.conv_5_2_instance = VGG16Conv(
            3, 512, name='conv5_2_instance',
            need_layer_norm=True
        )
        self.conv_5_3_instance = VGG16Conv(
            3, 512, name='conv5_3_instance',
            need_layer_norm=True
        )

    def call(self, input_tensor, training=False):
        #input_tensor = Input(in_shape)
        shared_net = self.conv_1_1(input_tensor, training=training)
        shared_net = self.conv_1_2(shared_net, training=training)
        
        self.encode_result['encode_stage_1_share'] = {
            'data': shared_net,
            'shape': shared_net.get_shape().as_list()
        }
        #print(self.encode_result['encode_stage_1_share'])

        shared_net = self.pool1(shared_net)
        shared_net = self.conv_2_1(shared_net, training=training)
        shared_net = self.conv_2_2(shared_net, training=training)
        self.encode_result['encode_stage_2_share'] = {
            'data': shared_net,
            'shape': shared_net.get_shape().as_list()
        }

        shared_net = self.pool2(shared_net)
        shared_net = self.conv_3_1(shared_net, training=training)
        shared_net = self.conv_3_2(shared_net, training=training)
        shared_net = self.conv_3_3(shared_net, training=training)
        self.encode_result['encode_stage_3_share'] = {
            'data': shared_net,
            'shape': shared_net.get_shape().as_list()
        }

        shared_net = self.pool3(shared_net)
        shared_net = self.conv_4_1(shared_net, training=training)
        shared_net = self.conv_4_2(shared_net, training=training)
        shared_net = self.conv_4_3(shared_net, training=training)
        self.encode_result['encode_stage_4_share'] = {
            'data': shared_net,
            'shape': shared_net.get_shape().as_list()
        }

        bin_out = self.pool4(shared_net)
        bin_out = self.conv_5_1_binary(bin_out, training=training)
        bin_out = self.conv_5_2_binary(bin_out, training=training)
        bin_out = self.conv_5_3_binary(bin_out, training=training)
        self.encode_result['encode_stage_5_binary'] = {
            'data': bin_out,
            'shape': bin_out.get_shape().as_list()
        }

        inst_out = self.pool4(shared_net)
        inst_out = self.conv_5_1_instance(inst_out, training=training)
        inst_out = self.conv_5_2_instance(inst_out, training=training)
        inst_out = self.conv_5_3_instance(inst_out, training=training)
        self.encode_result['encode_stage_5_instance'] = {
            'data': inst_out,
            'shape': inst_out.get_shape().as_list()
        }
        #print(bin_out.get_shape().as_list())
        return self.encode_result

class VGG16Deconv(Model):
    def __init__(self, out_channel, kernel_size, name=None, need_activate=True,
                stride=2, is_training=False, use_bias=True, 
                k_init='glorot_uniform', b_init='zeros'):
        super(VGG16Deconv, self).__init__()
        self.need_activate = need_activate
        self.is_training = is_training

        self.deconv2d = Conv2DTranspose(out_channel, kernel_size, strides=stride,
                                        use_bias=use_bias,kernel_initializer=k_init,padding='same',
                                        bias_initializer=b_init, name=name+'deconv')
        self.bn = BatchNormalization(name=name+'bn')
        self.relu = ReLU(name=name+'relu')

    def call(self, input_tensor, feat_tensor, training=False):
        deconv = self.deconv2d(input_tensor, training=training)
        deconv = self.bn(deconv, training=training)
        deconv = self.relu(deconv)
        #print(deconv.get_shape().as_list(), feat_tensor.get_shape().as_list())
        add_feats = Add()([deconv, feat_tensor])
        if self.need_activate:
            add_feats = self.bn(add_feats, training=training)
            add_feats = self.relu(add_feats)
        return add_feats



class VGG16FCNDecode(Model):
    def __init__(self, num_class):
        super(VGG16FCNDecode, self).__init__()
        self.decoder_result = collections.OrderedDict()
        self.decode_stage_4_bin = VGG16Deconv(512, 4, name='decode_stage_4_bin')
        self.decode_stage_3_bin = VGG16Deconv(256, 4, name='decode_stage_3_bin')
        self.decode_stage_2_bin = VGG16Deconv(128, 4, name='decode_stage_2_bin')
        self.decode_stage_1_bin = VGG16Deconv(64, 4, name='decode_stage_1_bin')
        self.binary_final_logits = Conv2D(num_class, 1, use_bias=False, name='binary_final_logits')

        self.decode_stage_4_inst = VGG16Deconv(512, 4, name='decode_stage_4_inst')
        self.decode_stage_3_inst = VGG16Deconv(256, 4, name='decode_stage_3_inst')
        self.decode_stage_2_inst = VGG16Deconv(128, 4, name='decode_stage_2_inst')
        self.decode_stage_1_inst = VGG16Deconv(64, 4, name='decode_stage_1_inst')

    def call(self, encoder_tensors, training=False):
        # Binary Path
        decode_stage_5_bin = encoder_tensors['encode_stage_5_binary']['data']
        bin_seg = self.decode_stage_4_bin(decode_stage_5_bin, encoder_tensors['encode_stage_4_share']['data'], training=training)
        bin_seg = self.decode_stage_3_bin(bin_seg, encoder_tensors['encode_stage_3_share']['data'], training=training)
        bin_seg = self.decode_stage_2_bin(bin_seg, encoder_tensors['encode_stage_2_share']['data'], training=training)
        bin_seg = self.decode_stage_1_bin(bin_seg, encoder_tensors['encode_stage_1_share']['data'], training=training)
        bin_seg = self.binary_final_logits(bin_seg, training=training)

        self.decoder_result['binary_segment_logits']= {
            'data': bin_seg,
            'shape': bin_seg.get_shape().as_list()
        }

        decode_stage_5_inst = encoder_tensors['encode_stage_5_instance']['data']
        inst_seg = self.decode_stage_4_inst(decode_stage_5_inst, encoder_tensors['encode_stage_4_share']['data'], training=training)
        inst_seg = self.decode_stage_3_inst(inst_seg, encoder_tensors['encode_stage_3_share']['data'], training=training)
        inst_seg = self.decode_stage_2_inst(inst_seg, encoder_tensors['encode_stage_2_share']['data'], training=training)
        inst_seg = self.decode_stage_1_inst(inst_seg, encoder_tensors['encode_stage_1_share']['data'], training=training)

        self.decoder_result['instance_segment_logits']= {
            'data': inst_seg,
            'shape': inst_seg.get_shape().as_list()
        }

        return self.decoder_result

class VGG16FCN(Model):

    def __init__(self, mode, cfg):
        super(VGG16FCN, self).__init__()
        self._cfg = cfg
        self._mode = mode
        self._is_training = self._is_net_training()
        self._class_num = self._cfg.DATASET.NUM_CLASSES
        self.net_results =collections.OrderedDict()
        
        if self._cfg.MODEL.FRONT_END == 'vgg':
            self.encoder = VGG16FCNEncode()
            self.decoder = VGG16FCNDecode(self._class_num)
        else:
            raise ValueError('Model has to be vgg')
        
    def call(self, input_tensors, training=False):
        #print(input_tensors)
        encoder = self.encoder(input_tensors, training=training)
        decoder = self.decoder(encoder, training=training)

        return encoder, decoder

    def _is_net_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._mode, tf.Tensor):
            mode = self._mode
        else:
            mode = tf.constant(self._mode, dtype=tf.string)

        return tf.equal(mode, tf.constant('train', dtype=tf.string))
    

# if __name__ == '__main__':
#     cfg = parse_config_utils.lanenet_cfg
#     net = VGG16FCN('train', cfg)
#     inp = tf.ones([1,256,512,3], dtype=tf.float32)
#     #print(inp.get_shape())
#     enc, dec = net(inp)

#     for i in enc:
#         print(i, enc[i]['shape'])

#     for i in dec:
#         print(i, dec[i]['shape'])