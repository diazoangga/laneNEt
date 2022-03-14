from selectors import DefaultSelector
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, GlobalAveragePooling2D, AveragePooling2D, Add, Dropout, Conv2D, Input, DepthwiseConv2D, BatchNormalization, ReLU, Layer, MaxPool2D
from tensorflow.keras import Model

tf.random.set_seed(40)

dropout = 0.2
class BiseNetV2(Model):
    def __init__(self):
        super(BiseNetV2, self).__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation_branch = AggregationBranch(128)
        self.bin_segmentation = BinarySegmentation(2)
        self.inst_segmentation = InstanceSegmentation(4)
    
    def call(self, input_tensor):
        d_branch = self.detail_branch(input_tensor)
        s_branch = self.semantic_branch(input_tensor)
        agg_brang = self.aggregation_branch(d_branch, s_branch)
        
        bin_pred = self.bin_segmentation(agg_brang)
        inst_seg = self.inst_segmentation(agg_brang)
        #print(s_branch)
        
        return [bin_pred, inst_seg]

    def get_config(self):
        cfg = super(BiseNetV2, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg  

    def build_model(self, in_shape):
        input_layer = Input(shape=in_shape)
        out_layer = BiseNetV2()(input_layer)

        return Model(input_layer, out_layer)

class ConvBlock(Layer):
    def __init__(self, out_ch, kernel_size, strides, padding, activation=True):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv2d = Conv2D(out_ch, kernel_size, strides, padding)
        self.bn = BatchNormalization()
        if activation:
            self.relu = ReLU()
            self.dropout = Dropout(dropout)

    def get_config(self):
        cfg = super(ConvBlock, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensors):
        conv = self.conv2d(input_tensors)
        bn = self.bn(conv)
        if self.activation:
            out = self.relu(bn)
            out = self.dropout(out)
        else:
            out = bn
        return out

class StemBlock(Layer):
    def __init__(self, out_ch):
        super(StemBlock, self).__init__()
        self.conv_0_1 = ConvBlock(out_ch, 3, 2, padding='SAME')
        self.conv_1_1 = ConvBlock(out_ch/2, 1, 1, padding='SAME')
        self.conv_1_2 = ConvBlock(out_ch, 3, 2, padding='SAME')
        self.mpool_2_0 = MaxPool2D(pool_size=(3,3), strides=2, padding='SAME')
        self.conv_0_2 = ConvBlock(out_ch, 3, 1, padding='SAME')

    def get_config(self):
        cfg = super(StemBlock, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensor):
        share = self.conv_0_1(input_tensor)
        conv_branch = self.conv_1_1(share)
        conv_branch = self.conv_1_2(conv_branch)
        mpool_branch = self.mpool_2_0(share)
        out = tf.concat([conv_branch, mpool_branch], axis=-1)
        out = self.conv_0_2(out)

        return out

class GatherExpansion(Layer):
    def __init__(self, in_ch, out_ch, strides):
        super(GatherExpansion, self).__init__()
        self.strides = strides
        if self.strides == 1:
            self.stride1_conv_1 = ConvBlock(out_ch, 3, 1, padding='SAME')
            self.stride1_dwconv_1 = DWBlock(3, strides=1, d_multiplier=6,
                                    padding='SAME')
            self.stride1_conv_2 = ConvBlock(out_ch, 1, 1, padding='SAME', activation=False)  
        
        if self.strides == 2:
            self.stride2_main_dw = DWBlock(3, strides=2, d_multiplier=1, padding='SAME')
            self.stride2_main_conv = ConvBlock(out_ch, 1, 1, padding='SAME', activation=False)
            self.stride2_sub_conv_1 = ConvBlock(in_ch, 3, 1, padding='SAME')
            self.stride2_sub_dw_1 = DWBlock(3, strides=2, d_multiplier=6, padding='SAME')
            self.stride2_sub_dw_2 = DWBlock(3, strides=1, d_multiplier=1, padding='SAME')
            self.stride2_sub_conv_2 = ConvBlock(out_ch, 3, 1, padding='SAME', activation=False)
    
    def get_config(self):
        cfg = super(GatherExpansion, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensor):
        if self.strides == 1:
            branch = self.stride1_conv_1(input_tensor)
            branch = self.stride1_dwconv_1(branch)
            branch = self.stride1_conv_2(branch)
            
            out = Add()([branch, input_tensor])
            out = ReLU()(out)

        if self.strides == 2:
            branch = self.stride2_main_dw(input_tensor)
            branch = self.stride2_main_conv(branch)

            main = self.stride2_sub_conv_1(input_tensor)
            main = self.stride2_sub_dw_1(main)
            main = self.stride2_sub_dw_2(main)
            main = self.stride2_sub_conv_2(main)

            out = Add()([main, branch])
            out = ReLU()(out)
        
        return out

class DWBlock(Layer):
    def __init__(self, k_size, strides, d_multiplier, padding='SAME'):
        super(DWBlock, self).__init__()
        self.dw_conv = DepthwiseConv2D(k_size, strides=strides, depth_multiplier=d_multiplier,
                                        padding=padding)
        self.bn = BatchNormalization()
    
    def get_config(self):
        cfg = super(DWBlock, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensor):
        out = self.dw_conv(input_tensor)
        out = self.bn(out)

        return out

class ContextEmbedding(Layer):
    def __init__(self, out_ch):
        super(ContextEmbedding, self).__init__()
        self.ga_pool = GlobalAveragePooling2D(keepdims=True)
        self.ga_pool_bn = BatchNormalization()
        self.conv_1 = ConvBlock(out_ch, 1, strides=1, padding='SAME')
        self.conv_2 = Conv2D(out_ch, 3, strides=1, padding='SAME')

    def get_config(self):
        cfg = super(ContextEmbedding, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensor):
        out = self.ga_pool(input_tensor)
        out = self.ga_pool_bn(out)
        #print(out)
        out = self.conv_1(out)
        out = Add()([out, input_tensor])
        out = self.conv_2(out)

        return out


class DetailBranch(Layer):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.arch = {
            'stage_1': [[3, 64, 2, 1], [3, 64, 1, 1]],
            'stage_2': [[3, 64, 2, 1], [3, 64, 1, 2]],
            'stage_3': [[3, 128, 2, 1], [3, 128, 1, 2]]
            }
        self.layer = {}
        stage = sorted(self.arch)
        for stage_idx in stage:
            for idx, info in enumerate(self.arch[stage_idx]):
                #print(globals()[f'{stage_idx}_{idx}_conv'])
                var = info
                k_size = var[0]
                out_ch = var[1]
                strides = var[2]
                repeat = info[3]
                for r in range(repeat):
                    self.layer['self.{}_{}_{}_conv'.format(stage_idx, idx, r)] = ConvBlock(out_ch, k_size, strides, padding='SAME')
    
    def get_config(self):
        cfg = super(DetailBranch, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensor):
        out = input_tensor
        layer = sorted(self.layer)
        for item in layer:
            out = self.layer[item](out)
        return out

class SemanticBranch(Layer):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.arch = {
            'stage_1': [['stem',3, 16, 0, 4, 1]],
            'stage_3': [['ge', 3, 32, 6, 2, 1], ['ge', 3, 32, 6, 1, 1]],
            'stage_4': [['ge', 3, 64, 6, 2, 1], ['ge', 3, 64, 6, 1, 1]],
            'stage_5': [['ge', 3, 128, 6, 2, 1], ['ge', 3, 128, 6, 1, 3], ['ce', 3, 128, 0, 1, 1]]
        }
        self.layer = {}
        stage = sorted(self.arch)
        temp = None
        for stage_idx in stage:
            for idx, info in enumerate(self.arch[stage_idx]):
                #print(globals()[f'{stage_idx}_{idx}_conv'])
                var = info
                opr_type = var[0]
                k_size = var[1]
                out_ch = var[2]
                depth_multi = var[3]
                strides = var[4]
                repeat = info[5]
                for r in range(repeat):
                    #print(temp, out_ch)
                    if opr_type == 'stem':
                        self.layer['self.{}_{}_{}_{}'.format(stage_idx, idx, opr_type, r)] =\
                            StemBlock(out_ch)
                    if opr_type == 'ge':
                        self.layer['self.{}_{}_{}_{}'.format(stage_idx, idx, opr_type, r)] =\
                            GatherExpansion(temp, out_ch, strides)
                    if opr_type == 'ce':
                        self.layer['self.{}_{}_{}_{}'.format(stage_idx, idx, opr_type, r)] =\
                            ContextEmbedding(out_ch)
                    temp = out_ch

    def get_config(self):
        cfg = super(SemanticBranch, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensors):
        out = input_tensors
        layer = sorted(self.layer)
        for item in layer:
            out = self.layer[item](out)
            #print(f'{item}:\n{out.numpy()}')
        return out

class AggregationBranch(Layer):
    def __init__(self, out_ch):
        super(AggregationBranch, self).__init__()
        self.d_branch_1_dw = DWBlock(3, strides=1, d_multiplier=1, padding='SAME')
        self.d_branch_1_conv = Conv2D(out_ch, 1, strides=1, padding='SAME')
        self.d_branch_2_conv = ConvBlock(out_ch, 3, strides=2, padding='SAME', activation=False)
        self.d_branch_2_apool = AveragePooling2D(pool_size=(3,3), strides=2, padding='SAME')
        self.s_branch_1_dw = DWBlock(3, strides=1, d_multiplier=1, padding='SAME')
        self.s_branch_1_conv = Conv2D(out_ch, 1, strides=1, padding='SAME', activation='sigmoid')
        self.s_branch_2_conv = ConvBlock(out_ch, 3, strides=1, padding='SAME', activation=False)
        self.s_branch_2_upsample = UpSampling2D(size=(4,4), interpolation='bilinear')
        self.s_branch_3_upsample = UpSampling2D(size=(4,4), interpolation='bilinear')
        self.main_conv = ConvBlock(out_ch, 3, strides=1, padding='SAME', activation=False)
    
    def get_config(self):
        cfg = super(AggregationBranch, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, detail_branch, semantic_branch):
        d_branch_main = self.d_branch_1_dw(detail_branch)
        d_branch_main = self.d_branch_1_conv(d_branch_main)
        d_branch_sub = self.d_branch_2_conv(detail_branch)
        d_branch_sub = self.d_branch_2_apool(d_branch_sub)
        s_branch_main = self.s_branch_1_dw(semantic_branch)
        s_branch_main = self.s_branch_1_conv(s_branch_main)
        s_branch_sub = self.s_branch_2_conv(semantic_branch)
        s_branch_sub = self.s_branch_2_upsample(s_branch_sub)
        s_branch_sub = tf.keras.activations.sigmoid(s_branch_sub)
        
        d_branch = tf.math.multiply(d_branch_main, s_branch_sub)
        s_branch = tf.math.multiply(s_branch_main, d_branch_sub)
        s_branch = self.s_branch_3_upsample(s_branch)

        out = Add()([d_branch, s_branch])
        out = self.main_conv(out)
        
        return out

class BinarySegmentation(Layer):
    def __init__(self, cls_num):
        super(BinarySegmentation, self).__init__()
        self.conv1 = ConvBlock(128, 1, strides=1, padding='SAME')
        self.conv2 = ConvBlock(64, 1, strides=1, padding='SAME')
        self.conv3 = ConvBlock(cls_num, 1, strides=1, padding='SAME', activation=False)
        
        upsample_size = 8
        self.upsample = UpSampling2D(size=upsample_size, interpolation='bilinear')
        self.softmax = tf.keras.layers.Softmax(axis=-1, name='bin_seg')
    
    def get_config(self):
        cfg = super(BinarySegmentation, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        out = self.softmax(out)
        
        return out


class InstanceSegmentation(Layer):
    def __init__(self, inst_n):
        super(InstanceSegmentation, self).__init__()
        self.conv1 = ConvBlock(128, 1, strides=1, padding='SAME')
        self.conv2 = ConvBlock(64, 1, strides=1, padding='SAME')
        self.conv3 = ConvBlock(inst_n, 1, strides=1, padding='SAME', activation=False)
        
        upsample_size = 8
        self.upsample = UpSampling2D(size=upsample_size, interpolation='bilinear', name='inst_seg')
    
    def get_config(self):
        cfg = super(InstanceSegmentation, self).get_config()
        cfg.update({
            'node_def': self.node_def.SerializeToString(),
            'node_def': self.node_def.SerializeToString().decode('utf-8'),
            'constants': {
                i: tf.keras.backend.get_value(c) for i, c in self.constants.items()
            }
        })
        return cfg

    def call(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        
        return out
# inputs = Input([256, 512, 3])
# m = BiseNetV2()
# out = m(inputs)
# model = Model(inputs=inputs, outputs=out)
# model.summary()
