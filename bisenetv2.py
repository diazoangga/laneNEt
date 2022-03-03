from selectors import DefaultSelector
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Add, Conv2D, Input, DepthwiseConv2D, BatchNormalization, ReLU, Layer, MaxPool2D
from tensorflow.keras import Model

class BiseNetV2(Model):
    def __init__(self):
        super(BiseNetV2, self).__init__()
        self.detail_branch = DetailBranch(name='detail_branch')
        self.semantic_branch = SemanticBranch(prepare_data_for_booster=False,
                                                    name='semantic_branch')
        self.aggregation_branch = AggregationBranch(name='aggregation_branch')
    
    def call(self, input_tensors):
        pass


class ConvBlock(Layer):
    def __init__(self, out_ch, kernel_size, strides, padding, activation=True):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv2d = Conv2D(out_ch, kernel_size, strides, padding)
        self.bn = BatchNormalization()
        self.relu = ReLU()
    def call(self, input_tensors):
        conv = self.conv2d(input_tensors)
        bn = self.bn(conv)
        if self.activation:
            out = self.relu(bn)
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
    def call(self, input_tensor):
        out = self.dw_conv(input_tensor)
        out = self.bn(out)

        return out

class ContextEmbedding(Layer):
    def __init__(self, out_ch):
        super(ContextEmbedding, self).__init__()
        self.ga_pool = GlobalAveragePooling2D(data_format='channels_last')
        self.ga_pool_bn = BatchNormalization()
        self.conv_1 = ConvBlock(out_ch, 1, strides=1, padding='SAME')
        self.conv_2 = Conv2D(out_ch, 3, strides=3, padding='SAME')

    def call(self, input_tensor):
        out = self.ga_pool(input_tensor)
        out = self.ga_pool_bn(out)
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
    
    def call(self, input_tensor):
        out = input_tensor
        layer = sorted(self.layer)
        for item in layer:
            out = self.layer[item](out)
        return out

class SemanticBranch(Layer):
    def __init__(self):
        arch = {
            'stage_1': [['stem',3, 16, 0, 4, 1]],
            'stage_3': [['ge', 3, 32, 6, 2, 1], ['ge', 3, 32, 6, 1, 1]],
            'stage_4': [['ge', 3, 64, 6, 2, 1], ['ge', 3, 64, 6, 1, 1]],
            'stage_5': [['ge', 3, 128, 6, 2, 1], ['ge', 3, 128, 6, 1, 3], ['ce', 128, 0, 1, 1]]
        }
        self.layer = {}
        stage = sorted(arch)
        
    def call(self):
        pass

class AggregationBranch(Layer):
    def __init__(self):
        pass
    def call(self):
        pass


inputs = Input([64,128,32])
m = GatherExpansion(32, 32, 1)
out = m(inputs)
model = Model(inputs, out)
model.summary()