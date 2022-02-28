import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax, BatchNormalization, Conv2D, ReLU

class LaneNetBackEnd(Model):
    def __init__(self, mode, cfg):
        super(LaneNetBackEnd, self).__init__()
        self._cfg = cfg
        self._mode = mode

        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._embedding_dims = self._cfg.MODEL.EMBEDDING_FEATS_DIMS
        self._binary_loss_type = self._cfg.SOLVER.LOSS_TYPE

        self.binary_score = Softmax()
        
        self.conv = Conv2D(self._embedding_dims, 1,
                            use_bias=False,
                            name='pix_embedding_conv')
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, bin_seg, inst_seg, training=False):
        binary_score = self.binary_score(bin_seg)
        binary_pred = tf.math.argmax(binary_score, axis=-1)
        
        pix_bn = self.bn(inst_seg, training=training)
        pix_relu = self.relu(pix_bn)
        inst_pred = self.conv(pix_relu, training=training)

        return binary_pred, inst_pred