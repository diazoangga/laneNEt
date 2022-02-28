import collections
import tensorflow as tf
from tensorflow.keras import Model
from semantic_segmentation_zoo.vgg16_unet import *

# net = VGG16FCN('train', cfg)
#     inp = tf.ones([1,256,512,3], dtype=tf.float32)
#     #print(inp.get_shape())
#     enc, dec = net(inp)

#     for i in enc:
#         print(i, enc[i]['shape'])

#     for i in dec:
#         print(i, dec[i]['shape'])
class LaneNetFrontEnd(Model):

    def __init__(self, mode, cfg, front_end_model):
        super(LaneNetFrontEnd, self).__init__()
        self._cfg = cfg
        self._mode = mode
        self._is_training = self._is_net_training()
        self._class_num = self._cfg.DATASET.NUM_CLASSES
        self.net_results =collections.OrderedDict()
        
        if front_end_model == 'vgg':
            self.encoder = VGG16FCNEncode()
            self.decoder = VGG16FCNDecode(self._class_num)
        else:
            raise ValueError('Model has to be vgg')
        
    def call(self, input_tensors, training=False):
        #print(input_tensors)
        encoder = self.encoder(input_tensors, training=training)
        decoder = self.decoder(encoder, training=training)

        return decoder

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