import tensorflow as tf
from lanenet_model.lanenet_backend import LaneNetBackEnd
from lanenet_model.lanenet_frontend import LaneNetFrontEnd
from tensorflow.keras import Model
# from local_utils.config_utils import parse_config_utils

class LaneNet(Model):
    def __init__(self, mode, cfg):
        super(LaneNet, self).__init__()
        self._cfg = cfg
        self._net_flag = self._cfg.MODEL.FRONT_END

        self.frontend = LaneNetFrontEnd(mode, self._cfg, self._net_flag)
        self.backend = LaneNetBackEnd(mode, self._cfg)
    
    def call(self, input_tensors, training=False):
        ext_feats = self.frontend(input_tensors, training=training)
        #print(ext_feats['instance_segment_logits']['data'])

        binary_pred, inst_pred = self.backend(ext_feats['binary_segment_logits']['data'],
                                                ext_feats['instance_segment_logits']['data'],
                                                training=training)

        return {
            'binary_logits': ext_feats['binary_segment_logits']['data'],
            'binary_prediction': binary_pred,
            'instance_prediction': inst_pred
        }


# cfg = parse_config_utils.lanenet_cfg
# inp = tf.ones([1,256,512,3])
# net = LaneNet('train', cfg)
# out = net(inp)

# print(out)