"""
Tusimple lanenet trainner
"""
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import os
import os.path as ops
from pickletools import optimize
from socketserver import ThreadingUDPServer
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import shutil
import time
import math

import numpy as np
import tensorflow as tf
import loguru
import tqdm
import cv2

from data_provider import lanenet_data_feed_piplinep
from lanenet_model import lanenet_p as lanenet
from semantic_segmentation_zoo.bisenetv2 import *

from lanenet_model.custom_loss import LossFunction
from lanenet_model.lanenet_discriminative_loss import discriminative_loss

LOG = loguru.logger

from local_utils.config_utils import parse_config_utils

class LaneNetTusimpleTrainer(object):
    """
    init lanenet single gpu trainner
    """

    def __init__(self, cfg):
        """
        initialize lanenet trainner
        """
        super(LaneNetTusimpleTrainer, self).__init__()
        self._cfg = cfg
        # device = tf.config.list_physical_devices('GPU')
        # gpus = tf.config.set_visible_devices(device[1:], 'GPU')
        # logical_devices = tf.config.list_logical_devices('GPU')
        # assert len(logical_devices) == len(device) - 1
        # try:
        #     for gpu in gpus:
        #         tf.config.experimental.set_memory_growth(gpu, self._cfg.GPU.TF_ALLOR_GROWTH)
        # except RuntimeError as error:
        #     print(error)

        # define solver params and dataset
        self._train_dataset = lanenet_data_feed_piplinep.LaneNetDataFeeder(flags='train')
        self._val_dataset = lanenet_data_feed_piplinep.LaneNetDataFeeder(flags='val')
        self._steps_per_epoch = len(self._train_dataset)

        self._model_name = '{:s}_{:s}'.format(self._cfg.MODEL.FRONT_END, self._cfg.MODEL.MODEL_NAME)

        self._train_epoch_nums = self._cfg.TRAIN.EPOCH_NUMS
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._val_batch_size =  self._cfg.TRAIN.VAL_BATCH_SIZE 
        self._snapshot_epoch = self._cfg.TRAIN.SNAPSHOT_EPOCH
        self._model_save_dir = ops.join(self._cfg.TRAIN.MODEL_SAVE_DIR, self._model_name)
        self._tboard_save_dir = ops.join(self._cfg.TRAIN.TBOARD_SAVE_DIR, self._model_name)
        self._enable_miou = self._cfg.TRAIN.COMPUTE_MIOU.ENABLE
        if self._enable_miou:
            self._record_miou_epoch = self._cfg.TRAIN.COMPUTE_MIOU.EPOCH
        self._input_tensor_size = [int(tmp) for tmp in self._cfg.AUG.TRAIN_CROP_SIZE]

        self._init_learning_rate = self._cfg.SOLVER.LR
        self._moving_ave_decay = self._cfg.SOLVER.MOVING_AVE_DECAY
        self._momentum = self._cfg.SOLVER.MOMENTUM
        self._lr_polynimal_decay_power = self._cfg.SOLVER.LR_POLYNOMIAL_POWER
        self._optimizer_mode = self._cfg.SOLVER.OPTIMIZER.lower()

        self._binary_loss_type = self._cfg.SOLVER.LOSS_TYPE
        self.loss = LossFunction(self._binary_loss_type)

        self._optimizer_mode = self._cfg.SOLVER.OPTIMIZER
        self._init_learning_rate = self._cfg.SOLVER.LR
        self._momentum = self._cfg.SOLVER.MOMENTUM
        self._weight_decay = self._cfg.SOLVER.WEIGHT_DECAY

        if self._cfg.RESUME_TRAINING.ENABLE == True:
            self._init_epoch = self._cfg.RESUME_TRAINING.INIT_EPOCH
            assert self._init_epoch <= self._train_epoch_nums, "The initial epoch has to be less than number of total epochs"
        else:
            self._init_epoch = 0
        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            self._initial_weight = self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.SNAPSHOT_PATH
        else:
            self._initial_weight = None
        if self._cfg.TRAIN.WARM_UP.ENABLE:
            self._warmup_epoches = self._cfg.TRAIN.WARM_UP.EPOCH_NUMS
            self._warmup_init_learning_rate = self._init_learning_rate / 1000.0
        else:
            self._warmup_epoches = 0
        
        if self._optimizer_mode == 'sgd':
            warmup_steps = tf.constant(
                self._warmup_epoches * self._steps_per_epoch, dtype=tf.float32)
            train_steps = tf.constant(
                self._train_epoch_nums * self._steps_per_epoch, dtype=tf.float32
            )
            lrate = LRSchedule(self._init_learning_rate, self._warmup_init_learning_rate,
                                warmup_steps, train_steps,
                                self._lr_polynimal_decay_power)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lrate,
                                                    momentum=self._momentum)

        #self._model = lanenet.LaneNet('train', self._cfg)
        self._model = BiseNetV2()
        self._model = self._model.build_model([256, 512, 3])
        self.overfit = False
        # print(np.unique(self._input_instance_label_image.numpy()))
        # for img in self._input_instance_label_image.numpy():
        #     try:
        #         cv2.imshow('src',img)
        #         cv2.waitKey(0)
        #     except:
        #         print('cant show img')
        #         continue
        print(self._batch_size)
        #self.train_dataset = self._train_dataset.next_batch(batch_size=self._batch_size)
        #self.val_dataset = self._val_dataset.next_batch(batch_size=self._val_batch_size)

        self.metrics = tf.keras.metrics.MeanIoU(2)
        self.val_metrics = tf.keras.metrics.MeanIoU(2)
        
    
    def train(self):
        self.overfitting_count = 0
        self.val_loss = np.Inf
        for epoch in range(self._init_epoch, self._train_epoch_nums):
            train_epoch_losses = []
            train_epoch_bloss = []
            train_epoch_iloss = []
            train_epoch_l2loss = []
            train_epoch_miou = []
            tprogress_bar = tqdm.tqdm(range(1, self._steps_per_epoch), position=0, leave=True)
            for _ in tprogress_bar:    
                input_src_img, input_bin_label, input_inst_label = self._train_dataset.next_batch()
                #print(step, input_src_img.shape, input_bin_label.shape, input_inst_label.shape)
        #         for img in input_src_img:
        #             #print(img)
        #             cv2.imshow('s', img.numpy())
        #             cv2.waitKey(0)
                with tf.GradientTape() as tape:
                    #tape.watch(train_var_list)
                    out_model = self._model(input_src_img, training=True)
                    bin_logits = out_model['binary_logits']
                    bin_pred = out_model['binary_prediction']
                    inst_pred = out_model['instance_prediction']
                    loss_dict = self.compute_loss(bin_logits, input_bin_label, inst_pred, input_inst_label, 'loss', False)
                    train_var_list = self._model.trainable_variables
                    
                self.metrics.update_state(input_bin_label, bin_pred)
                
                train_epoch_losses.append(loss_dict['total_loss'].numpy())
                train_epoch_bloss.append(loss_dict['binary_segmenatation_loss'].numpy())
                train_epoch_iloss.append(loss_dict['instance_segmentation_loss'].numpy())
                train_epoch_l2loss.append(loss_dict['l2_reg_loss'].numpy())
                train_epoch_miou.append(self.metrics.result())

                tprogress_bar.set_description(('Epoch {:d}:   train loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}, l2reg_loss: {:.5f}, miou: {:.5f}').format(
                        epoch,
                        loss_dict['total_loss'].numpy(),
                        loss_dict['binary_segmenatation_loss'].numpy(),
                        loss_dict['instance_segmentation_loss'].numpy(),
                        loss_dict['l2_reg_loss'].numpy(),
                        self.metrics.result()
                )
                )
                gradients = tape.gradient(loss_dict['total_loss'], train_var_list)
                #print(train_var_list)
                self.optimizer.apply_gradients(zip(gradients, train_var_list))
                #time.sleep(0.2)
                #print(self.optimizer)
            
            train_epoch_losses = np.mean(train_epoch_losses)
            train_epoch_bloss = np.mean(train_epoch_bloss)
            train_epoch_iloss = np.mean(train_epoch_iloss)
            train_epoch_l2loss = np.mean(train_epoch_l2loss)
            train_epoch_miou = np.mean(train_epoch_miou)

            self.metrics.reset_states()

            #validation step
            val_epoch_loss = []
            val_epoch_bloss = []
            val_epoch_iloss = []
            val_epoch_l2loss = []
            val_epoch_miou = []

            for _ in range(self._val_batch_size):
                val_src_img, val_bin_label, val_inst_label = self._val_dataset.next_batch()
                val_out = self._model(val_src_img, training=False)
                val_bin_logits = val_out['binary_logits']
                val_bin_pred = val_out['binary_prediction']
                val_inst_pred = val_out['instance_prediction']
                val_loss_dict = self.compute_loss(val_bin_logits, val_bin_label, val_inst_pred, val_inst_label, 'loss', False)

                val_epoch_loss.append(val_loss_dict['total_loss'].numpy())
                val_epoch_bloss.append(val_loss_dict['binary_segmenatation_loss'].numpy())
                val_epoch_iloss.append(val_loss_dict['instance_segmentation_loss'].numpy())
                val_epoch_l2loss.append(val_loss_dict['l2_reg_loss'].numpy())
                self.val_metrics.update_state(val_bin_label, val_bin_pred)
                val_epoch_miou.append(self.val_metrics.result())
            
            val_epoch_loss = np.mean(val_epoch_loss)
            val_epoch_bloss = np.mean(val_epoch_bloss)
            val_epoch_iloss = np.mean(val_epoch_iloss)
            val_epoch_l2loss = np.mean(val_epoch_l2loss)
            val_epoch_miou = np.mean(val_epoch_miou)

            log_time = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime(time.time()))
            LOG.info(
                '\n---> Epoch: {:d} Time: {:s} Train loss: {:.5f} Train bloss: {:.5f} Train iloss: {:.5f} Train l2loss: {:.5f} Train miou: {:.5f} ...\n---> Epoch: {:d} Time: {:s} Val loss: {:.5f} Val bloss: {:.5f} Val iloss: {:.5f} Val l2loss: {:.5f} Val miou: {:.5f} ...'.format(
                    epoch, log_time,
                    train_epoch_losses,
                    train_epoch_bloss,
                    train_epoch_iloss,
                    train_epoch_l2loss,
                    train_epoch_miou,
                    epoch, log_time,
                    val_epoch_loss,
                    val_epoch_bloss,
                    val_epoch_iloss,
                    val_epoch_l2loss,
                    val_epoch_miou
                )
            )
            if epoch > self._warmup_epoches:
                self.overfit = self.check_overfitting(val_epoch_loss, 5)
            if epoch % self._snapshot_epoch == 0:
                self.save_checkpoint(train_epoch_miou)
            if self.overfit:
                self._model.stop_training = True
                #LOG.info('Model is stopped training due to overfitting at epoch {:d}'. format(epoch))
                save_dir = ops.join(self._model_save_dir, 'lanenet_{}_miou={:.4f}'.format(log_time, train_epoch_miou))
                #inp = tf.keras.layers.Input([256,512,3])
                #model = self._model(inp)
                #model = tf.keras.models.Model(inp, outputs=[model['binary_logits'], model['binary_prediction'], model['instance_prediction']])
                self._model.save(save_dir, save_format='tf')
                LOG.info('Model is stopped training due to overfitting at epoch {:d} and saved in {}'. format(epoch, save_dir))
                exit()
            
            self.val_metrics.reset_states()
        
        save_dir = ops.join(self._model_save_dir, 'lanenet_{}_miou={:.4f}.h5'.format(log_time, train_epoch_miou))
        # inp = Input([256,512,3])
        # y = self._model(inp)
        # out = Model(inp, y)
        self._model.save_weights(save_dir)
        #tf.keras.models.save_model(self._model, save_dir)
        LOG.info('Model is saved in {}'. format(save_dir))
    
    def save_checkpoint(self, miou):
        inp = tf.keras.layers.Input([256,512,3])
        model = self._model(inp)
        model = tf.keras.models.Model(inp, outputs=[model['binary_logits'], model['binary_prediction'], model['instance_prediction']])
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        checkpoint_name = 'lanenet_train_miou={:.4f}.ckpt'.format(miou)
        checkpoint_path = ops.join(self._model_save_dir, checkpoint_name)
        checkpoint_prefix = ops.join(checkpoint_path, 'ckpt')
        os.makedirs(self._model_save_dir, exist_ok=True)
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def check_overfitting(self, val_loss, patience):
        
        if val_loss > self.val_loss:
            self.overfitting_count += 1
        else:
            self.val_loss = val_loss
            self.overfitting_count = 0
        
        overfit = True if self.overfitting_count > patience else False
        return overfit
    def compute_loss(self, binary_seg_logits, binary_label,
                     instance_seg_logits, instance_label,
                     name, reuse):
        """
        compute lanenet loss
        :param binary_seg_logits:
        :param binary_label:
        :param instance_seg_logits:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        binary_label_onehot = tf.one_hot(
            tf.reshape(
                tf.cast(binary_label, tf.int32),
                shape=[binary_label.get_shape().as_list()[0],
                        binary_label.get_shape().as_list()[1],
                        binary_label.get_shape().as_list()[2]]),
            depth=binary_seg_logits.get_shape().as_list()[3],
            axis=-1
        )
        #print(binary_label_onehot.get_shape())
        # cv2.imshow('s', (binary_label_onehot.numpy()[1,:,:,1]*255.0).astype(np.uint8))
        # cv2.waitKey(0)

        binary_label_plain = tf.reshape(
            binary_label,
            shape=[binary_label.get_shape().as_list()[0] *
                    binary_label.get_shape().as_list()[1] *
                    binary_label.get_shape().as_list()[2] *
                    binary_label.get_shape().as_list()[3]])
        unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
        counts = tf.cast(counts, tf.float32)
        inverse_weights = tf.math.divide(
            1.0,
            tf.math.log(tf.math.add(tf.math.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
        )
        if self._binary_loss_type == 'cross_entropy':
            binary_segmenatation_loss = self.loss.cross_entropy_loss(
                onehot_labels=binary_label_onehot,
                logits=binary_seg_logits,
                classes_weights=inverse_weights
            )
        
        # elif self._binary_loss_type == 'focal':
        #     binary_segmenatation_loss = self._multi_category_focal_loss(
        #         onehot_labels=binary_label_onehot,
        #         logits=binary_seg_logits,
        #         classes_weights=inverse_weights
        #     )
        # else:
        #     raise NotImplementedError

        pix_image_shape = (instance_seg_logits.get_shape().as_list()[1], instance_seg_logits.get_shape().as_list()[2])
        instance_segmentation_loss, l_var, l_dist, l_reg = \
            discriminative_loss(
                instance_seg_logits, instance_label, self._cfg.MODEL.EMBEDDING_FEATS_DIMS,
                pix_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
            )

        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in self._model.trainable_variables:
            if 'bn' in vv.name or 'gn' in vv.name:
                continue
            else:
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        l2_reg_loss *= 0.001
        total_loss = binary_segmenatation_loss + instance_segmentation_loss + l2_reg_loss
        
        
        

        # ret = {
        #     'total_loss': total_loss,
        #     'binary_seg_logits': binary_seg_logits,
        #     'instance_seg_logits': pix_embedding,
        #     'binary_seg_loss': binary_segmenatation_loss,
        #     'discriminative_loss': instance_segmentation_loss
        # }

        return {
            'total_loss': total_loss,
            'binary_segmenatation_loss': binary_segmenatation_loss,
            'instance_segmentation_loss': instance_segmentation_loss,
            'l2_reg_loss': l2_reg_loss
        }

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_learning_rate, warmup_init_learning_rate, warmup_steps, train_epoch_nums, lr_polynimal_decay_power):
        self._init_learning_rate = init_learning_rate
        self._warmup_init_learning_rate = warmup_init_learning_rate
        self._train_epoch_steps = train_epoch_nums
        self._lr_polynimal_decay_power = lr_polynimal_decay_power
        self._warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self._warmup_steps:
            factor = tf.math.pow(self._init_learning_rate / self._warmup_init_learning_rate, 1.0 / self._warmup_steps)
            warmup_lr = self._warmup_init_learning_rate * tf.math.pow(factor, step)
        else:
            warmup_lr = (self._init_learning_rate - 0.000001) * \
                        (1 - step/self._train_epoch_steps)**(self._lr_polynimal_decay_power) + \
                        0.000001
        #print('LR: {}'.format(warmup_lr))
        return warmup_lr

cfg = parse_config_utils.lanenet_cfg
test = LaneNetTusimpleTrainer(cfg)
test.train()
