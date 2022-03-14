import os
from pickle import TRUE
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import os.path as ops

from local_utils.config_utils import parse_config_utils\

import math

from focal_loss import SparseCategoricalFocalLoss
from focal_loss import BinaryFocalLoss
import datetime
import time



import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint
from data_provider import lanenet_data_feed_piplinep
from semantic_segmentation_zoo.bisenetv2_1 import *

from lanenet_model.custom_loss import LossFunction
from lanenet_model.lanenet_instance_loss import InstanceLoss
from mode import simple_unet_model

from data_provider import tf_io_pipline_tools

import numpy as np
import tqdm

import matplotlib.pyplot as plt

tf.random.set_seed(40)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

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

        self.len_dataset = len(lanenet_data_feed_piplinep.LaneNetDataFeeder(flags='train'))
        print(self.len_dataset)

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
        #self.b_loss = SparseCategoricalFocalLoss(gamma=2, from_logits=True)
        self.b_loss = BinaryFocalLoss(gamma=2)

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

        #self._model = lanenet.LaneNet('train', self._cfg)
        # self._model = BiseNetV2()
        # self._model = self._model.build_model([256, 512, 3])
        # self._model = bisenetv2()
        self._model = simple_unet_model(256,512,3)
        self.overfit = False

        self.metrics = tf.keras.metrics.MeanIoU(2)
        self.val_metrics = tf.keras.metrics.MeanIoU(2)

        self._embedded_feat_dims = self._cfg.MODEL.EMBEDDING_FEATS_DIMS
    
    def train(self):
        tf.random.set_seed(40)
        train_dataset = training_dataset(self._cfg)
        val_dataset = valid_dataset(self._cfg)
        loss_weight = [1,1]
        schedule = optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self._init_learning_rate,
                decay_steps=self._weight_decay,
                power=self._lr_polynimal_decay_power
            )
        early_stop = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', min_delta=0, patience=8, verbose=0,
                            mode='auto', baseline=None, restore_best_weights=False
                            )
        checkpoint = ModelCheckpoint(filepath=ops.join(self._model_save_dir, 'lanenet_ckpt.epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
                                    monitor='val_loss',
                                    verbose=1,
                                    save_weights_only=True,
                                    save_best_only=True,
                                    mode='min')
        lr = tf.keras.callbacks.LearningRateScheduler(custom_lr)
        optimizer = tf.keras.optimizers.Adam(learning_rate = schedule)
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self._model.compile(optimizer=optimizer,
                            loss = [self.b_loss, InstanceLoss(self._embedded_feat_dims)],
                            loss_weights = loss_weight,
                            metrics = [miou, 'accuracy']
                            )

        history = self._model.fit(train_dataset,
                       #batch_size = self._batch_size,
                       verbose=1,
                       epochs=self._train_epoch_nums,
                       steps_per_epoch=762,
                       validation_data=val_dataset,
                        shuffle=False,
                       callbacks=[tensorboard_callback, early_stop, checkpoint]
                       )
        #print(history['metrics'])
        log_time = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime(time.time()))
        save_dir = ops.join(self._model_save_dir, 'lanenet_final_{}.h5'.format(log_time))
        self._model.save_weights(save_dir)

class ArgmaxMeanIOU(metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_pred.get_shape().as_list()[3] == 2:
            return super().update_state(tf.argmax(y_true,axis=3), tf.argmax(tf.nn.softmax(y_pred,axis=-1), axis=-1), sample_weight)
        else:
            return None

def miou(y_true, y_pred):
    #print(y_true.get_shape())
    def f(y_true, y_pred):
        y_pred = np.where(y_pred > 0.5, 1, 0)
        axes=(1,2)
        intersection = np.sum(np.logical_and(y_true, y_pred))
        union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    
    #y_pred = tf.argmax(tf.nn.softmax(y_pred, axis=-1), axis=-1)
    #y_true = tf.squeeze(y_true, axis=-1)
    #n = y_pred.get_shape().as_list()[]
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
   

def custom_lr(epoch):
    cfg = parse_config_utils.lanenet_cfg
    warmup_epoches = cfg.TRAIN.WARM_UP.EPOCH_NUMS
    init_learning_rate = cfg.SOLVER.LR
    warmup_init_learning_rate = init_learning_rate / 1000.0
    lr_polynimal_decay_power = cfg.SOLVER.WEIGHT_DECAY
    train_epoch_nums = cfg.TRAIN.EPOCH_NUMS
    if epoch < warmup_epoches:
        grad = (init_learning_rate - warmup_init_learning_rate)/warmup_epoches
        lr = warmup_init_learning_rate+ (grad*epoch)
    else:
        lr = 0.5*(2.3209*(10**(-4))*(0.9**epoch) + 0.93835*(10**(-7)))
    #print('LR: {}'.format(warmup_lr))
    return lr

def load_dataset(CFG, flag='train'):
    dataset_dir = CFG.DATASET.DATA_DIR
    epoch_nums = CFG.TRAIN.EPOCH_NUMS
    train_batch_size = CFG.TRAIN.BATCH_SIZE
    val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE

    tfrecords_dir = ops.join(dataset_dir, 'tfrecords')
    if not ops.exists(tfrecords_dir):
        raise ValueError('{:s} not exist, please check again'.format(tfrecords_dir))

    dataset_flags = flag.lower()
    if dataset_flags not in ['train', 'val']:
        raise ValueError('flags of the data feeder should be \'train\', \'val\'')

    tfrecords_file_paths = ops.join(tfrecords_dir, 'tusimple_{:s}.tfrecords'.format(dataset_flags))
    assert ops.exists(tfrecords_file_paths), '{:s} not exist'.format(tfrecords_file_paths)

class customCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        dataset = training_dataset(cfg)
        a, b, c = next(iter(dataset))
        plt.figure(figsize=(50,50))
        for n in range(8):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(a[n])
            plt.axis("off")
        
        plt.show()
def training_dataset(CFG):
    dataset_dir = CFG.DATASET.DATA_DIR
    epoch_nums = CFG.TRAIN.EPOCH_NUMS
    train_batch_size = CFG.TRAIN.BATCH_SIZE
    val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
    flags = 'train'

    tfrecords_dir = ops.join(dataset_dir, 'tfrecords')
    if not ops.exists(tfrecords_dir):
        raise ValueError('{:s} not exist, please check again'.format(tfrecords_dir))

    dataset_flags = flags.lower()
    if dataset_flags not in ['train', 'val']:
        raise ValueError('flags of the data feeder should be \'train\', \'val\'')

    tfrecords_file_paths = ops.join(tfrecords_dir, 'tusimple_{:s}.tfrecords'.format(dataset_flags))
    assert ops.exists(tfrecords_file_paths), '{:s} not exist'.format(tfrecords_file_paths)

    with tf.device('/cpu:0'):

        # TFRecordDataset opens a binary file and reads one record at a time.
        # `tfrecords_file_paths` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(tfrecords_file_paths)
        dataset = dataset.shuffle(buffer_size=512)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(
            map_func=tf_io_pipline_tools.decode,
            num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
        )
        if dataset_flags == 'train':
            dataset = dataset.map(
                map_func=tf_io_pipline_tools.augment_for_train,
                num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
            )
            batch_size = train_batch_size
        elif dataset_flags == 'val':
            dataset = dataset.map(
                map_func=tf_io_pipline_tools.augment_for_test,
                num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
            )
            batch_size = val_batch_size
        dataset = dataset.map(
            map_func=tf_io_pipline_tools.normalize,
            num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
        )
        
        #dataset = dataset.repeat(epoch_nums)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.repeat(epoch_nums)
        # repeat num epochs
        
        
        # dataset = dataset.repeat(epoch_nums)

        
        
    return dataset

def valid_dataset(CFG):
    dataset_dir = CFG.DATASET.DATA_DIR
    epoch_nums = CFG.TRAIN.EPOCH_NUMS
    train_batch_size = CFG.TRAIN.BATCH_SIZE
    val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
    flags = 'val'

    tfrecords_dir = ops.join(dataset_dir, 'tfrecords')
    if not ops.exists(tfrecords_dir):
        raise ValueError('{:s} not exist, please check again'.format(tfrecords_dir))

    dataset_flags = flags.lower()
    if dataset_flags not in ['train', 'val']:
        raise ValueError('flags of the data feeder should be \'train\', \'val\'')

    tfrecords_file_paths = ops.join(tfrecords_dir, 'tusimple_{:s}.tfrecords'.format(dataset_flags))
    assert ops.exists(tfrecords_file_paths), '{:s} not exist'.format(tfrecords_file_paths)

    with tf.device('/cpu:0'):

        # TFRecordDataset opens a binary file and reads one record at a time.
        # `tfrecords_file_paths` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(tfrecords_file_paths)
        dataset = dataset.shuffle(buffer_size=512)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(
            map_func=tf_io_pipline_tools.decode,
            num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
        )
        if dataset_flags == 'train':
            dataset = dataset.map(
                map_func=tf_io_pipline_tools.augment_for_train,
                num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
            )
            batch_size = train_batch_size
        elif dataset_flags == 'val':
            dataset = dataset.map(
                map_func=tf_io_pipline_tools.augment_for_test,
                num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
            )
            batch_size = val_batch_size
        dataset = dataset.map(
            map_func=tf_io_pipline_tools.normalize,
            num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
        )

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # dataset = dataset.repeat(epoch_nums)
        
    return dataset

cfg = parse_config_utils.lanenet_cfg
a = LaneNetTusimpleTrainer(cfg)
a.train()
# d = training_dataset(cfg)
# for idx,k in enumerate(d):
#     print(idx)
#     # plt.figure(figsize=(10,10))
#     # for a in range(8):
#     #    plt.subplot(4,2,a+1)
#     #    plt.imshow(np.array(i)[a])
#     # plt.show()
