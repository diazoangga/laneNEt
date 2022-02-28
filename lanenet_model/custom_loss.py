import tensorflow as tf

class LossFunction(object):
    def __init__(self, binary_loss_type):
        if binary_loss_type == 'cross_entropy':
            self.binary_loss = tf.keras.losses.CategoricalCrossentropy(
                    from_logits=True)
    def cross_entropy_loss(self, onehot_labels, logits, classes_weights):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        loss = self.binary_loss(
            y_true=onehot_labels,
            y_pred=logits,
            sample_weight=loss_weights
        )

        return loss

    def focal_loss(self, onehot_labels, logits, classes_weights, gamma=2.0):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :param gamma:
        :return:
        """
        epsilon = 1.e-7
        alpha = tf.multiply(onehot_labels, classes_weights)
        alpha = tf.cast(alpha, tf.float32)
        gamma = float(gamma)
        y_true = tf.cast(onehot_labels, tf.float32)
        y_pred = tf.nn.softmax(logits, dim=-1)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
    
        return loss