from config import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class ModelManager:
    def __init__(self):
        self.params = Parameters()

    def train_model(self, x_set, y_set, prep_model, model, loss_fn, optimizer):
        print("Starting Training")
        steps_per_epoch = int(len(x_set)/self.params.BATCH_SIZE)
        for epoch in range(self.params.EPOCHS):
            print("Epoch: " + str(epoch) + " [", end='')
            running_loss = 0
            batch_log_count = 0
            for batch in range(steps_per_epoch):
                batch_slice = slice(batch*self.params.BATCH_SIZE, (batch+1)*self.params.BATCH_SIZE)
                x_batch, y_batch = x_set[batch_slice], y_set[batch_slice]

                if prep_model is not None:
                    x_batch = prep_model.call(x_batch)

                with tf.GradientTape() as tape:
                    logits = model.call(x_batch)
                    loss_value = loss_fn(y_batch, logits)
                grads = tape.gradient(loss_value, model.get_model().trainable_weights)
                optimizer.apply_gradients(zip(grads, model.get_model().trainable_weights))

                running_loss += loss_value

                log_tick = int((batch/steps_per_epoch)*100)
                interval = 4
                if log_tick % interval == 0 and log_tick != batch_log_count:
                    print("=", end='')
                    batch_log_count += interval
            print("] Loss {loss:.2f}".format(loss=running_loss/steps_per_epoch))
        model.freeze_layers()
        # return model

    @staticmethod
    def rpn_loss(y_true, y_pred):
        # It is assumed that the classification for foreground/background is the last 'layer' of the output array
        y_true_anchors = y_true[:, :, :, :-10]
        y_pred_anchors = y_pred[:, :, :, :-10]
        y_true_anchor_sel = y_true[:, :, :, 36:-1]
        y_pred_anchor_sel = y_pred[:, :, :, 36:-1]
        y_true_cls = y_true[:, :, :, -1]
        y_pred_cls = y_pred[:, :, :, -1]

        pos_indices = tf.where(y_true_cls == 1)
        neg_indices = tf.where(y_true_cls == -1)

        num_positives = tf.reduce_sum(tf.cast(tf.equal(y_true_cls, tf.constant(1.0)), tf.int32))
        slice = tf.math.minimum(tf.constant(128), num_positives)
        pos_slice = slice
        neg_slice = 128-slice
        pos_indices = pos_indices[:pos_slice]
        neg_indices = neg_indices[:neg_slice]

        concat_indices = tf.concat([pos_indices, neg_indices], axis=0)
        cls_true_collected = tf.clip_by_value(tf.gather_nd(y_true_cls, concat_indices), 0, 1)
        cls_pred_collected = tf.gather_nd(y_pred_cls, concat_indices)
        cls_loss = tf.keras.losses.BinaryCrossentropy()(cls_true_collected, cls_pred_collected)

        a_sel_addition = tf.concat([tf.zeros([9, 3], dtype=tf.int64), tf.reshape(tf.range(0, 9, dtype=tf.int64), [9, 1])], axis=-1)
        a_sel_concat = tf.pad(pos_indices, tf.constant([[0, 0], [0, 1]]), 'CONSTANT')
        # a_sel_concat = tf.concat([pos_indices, tf.zeros([None, 1], dtype=tf.int64)], axis=-1)
        a_sel_tiled = tf.tile(tf.expand_dims(a_sel_concat, axis=1), tf.constant([1, 9, 1]))
        a_sel_indices = tf.add(a_sel_tiled, a_sel_addition)
        a_sel_true_organized = tf.gather_nd(y_true_anchor_sel, a_sel_indices)
        a_sel_pred_organized = tf.gather_nd(y_pred_anchor_sel, a_sel_indices)
        a_sel_loss = tf.keras.losses.CategoricalCrossentropy()(a_sel_true_organized, a_sel_pred_organized)

        gathered_a_reg_true = tf.gather_nd(y_true_anchor_sel, a_sel_indices)
        pos_slices_true = tf.gather_nd(y_true, pos_indices)
        pos_slices_pred = tf.gather_nd(y_pred, pos_indices)
        a_reg_indices = tf.where(gathered_a_reg_true == 1)
        b_reg_indices = tf.add(tf.multiply(tf.tile(tf.expand_dims(a_reg_indices, axis=1),
                                                   tf.constant([1, 4, 1], dtype=tf.int64)),
                                           tf.constant([1, 4], dtype=tf.int64)),
                               tf.concat([tf.zeros([4, 1], dtype=tf.int64), tf.reshape(tf.range(4, dtype=tf.int64), [4, 1])], axis=-1))
        organized_b_reg_true = tf.gather_nd(pos_slices_true, b_reg_indices)
        organized_b_reg_pred = tf.gather_nd(pos_slices_pred, b_reg_indices)
        anchor_loss = tf.keras.losses.Huber()(organized_b_reg_true, organized_b_reg_pred)

        loss = 10*cls_loss + 5*anchor_loss + 2*a_sel_loss  # was 7, 5, 2
        return loss

    @staticmethod
    def cls_loss(y_true, y_pred):
        pred = y_pred[:, 0]
        true = y_true[:, 0]
        out = tf.math.reduce_mean(keras.losses.categorical_crossentropy(true, pred))
        return out
