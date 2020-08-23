from config import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from .parameters import Parameters


class ModelManager:
    def __init__(self):
        self.params = Parameters()

    def generate_frcnn(self):
        pre_trained_model = VGG16(input_shape=(self.params.IMAGE_HEIGHT, self.params.IMAGE_WIDTH, 3), include_top=False)

        input = Input(shape=(self.params.IMAGE_HEIGHT, self.params.IMAGE_WIDTH, 3))
        x = input
        for layer in pre_trained_model.layers[1:-1]:
            layer.trainable = False
            x = layer(x)
        num_anchors = self.params.num_anchors()
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        bbox_reg = Conv2D(num_anchors*4, (1, 1), padding='same')(x)
        cls_scr = Conv2D(num_anchors, (1, 1), activation='sigmoid', padding='same')(x)
        output = Concatenate()([bbox_reg, cls_scr])

        model = Model(inputs=[input], outputs=[output])
        return model

    def generate_cls_model(self):
        input = Input(shape=(self.params.IMAGE_HEIGHT, self.params.IMAGE_WIDTH, 3))
        x = Conv2D(16, (5, 5), padding='same', activation='relu')(input)
        x = MaxPooling2D((2,2), (2,2))(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), (2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), (2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.params.NUM_CLASSES, activation='softmax')(x)
        return Model(inputs=[input], outputs=[x])

    def train_model(self, model, x_set, y_set):
        epochs = self.params.EPOCHS
        batch_size = self.params.BATCH_SIZE
        num_val_batches = self.params.NUM_VAL_BATCHES

        # modifying number of data points to be divisible by the batch size
        new_num_data_points = int(math.floor(len(x_set) / batch_size) * batch_size)
        x_set = x_set[:new_num_data_points]
        y_set = y_set[:new_num_data_points]
        if num_val_batches != 0:
            train_x_set = x_set[:-num_val_batches * batch_size]
            train_y_set = y_set[:-num_val_batches * batch_size]
            val_x_set = x_set[-num_val_batches * batch_size:]
            val_y_set = y_set[-num_val_batches * batch_size:]
            fit_data = model.fit(train_x_set, train_y_set, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.1, verbose=1)
        else:
            train_x_set = x_set
            train_y_set = y_set
            fit_data = model.fit(train_x_set, train_y_set, epochs=epochs, batch_size=batch_size, verbose=1)
        return model, fit_data

    @staticmethod
    def rpn_loss(y_true, y_pred):
        # It is assumed that the classification for foreground/background is the last 'layer' of the output array
        y_true_anchors = y_true[:, :, :, :-9]
        y_pred_anchors = y_pred[:, :, :, :-18]
        y_true_cls = y_true[:, :, :, -9:]
        y_pred_cls = y_pred[:, :, :, -18:]

        pos_indices = tf.where(y_true_cls == 1)
        neg_indices = tf.where(y_true_cls == -1)

        num_positives = tf.reduce_sum(tf.cast(tf.equal(y_true_cls, tf.constant(1.0)), tf.int32))
        slice = tf.math.minimum(tf.constant(128), num_positives)
        pos_slice = slice
        neg_slice = slice
        pos_indices = pos_indices[:pos_slice]
        neg_indices = neg_indices[:neg_slice]

        cls_true_collected = tf.concat([tf.where(pos_indices[:, :2] >= 0, [1, 0], [0, 0]), tf.where(neg_indices[:, :2] >= 0, [0, 1], [0, 0])], axis=0)

        concat_indices = tf.concat([pos_indices, neg_indices], axis=0)
        # cls_true_collected = tf.clip_by_value(tf.gather_nd(y_true_cls, concat_indices), 0, 1)

        concat_indices = tf.expand_dims(concat_indices, axis=1)
        tiling = tf.constant([1, 2, 1], dtype=tf.int64)
        scaling = tf.constant([1, 1, 1, 2], dtype=tf.int64)
        addition = tf.concat([tf.zeros([2, 3], dtype=tf.int64), tf.reshape(tf.range(2, dtype=tf.int64), [2, 1])], axis=1)
        concat_indices = tf.add(tf.tile(tf.multiply(concat_indices, scaling), tiling),addition)

        cls_pred_collected = tf.gather_nd(y_pred_cls, concat_indices)
        cls_loss = tf.keras.losses.CategoricalCrossentropy()(cls_true_collected, cls_pred_collected)

        scaling = tf.reshape(tf.constant([1, 1, 1, 4], dtype=tf.int64), [1, 4])
        mapped_pos_indices = tf.multiply(pos_indices, scaling)
        tiling = tf.constant([1, 4, 1])
        tiled_pos_indices = tf.tile(tf.expand_dims(mapped_pos_indices, axis=1), tiling)
        addition_mat = tf.concat(
            [tf.zeros([4, 3], dtype=tf.int64), tf.transpose(tf.reshape(tf.range(4, dtype=tf.int64), [1, 4]))], axis=1)
        org_pos_indices = tf.add(tiled_pos_indices, addition_mat)
        anchor_true_collected = tf.gather_nd(y_true_anchors, org_pos_indices)
        anchor_pred_collected = tf.gather_nd(y_pred_anchors, org_pos_indices)
        anchor_loss = tf.keras.losses.Huber()(anchor_true_collected, anchor_pred_collected) / 2
        loss = cls_loss + 10 * anchor_loss
        return loss

    @staticmethod
    def rpn_loss_v2(y_true, y_pred):
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

        loss = 7*cls_loss + 5*anchor_loss + 2*a_sel_loss
        return loss

    @staticmethod
    def cls_loss(y_true, y_pred):
        pred = y_pred[:, 0]
        true = y_true[:, 0]
        out = tf.math.reduce_mean(keras.losses.categorical_crossentropy(true, pred))
        return out


if __name__ == "__main__":
    # add = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
    true = tf.ones([2, 28, 28, 46])
    pred = tf.ones([2, 28, 28, 46])

    # a_true = true[:, :, :, :36]
    # sel = true[:, :, :, 36:-1]
    # cls = tf.expand_dims(true[:, :, :, -1], axis=-1)
    # sel = tf.add(sel, add)
    # true = tf.concat([a_true, sel, cls], axis=-1)
    # cls_pred = tf.reshape(tf.subtract((tf.ones([2, 28, 28, 9, 2]) *0.75), tf.constant([0.5, 0])), [2, 28, 28, 18])
    # pred = tf.concat([tf.ones([2, 28, 28, 36]), cls_pred], axis=3)
    # true = true.numpy()
    # true[:, :, :, 36:40] = true[:, :, :, 36:40]*-1
    ModelManager.rpn_loss_v2(true, pred)