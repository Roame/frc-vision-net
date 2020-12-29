from config import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np


class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, batch_size=32, stride=16, output_size=[7, 7], training=True, **kwargs):
        super(ROIPooling, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.stride = stride
        self.output_size = output_size
        self.training = training

    def call(self, inputs, **kwargs):
        feature_maps = inputs[0]
        bboxes = inputs[1]
        map_stack = []
        for b in range(self.batch_size):
            coords = tf.cast(tf.stack([bboxes[b, :, 1]-bboxes[b, :, 3]/2, bboxes[b, :, 0]-bboxes[b, :, 2]/2,
                                       bboxes[b, :, 1]+bboxes[b, :, 3]/2, bboxes[b, :, 0]+bboxes[b, :, 2]/2], axis=1), tf.float32)
            relative_scaling = tf.cast(tf.constant([1/(feature_maps.shape[1]*self.stride), 1/(feature_maps.shape[2]*self.stride),
                                                    1/(feature_maps.shape[1]*self.stride), 1/(feature_maps.shape[2]*self.stride)]), tf.float32)
            scaled_coords = tf.clip_by_value(tf.multiply(coords, relative_scaling), 0, 1)
            maps = tf.image.crop_and_resize(feature_maps, scaled_coords, tf.ones([10], dtype=tf.int32)*b, self.output_size, method='nearest')
            map_stack.append(maps)
        map_stack = tf.stack(map_stack)
        if self.training:
            return map_stack[:, 0, :, :, :]
        else:
            return map_stack

    def compute_output_shape(self, input_shape):
        if self.training:
            return None, self.output_size[0], self.output_size[1], 512
        else:
            return None, 10, self.output_size[0], self.output_size[1], 512

    def get_config(self):
        config = super(ROIPooling, self).get_config()
        config.update({"batch_size": self.batch_size, "stride": self.stride, "output_size": self.output_size, "training": self.training})
        return config


class NMSLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, batch_size=32, stride=16, cls_thresh=0.95, max_iou=0.1, num_proposals=10, **kwargs):
        super(NMSLayer, self).__init__(**kwargs)

        self.anchors = tf.constant(anchors)
        self.batch_size = batch_size
        self.stride = stride
        self.cls_thresh = cls_thresh
        self.max_iou = max_iou
        self.num_proposals = num_proposals

    def call(self, inputs, **kwargs):
        cls_scores = inputs[:, :, :, -1]
        cancel_mat = tf.pad(tf.ones([tf.shape(cls_scores)[0], tf.shape(cls_scores)[1]-2, tf.shape(cls_scores)[2]-2]), [[0, 0], [1, 1], [1, 1]])
        cls_scores = tf.multiply(cls_scores, cancel_mat)
        a_reg = inputs[:, :, :, -10:-1]
        bbox_reg = inputs[:, :, :, :-10]

        indices = tf.where(cls_scores > self.cls_thresh)
        a_indices = tf.argmax(tf.gather_nd(a_reg, indices), axis=1)

        scores = tf.gather_nd(cls_scores, indices)
        anchors = tf.cast(tf.gather(self.anchors, a_indices), tf.float32)
        a_bboxes = tf.stack([(tf.cast(indices[:, 2], tf.float32)+0.5)*self.stride,
                             (tf.cast(indices[:, 1], tf.float32)+0.5)*self.stride,
                              anchors[:, 1], anchors[:, 1]/anchors[:, 0]], axis=1)
        scaling = tf.constant([1, 1, 1, 4])
        tiling = tf.constant([1, 4, 1])
        addition = tf.concat([tf.zeros([4, 3], dtype=tf.int32), tf.reshape(tf.range(4, dtype=tf.int32), [4, 1])], axis=1)
        reg_indices = tf.add(tf.tile(tf.expand_dims(tf.multiply(tf.cast(tf.concat([indices, tf.expand_dims(a_indices, axis=-1)], axis=-1), tf.int32), scaling), axis=1), tiling), addition)
        deltas = tf.gather_nd(bbox_reg, reg_indices)
        bboxes = tf.stack([deltas[:, 0]*a_bboxes[:, 2]+a_bboxes[:, 0],
                          deltas[:, 1]*a_bboxes[:, 3]+a_bboxes[:, 1],
                          tf.exp(deltas[:, 2])*a_bboxes[:, 2],
                          tf.exp(deltas[:, 3])*a_bboxes[:, 3]], axis=1)
        coords = tf.stack([bboxes[:, 1]-bboxes[:, 3]/2, bboxes[:, 0]-bboxes[:, 2]/2,
                           bboxes[:, 1]+bboxes[:, 3]/2, bboxes[:, 0]+bboxes[:, 2]/2], axis=1)
        output = []
        for i in range(self.batch_size):
            b_indices = tf.where(indices[:, 0] == i)
            b_coords = tf.gather_nd(coords, b_indices)
            b_scores = tf.gather_nd(scores, b_indices)
            b_bboxes = tf.gather_nd(bboxes, b_indices)
            selected_indices = tf.image.non_max_suppression(b_coords, b_scores, self.num_proposals, self.max_iou)
            s_bboxes = tf.gather(b_bboxes, selected_indices)
            output.append(tf.pad(s_bboxes, [[0, 10-tf.shape(s_bboxes)[0]], [0, 0]]))
        return tf.stack(output)

    def get_config(self):
        config = super(NMSLayer, self).get_config()
        config.update({"anchors": self.anchors.numpy(), "batch_size": self.batch_size, "stride": self.stride,
                       "cls_thresh": self.cls_thresh, "max_iou": self.max_iou, "num_proposals": self.num_proposals})
        return config

    # def compute_output_shape(self, input_shape):
    #     return None, 10, 4


class LoopedDense(tf.keras.layers.Layer):
    def __init__(self, dense_model, num_classes, weights=None, **kwargs):
        super(LoopedDense, self).__init__(**kwargs)
        if isinstance(dense_model, str):
            self.dense_model = keras.models.model_from_json(dense_model)
            for i in range(len(weights)):
                weights[i] = tf.convert_to_tensor(weights[i])
            self.dense_model.set_weights(weights)
        else:
            self.dense_model = dense_model
        self.num_classes = num_classes

    def call(self, inputs, **kwargs):
        classifications = []
        for i in range(inputs.shape[1]):
            map = inputs[:, i]
            x = self.dense_model(map)
            classifications.append(x[0])
        output = tf.expand_dims(tf.stack(classifications), axis=0)
        return output

    # def compute_output_shape(self, input_shape):
    #     return None, 10, self.num_classes

    def get_config(self):
        config = super(LoopedDense, self).get_config()
        config.update({"dense_model": self.dense_model.to_json(), "num_classes": self.num_classes, "weights": self.dense_model.get_weights()})
        return config


if __name__ == "__main__":
    # anchors = [[ratio, scale] for ratio in np.arange(1, 2.5, 0.5) for scale in np.arange(100, 250, 50)]
    # layer = ROIPooling([[ratio, scale] for ratio in np.arange(1, 2.5, 0.5) for scale in np.arange(100, 250, 50)], batch_size=2, stride=16, output_size=[7, 7])
    # print(layer([tf.reshape(tf.range(0, 255, delta=255/25088), [2, 14, 14, 64]), tf.ones([2, 10, 4]) * 0.85]))
    # print(layer.call([tf.reshape(tf.range(0, 255, delta=255/25088), [2, 14, 14, 64]), tf.ones([2, 10, 4]) * 0.85]))

    anchors = [[ratio, scale] for ratio in np.arange(1, 2.5, 0.5) for scale in np.arange(100, 250, 50)]
    layer = NMSLayer(anchors=anchors, batch_size=2)
    # print(layer(tf.reshape(tf.range(0, 255, delta=255/18032), [2, 14, 14, 46])))
    # print(layer.call(tf.ones([2, 14, 14, 46])))
    # print(layer(tf.reshape(tf.range(0, 255, delta=255 / 17640), [2, 14, 14, 45])))
    print(layer.call(tf.reshape(tf.range(0, 255, delta=255 / 17640), [2, 14, 14, 45])))

    # anchors = [[ratio, scale] for ratio in np.arange(1, 2.5, 0.5) for scale in np.arange(100, 250, 50)]
    # feature_map = tf.reshape(tf.range(0, 1, delta=1/(401408*2)), [2, 28, 28, 512])
    # proposals = tf.reshape(tf.range(0, 1, delta=1/(17640*4)), [2, 28, 28, 45])
    # bboxes = NMSLayer(anchors=anchors, batch_size=2).call(proposals)
    # pooling = ROIPooling(anchors, batch_size=2, stride=16, output_size=[7, 7]).call([feature_map, bboxes])
    # print(pooling)


