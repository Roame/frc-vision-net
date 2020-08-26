from config import *
from utility.custom_layers import *


from tensorflow.keras.layers import *
from enum import Enum


class ModelBlock:
    class BlockState(Enum):
        TRAINING = 0,
        RUNNING = 1

    def __init__(self):
        self._model = self._get_training_model()
        self.state = ModelBlock.BlockState.TRAINING
        print("initialized")

    def call(self, inputs):
        return self._model(inputs)

    def freeze_layers(self, slice=None):
        if slice is not None:
            for layer in self._model.layers[slice]:
                layer.trainable = False
        else:
            for layer in self._model.layers:
                layer.trainable = False

    def freeze_model(self):
        self._model.trainable = False

    def unfreeze_layers(self, slice=None):
        if slice is not None:
            for layer in self._model.layers[slice]:
                layer.trainable = True
        else:
            for layer in self._model.layers:
                layer.trainable = True

    def unfreeze_model(self):
        self._model.trainable = True

    def set_model(self, model):
        self._model = model

    def save(self, filepath):
        self._model.save(filepath)

    def load(self, filepath):
        self._model = keras.models.load_model(filepath)

    def configure(self, desired_state):
        if desired_state == ModelBlock.BlockState.TRAINING and self.state != desired_state:
            self._configure_for_training()
            self.state = ModelBlock.BlockState.TRAINING
        elif desired_state == ModelBlock.BlockState.RUNNING and self.state != desired_state:
            self._configure_for_running()
            self.freeze_layers()
            self.state = ModelBlock.BlockState.RUNNING
        else:
            ValueError(desired_state)

    def _configure_for_training(self): pass

    def _configure_for_running(self): pass

    def _get_training_model(self): pass


class FeatureExtractorBlock(ModelBlock):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def _get_training_model(self):
        return keras.applications.VGG16(input_shape=self.input_shape, include_top=False)


class RPNBlock(ModelBlock):
    def __init__(self, input_shape):
        self.params = Parameters()
        self.input_shape = input_shape
        self.nms_layer = NMSLayer(self.params.get_anchors(), batch_size=self.params.BATCH_SIZE,
                                  stride=self.params.STRIDE, cls_thresh=0.5, name="nms")
        self._model = self.__get_rpn_model()

    def _configure_for_training(self):
        inputs = self._model.inputs
        outputs = [self._model.get_layer("concat").output]
        self._model = keras.Model(inputs=inputs, outputs=outputs)

    def _configure_for_running(self):
        inputs = self._model.inputs
        proposals = self._model.get_layer("concat").output
        outputs = [self.nms_layer(proposals)]
        self._model = keras.Model(inputs=inputs, outputs=outputs)

    def _get_training_model(self):
        input = Input(shape=self.input_shape)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='rpn_conv')(input)
        x = Dropout(0.1)(x)
        b_reg = Conv2D(self.params.num_anchors() * 4, (1, 1), name='b_reg')(x)
        a_reg = Conv2D(self.params.num_anchors(), (1, 1), activation='softmax', name='a_reg')(x)
        cls = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='cls')(x)
        proposals = Concatenate(name='concat')([b_reg, a_reg, cls])

        return keras.Model(inputs=[input], outputs=[proposals])


class ClassifierBlock(ModelBlock):
    def __init__(self):
        self.params = Parameters()
        super().__init__()
        self._model.summary()

    def _configure_for_training(self):
        input = Input(shape=(7 * 7 * 512))
        x = Dense.from_config(self._model.layers[-3].get_config())(input)
        x = Dense.from_config(self._model.layers[-2].get_config())(x)
        x = Dense.from_config(self._model.layers[-1].get_config())(x)
        self.cls_model = keras.Model(inputs=[input], outputs=[x])
        self.cls_model.layers[1].set_weights(self._model.layers[-3].get_weights())
        self.cls_model.layers[2].set_weights(self._model.layers[-2].get_weights())
        self.cls_model.layers[3].set_weights(self._model.layers[-1].get_weights())

    def _configure_for_running(self):
        # self._model = LoopedDense()(self.roi_pooling_layer)
        input = Input(shape=(7*7*512))
        x = Dense.from_config(self._model.layers[-3].get_config())(input)
        x = Dense.from_config(self._model.layers[-2].get_config())(x)
        x = Dense.from_config(self._model.layers[-1].get_config())(x)
        self.cls_model = keras.Model(inputs=[input], outputs=[x])
        self.cls_model.layers[1].set_weights(self._model.layers[-3].get_weights())
        self.cls_model.layers[2].set_weights(self._model.layers[-2].get_weights())
        self.cls_model.layers[3].set_weights(self._model.layers[-1].get_weights())
        pass

    def _get_training_model(self):
        # input = Input(shape=(7, 7, 512))
        inputs = [Input(shape=(28, 28, 512)), Input(shape=(10, 4))]
        x = ROIPooling(self.params.BATCH_SIZE, stride=self.params.STRIDE, output_size=[7, 7], training=True,
                            name='roi_pooling')(inputs)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(self.params.NUM_CLASSES, activation='softmax')(x)
        return keras.Model(inputs=inputs, outputs=[output])

    def __get_roi_pooling_model(self):
        inputs = [Input(shape=(28, 28, 512)), Input(shape=(10, 4))]
        output = ROIPooling(self.params.BATCH_SIZE, stride=self.params.STRIDE, output_size=[7, 7], training=False,
                   name='roi_pooling')(inputs)
        return keras.Model(inputs=inputs, outputs=[output])


if __name__ == "__main__":
    cls = ClassifierBlock()

