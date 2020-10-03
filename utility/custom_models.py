from config import *
from utility.custom_layers import *


from tensorflow.keras.layers import *
from enum import Enum


class ModelBlock:
    class BlockState(Enum):
        TRAINING = 0,
        RUNNING = 1,
        ALT = 2

    def __init__(self, input_model=None):
        if input_model:
            self._model = input_model
            self.pre_configured = False
            self.state = ModelBlock.BlockState.ALT
        else:
            self._model = self._get_training_model()
            self.state = ModelBlock.BlockState.TRAINING

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
        if self.state is not ModelBlock.BlockState.ALT:
           self.configure(ModelBlock.BlockState.RUNNING)
        self._model.save(filepath)

    def load(self, filepath):
        self._model = keras.models.load_model(filepath)
        self.state = ModelBlock.BlockState.RUNNING

    @staticmethod
    def concat_blocks(block_array):
        first_run = True
        for block in block_array:
            model = block.get_model()
            if first_run:
                inputs = model.inputs
                x = model(inputs)
                first_run = False
            else:
                x = model(x)
        output_model = keras.Model(inputs=inputs, outputs=[x])
        new_block = ModelBlock(output_model)
        return new_block

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

    def get_model(self):
        return self._model

    def get_output_shape(self):
        return self._model.output_shape[1:]

    def summary(self):
        self._model.summary()

    def _configure_for_training(self): pass

    def _configure_for_running(self): pass

    def _get_training_model(self): pass


class FeatureExtractorBlock(ModelBlock):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super().__init__()

    def _get_training_model(self):
        vgg = keras.applications.VGG16(input_shape=self.input_shape, include_top=False)
        inputs = vgg.inputs
        x = inputs[0]
        for layer in vgg.layers[:-1]:
            x = layer(x)
        model = keras.Model(inputs=inputs, outputs=[x], name="vgg16")
        return model


class RPNBlock(ModelBlock):
    def __init__(self, input_shape):
        self.params = Parameters()
        self.input_shape = input_shape
        self.nms_layer = NMSLayer(self.params.get_anchors(), batch_size=self.params.BATCH_SIZE,
                                  stride=self.params.STRIDE, cls_thresh=0.5, name="nms")
        super().__init__()

    def _configure_for_training(self):
        inputs = self._model.inputs
        self.nms_layer.batch_size = self.params.BATCH_SIZE
        outputs = [self._model.get_layer("concat").output]
        self._model = keras.Model(inputs=inputs, outputs=outputs, name="rpn")

    def _configure_for_running(self):
        inputs = self._model.inputs
        proposals = self._model.get_layer("concat").output
        self.nms_layer.batch_size = self.params.BATCH_SIZE
        outputs = [self._model.inputs[0], self.nms_layer(proposals)]
        self._model = keras.Model(inputs=inputs, outputs=outputs, name="rpn")

    def _get_training_model(self):
        input = Input(shape=self.input_shape)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='rpn_conv')(input)
        x = Dropout(0.1)(x)
        b_reg = Conv2D(self.params.num_anchors() * 4, (1, 1), name='b_reg')(x)
        a_reg = Conv2D(self.params.num_anchors(), (1, 1), activation='softmax', name='a_reg')(x)
        cls = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='cls')(x)
        proposals = Concatenate(name='concat')([b_reg, a_reg, cls])

        return keras.Model(inputs=[input], outputs=[proposals], name='rpn')


class ClassifierBlock(ModelBlock):
    def __init__(self):
        self.params = Parameters()
        super().__init__()

    def _configure_for_training(self):
        inputs = [Input(shape=(28, 28, 512)), Input(shape=(10, 4))]
        pooling = self._model.get_layer('roi_pooling')
        pooling.training = True
        x = pooling(inputs)
        x = self._model.get_layer('looped_dense').get_dense_model()(x)
        self._model = keras.Model(inputs=inputs, outputs=[x], name="classifier")

    def _configure_for_running(self):
        inputs = [Input(shape=(28, 28, 512)), Input(shape=(10, 4))]
        pooling = self._model.get_layer('roi_pooling')
        pooling.training = False
        # pooling.batch_size = 1
        x = pooling(inputs)
        x = LoopedDense(self._model.get_layer('dense_layers'), self.params.NUM_CLASSES, name="looped_dense")(x)
        self._model = keras.Model(inputs=inputs, outputs=[inputs[1], x], name="classifier")

    def _get_training_model(self):
        inputs = [Input(shape=(28, 28, 512)), Input(shape=(10, 4))]
        x = ROIPooling(self.params.BATCH_SIZE, stride=self.params.STRIDE, output_size=[7, 7], training=True,
                            name='roi_pooling')(inputs)
        x = Flatten()(x)
        output = self.__get_dense_layers()(x)
        return keras.Model(inputs=inputs, outputs=[output], name='classifier')

    def __get_dense_layers(self):
        input = Input(shape=(25088,))
        x = Dense(512, activation='relu')(input)
        x = Dense(512, activation='relu')(x)
        output = Dense(self.params.NUM_CLASSES, activation='softmax')(x)
        return keras.Model(inputs=[input], outputs=[output], name='dense_layers')

    # def __get_roi_pooling_model(self):
    #     inputs = [Input(shape=(28, 28, 512)), Input(shape=(10, 4))]
    #     output = ROIPooling(self.params.BATCH_SIZE, stride=self.params.STRIDE, output_size=[7, 7], training=False,
    #                name='roi_pooling')(inputs)
    #     return keras.Model(inputs=inputs, outputs=[output])


if __name__ == "__main__":
    cls = ClassifierBlock()

