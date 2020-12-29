from config import *
from utility.data_sourcer import *
from utility.custom_layers import *
from utility.model_manager import ModelManager
from utility.util import *

from tensorflow.keras.layers import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from matplotlib import pyplot as plt

DOG_ID = 'n02084071'
KIT_FOX_ID = 'n02119789'
AIRPLANE_ID = 'n04552348'
GOLF_BALL_ID = 'n03445777'
IDS = [DOG_ID, KIT_FOX_ID, AIRPLANE_ID, GOLF_BALL_ID]
IDS_DICT = {'dog': DOG_ID, 'fox': KIT_FOX_ID, 'airplane': AIRPLANE_ID, 'golfball': GOLF_BALL_ID}


if __name__ == "__main__":
    # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
    #                                   allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1})
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    params = Parameters()
    params.IMAGE_WIDTH = 448
    params.IMAGE_HEIGHT = 448
    params.ANCHOR_RATIOS = [0.6041739339219643, 0.9994668733549451, 1.8924804524684733]
    params.ANCHOR_SCALES = [76.8881118881119, 160.56993006993008, 260.7412587412587]
    params.VALIDATION_SPLIT = 0.05
    params.STRIDE = 16
    params.NUM_CLASSES = 2

    sourcer = DataSourcer()
    model_manager = ModelManager()

    split = None
    output_file_path = 'utility/data_sets/test'
    data_name = 'multi'
    rpn_path = 'utility/models/test_rpn'
    cls_path = 'utility/models/test_cls'
    net_path = 'utility/models/test_net'


    # =================================
    # Data Prep
    # =================================

    # with open('utility/data_sets/ball_images.pickle', 'rb') as f:
    #     ball_images = pickle.load(f)
    # with open('utility/data_sets/ball_bboxes.pickle', 'rb') as f:
    #     ball_bboxes = pickle.load(f)
    # ball_images, ball_bboxes = sourcer.rotate_expansion(ball_images, ball_bboxes)
    #
    # with open('utility/data_sets/ukulele_images.pickle', 'rb') as f:
    #     uk_images = pickle.load(f)
    # with open('utility/data_sets/ukulele_bboxes.pickle', 'rb') as f:
    #     uk_bboxes = pickle.load(f)
    # uk_images, uk_bboxes = sourcer.rotate_expansion(uk_images, uk_bboxes)
    #
    # slice = min(ball_images.shape[0], int(uk_images.shape[0]/2))
    # ball_images = ball_images[:slice]
    # ball_bboxes = ball_bboxes[:slice]
    # uk_images = uk_images[:slice*2]
    # uk_bboxes = uk_bboxes[:slice*2]
    #
    # ball_cls = np.array([[1, 0] for i in range(slice)])
    # uk_cls = np.array([[0, 1] for i in range(slice*2)])
    #
    # images = np.concatenate((ball_images, uk_images), axis=0)
    # bboxes = np.concatenate((ball_bboxes, uk_bboxes), axis=0)
    # cls = np.concatenate((ball_cls, uk_cls), axis=0)
    # print(images.shape, bboxes.shape, cls.shape)
    #
    # anchors = smart_calc(bboxes)
    # params.ANCHOR_RATIOS = anchors[0]
    # params.ANCHOR_SCALES = anchors[1]
    # print(anchors)
    #
    # indices = np.random.choice(range(images.shape[0]), (1200,), replace=False)
    # images = images[indices]
    # bboxes = bboxes[indices]
    # cls = cls[indices]
    #
    # x_set, y_set = sourcer.create_bbox_set(images, bboxes)
    # os.makedirs(output_file_path, exist_ok=True)
    # sourcer.split_write(output_file_path, data_name + '_images', images, split)
    # sourcer.split_write(output_file_path, data_name + '_bboxes', bboxes, split)
    # sourcer.split_write(output_file_path, data_name + '_cls', cls, split)
    # sourcer.split_write(output_file_path, data_name + '_x_set', x_set, split)
    # sourcer.split_write(output_file_path, data_name + '_y_set', y_set, split)

    # =================================
    # RPN Model Gen and Training
    # =================================

    # params.EPOCHS = 8
    # params.BATCH_SIZE = 3
    #
    # x_set = sourcer.split_read(output_file_path, data_name + '_x_set', split)
    # y_set = sourcer.split_read(output_file_path, data_name + '_y_set', split)
    # print(x_set.shape, y_set.shape)
    #
    # model = VGG16(input_shape=(params.IMAGE_HEIGHT, params.IMAGE_WIDTH, 3), include_top=False)
    # input = Input((params.IMAGE_HEIGHT, params.IMAGE_WIDTH, 3), name="Input")
    # x = input
    # for layer in model.layers[1:-1]:
    #     layer.trainable = False
    #     x = layer(x)
    # num_anchors = params.num_anchors()
    #
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='rpn_conv')(x)
    # x = Dropout(0.1)(x)
    # b_reg = Conv2D(num_anchors * 4, (1, 1), name='b_reg')(x)
    # a_reg = Conv2D(num_anchors, (1, 1), activation='softmax', name='a_reg')(x)
    # cls = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='cls')(x)
    #
    # output = Concatenate(name='concat')([b_reg, a_reg, cls])
    #
    # model = keras.Model(inputs=[input], outputs=[output])
    # model.summary()
    #
    # lr_schedule = ExponentialDecay(0.01, decay_steps=int((x_set.shape[0]/params.BATCH_SIZE)*params.EPOCHS/3), decay_rate=0.1, staircase=True)
    # model.compile(optimizer=SGD(learning_rate=lr_schedule), loss=ModelManager.rpn_loss)
    # fit_data = model.fit(x_set, y_set, params.BATCH_SIZE, params.EPOCHS, validation_split=0.05)
    # model.save(rpn_path)

    # plt.plot(fit_data.history['loss'], label='loss')
    # plt.plot(fit_data.history['val_loss'], label='val_loss')
    # plt.ylim(0, 0.2)
    # plt.legend()
    # plt.show()

    # =======================================
    # Classification Model Gen and Training
    # =======================================

    params.BATCH_SIZE = 16
    params.EPOCHS = 30

    x_set = sourcer.split_read(output_file_path, data_name + '_x_set', split)
    y_set = sourcer.split_read(output_file_path, data_name + '_cls', split)

    model = keras.models.load_model(rpn_path, compile=False)
    for layer in model.layers:
        layer.trainable = False
    feature_map = model.get_layer('block5_conv3').output
    proposals = model.get_layer('concat').output

    nms_layer = NMSLayer(params.get_anchors(), batch_size=params.BATCH_SIZE, stride=params.STRIDE, cls_thresh=0.5, name='nms')
    pooling = ROIPooling(params.BATCH_SIZE, stride=params.STRIDE, output_size=[7, 7], training=True, name='pooling')

    bboxes = nms_layer(proposals)
    pooling_out = pooling([feature_map, bboxes])

    bot_input = Input(shape=(7, 7, 512))
    x = Flatten()(bot_input)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(params.NUM_CLASSES, activation='softmax')(x)
    top_model = keras.Model(inputs=model.inputs, outputs=[pooling_out])
    dense_model = keras.Model(inputs=[bot_input], outputs=[output])

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    lr_schedule = ExponentialDecay(0.01, decay_steps=int((x_set.shape[0] / params.BATCH_SIZE) * params.EPOCHS / 3),
                                   decay_rate=0.1, staircase=True)
    optimizer = SGD(learning_rate=lr_schedule)

    for batch in range(int(len(x_set)/params.BATCH_SIZE)):
        x_batch = x_set[batch*params.BATCH_SIZE:(batch+1)*params.BATCH_SIZE]
        y_batch = y_set[batch * params.BATCH_SIZE:(batch + 1) * params.BATCH_SIZE]
        feed = top_model(x_batch)
        with tf.GradientTape() as tape:
            logits = dense_model(feed)
            loss_value = loss_fn(y_batch, logits)
        gradients = tape.gradient(loss_value, dense_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, dense_model.trainable_weights))
        print(loss_value)

    print("Classification Training Complete")

    # Reorganizing for Implementation

    nms_layer.batch_size = 1
    nms_layer.cls_thresh = 0.85
    nms_layer.max_iou = 0.05
    pooling.batch_size = 1
    pooling.training = False

    inputs = model.inputs

    feat_out = model.get_layer('block5_conv3').output
    rpn_out = model.get_layer('concat').output
    bboxes = nms_layer(rpn_out)
    pooling_out = pooling([feat_out, bboxes])

    cls = LoopedDense(dense_model, params.NUM_CLASSES)(pooling_out)
    net = keras.Model(inputs, [bboxes, cls])
    for layer in net.layers:
        layer.trainable = False
    net.summary()
    net.save(net_path)

