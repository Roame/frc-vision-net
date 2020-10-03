from config import *
from utility.data_sourcer import *
from utility.custom_layers import *
from utility.custom_models import *
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
    # Setting up session
    # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
    #                                   allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1})
    # config.gpu_options.per_process_gpu_memory_fraction = 0.75
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)

    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])

    params = Parameters()
    params.IMAGE_WIDTH = 448
    params.IMAGE_HEIGHT = 448
    params.ANCHOR_RATIOS = [0.6041739339219643, 0.9994668733549451, 1.8924804524684733]
    params.ANCHOR_SCALES = [76.8881118881119, 160.56993006993008, 260.7412587412587]
    params.BATCH_SIZE = 16
    params.EPOCHS = 1  # 8 epochs for rpn and 10*3 for cls, start at lr of 0.001 for cls, rpn get decay from 0.01
    params.NUM_VAL_BATCHES = 2
    params.STRIDE = 16
    params.NUM_CLASSES = 2

    sourcer = DataSourcer()
    model_manager = ModelManager()

    split = None
    output_file_path = 'utility/data_sets/test'
    data_name = 'multi'
    extractor_path = 'utility/models/test_extractor'
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
    # anchors = smart_calc_anchors(bboxes)
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

    # x_set = sourcer.split_read(output_file_path, data_name + '_x_set', split)
    # y_set = sourcer.split_read(output_file_path, data_name + '_y_set', split)
    # print("Data Shape: ", x_set.shape, y_set.shape)
    #
    # feat_extractor = FeatureExtractorBlock((params.IMAGE_HEIGHT, params.IMAGE_WIDTH, 3))
    # feat_extractor.freeze_layers()
    # rpn = RPNBlock(feat_extractor.get_output_shape())
    #
    # full_rpn = ModelBlock.concat_blocks([feat_extractor, rpn])
    # full_rpn.summary()
    #
    # lr_schedule = ExponentialDecay(0.01, decay_steps=int((x_set.shape[0]/params.BATCH_SIZE)*params.EPOCHS/3), decay_rate=0.1, staircase=True)
    # opt = SGD(learning_rate=lr_schedule)
    # loss_fn = ModelManager.rpn_loss
    # model_manager.train_model(x_set, y_set, None, full_rpn, loss_fn, opt)
    #
    # feat_extractor.save(extractor_path)
    # rpn.save(rpn_path)
    # # rpn.summary()

    # =======================================
    # Classification Model Gen and Training
    # =======================================

    x_set = sourcer.split_read(output_file_path, data_name + '_x_set', split)
    y_set = sourcer.split_read(output_file_path, data_name + '_cls', split)

    extractor = FeatureExtractorBlock((params.IMAGE_HEIGHT, params.IMAGE_WIDTH, 3))
    extractor.load(extractor_path)
    extractor.freeze_model()

    rpn = RPNBlock(extractor.get_output_shape())
    rpn.load(rpn_path)
    rpn.freeze_model()
    rpn.get_model().get_layer('nms').batch_size = params.BATCH_SIZE
    rpn.summary()
    # rpn.summary()

    cls = ClassifierBlock()

    feat_model = extractor.get_model()
    feat_model.summary()
    rpn_model = rpn.get_model()
    rpn_model.summary()
    cls_model = cls.get_model()
    cls_model.summary()

    input = Input(shape=(448, 448, 3))
    feat_out = feat_model(input)
    rpn_out = rpn_model(feat_out)
    # output = cls_model([feat_out, rpn_out])

    prep_model = ModelBlock(keras.Model([input], [rpn_out]))
    end_model = cls
    prep_model.summary()
    end_model.summary()

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    opt = SGD(learning_rate=0.01)

    model_manager.train_model(x_set, y_set, prep_model, cls, loss_fn, opt)

    cls.save(cls_path)
    output = cls.get_model()(rpn_out)
    full_model = ModelBlock(keras.Model([input], [output]))
    full_model.save(net_path)
    full_model.summary()

    # =======================================
    # Reorganizing for Implementation
    # =======================================

    # model = keras.models.load_model('example.h5', compile=False,
    #                                 custom_objects={'NMSLayerV2': NMSLayerV2, 'ROIPooling': ROIPooling})
    # for layer in model.layers:
    #     layer.trainable = False
    # rpn_out = model.get_layer('concat').output
    # bboxes = NMSLayerV2(anchors=params.get_anchors(), batch_size=1, cls_thresh=0.85, max_iou=0.05)(rpn_out)
    #
    # maps = model.get_layer('block5_conv3').output
    # pooled_maps = ROIPooling(batch_size=1, training=False)([maps, bboxes])
    # cls = LoopedDense(multi_model=model, num_classes=2)(pooled_maps)
    # model = keras.Model(inputs=model.inputs, outputs=[bboxes, cls])
    # model.summary()
    # model.save('utility/models/example_net', include_optimizer=False)
