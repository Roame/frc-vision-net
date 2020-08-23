from config import *
from utility.custom_layers import NMSLayer, ROIPooling, LoopedDense, NMSLayerV2
from scipy.stats import zscore
import multiprocessing
from multiprocessing import shared_memory
import keyboard


def anchor_to_bbox(anchor, x, y):
    width = anchor[1]
    height = int(anchor[1] / anchor[0])
    return [x, y, width, height]


def apply_deltas(bbox, deltas):
    bbox[0] = (deltas[0]*bbox[2])+bbox[0]
    bbox[1] = (deltas[1] * bbox[3]) + bbox[1]
    bbox[2] = math.exp(deltas[2])*bbox[2]
    bbox[3] = math.exp(deltas[3]) * bbox[3]
    return bbox


def bbox_to_coords(bbox):
    x1 = int(bbox[0] - bbox[2] / 2)
    x2 = int(bbox[0] + bbox[2] / 2)
    y1 = int(bbox[1] - bbox[3] / 2)
    y2 = int(bbox[1] + bbox[3] / 2)
    return [x1, y1, x2, y2]


# For balls:
# anchor_ratios = [0.66, 1.0, 1.5]
# anchor_scales = [50, 90, 205]

# For ukuleles:
# anchor_ratios = [0.5, 1.0, 2.25]
# anchor_scales = [115, 200, 285]

# For example RPN:
# anchor_ratios = [0.6632113935243945, 0.9629369521740117, 1.7071313877501033]
# anchor_scales = [143, 286, 425]
# anchor_ratios = [0.25, 0.9623383197898614, 2.5]
# anchor_scales = [142, 285, 425]

# For multi RPN:
anchor_ratios = [0.6041739339219643, 0.9994668733549451, 1.8924804524684733]
anchor_scales = [76.8881118881119, 160.56993006993008, 260.7412587412587]

anchors = [[r, s] for r in anchor_ratios for s in anchor_scales]

BATCH_SIZE = 1
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
STRIDE = 16


class ObjectDetection:
    def __init__(self, mode='multi'):
        model = keras.models.load_model('models/example_net.h5', compile=False,
                                        custom_objects={'NMSLayerV2': NMSLayerV2,
                                                        'ROIPooling': ROIPooling,
                                                        'LoopedDense': LoopedDense})
        # self.model = model
        model.layers[0].batch_input_shape = (None, 240, 320, 3)
        new_model = keras.models.model_from_json(model.to_json(), custom_objects={'NMSLayerV2': NMSLayerV2,
                                                                                  'ROIPooling': ROIPooling,
                                                                                  'LoopedDense': LoopedDense})
        for layer in new_model.layers:
            layer.trainable = False
            try:
                layer.set_weights(model.get_layer(name=layer.name).get_weights())
            except:
                print("ah man")
        self.model = new_model
        self.model.summary()
        self.mode = mode

        self.lock = multiprocessing.Lock()
        self.shm_image = shared_memory.SharedMemory(create=True, size=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8).nbytes)
        self.shm_proc = shared_memory.SharedMemory(create=True, size=np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float64).nbytes)
        self.shm_control = shared_memory.SharedMemory(create=True, size=2)
        self.shm_control.buf[:] = bytearray([1, 0])

        self.frame_grabber = multiprocessing.Process(target=self.frame_grabbing, args=(self.lock, self.shm_image, self.shm_proc, self.shm_control))
        self.frame_grabber.start()

        self.image_obj = np.ndarray((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8, buffer=self.shm_image.buf)
        self.proc_obj = np.ndarray((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float64, buffer=self.shm_proc.buf)

        self.fps_start = 0
        self.running_fps = 0
        self.fps_a = 1 / 30

        self.nms = NMSLayer(anchors=anchors, batch_size=1, num_proposals=2)

    @staticmethod
    def frame_grabbing(lock, shm_image, shm_proc, shm_control):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        img_obj = np.ndarray((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8, buffer=shm_image.buf)
        proc_obj = np.ndarray((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float64, buffer=shm_proc.buf)

        run_val = 1
        while run_val:
            _, image = cap.read()
            image = cv2.resize(image, (320, 240))
            # image = image[16:464, 96:544]
            processed = np.expand_dims(zscore(image), axis=0)

            lock.acquire()
            img_obj[:] = image[:]
            proc_obj[:] = processed[:]
            shm_control.buf[1] = 1
            run_val = shm_control.buf[0]
            lock.release()

        shm_image.close()
        shm_image.unlink()
        shm_proc.close()
        shm_proc.unlink()
        shm_control.close()
        shm_control.unlink()

    def run(self):
        fps_elapse = time.time() - self.fps_start
        self.fps_start = time.time()

        start = time.time()
        ready = 0
        while not ready:
            self.lock.acquire()
            ready = self.shm_control.buf[1]
            self.lock.release()
        cam_elapse = time.time() - start

        self.lock.acquire()
        test_image = self.image_obj[:]
        model_input = self.proc_obj[:]
        self.shm_control.buf[1] = 0
        self.lock.release()

        start = time.time()
        prediction = self.model.predict(model_input)
        network_elapse = time.time() - start

        if self.mode == 'rpn':
            # bboxes = non_max_suppression(prediction)
            bboxes = self.nms.call(prediction)[0]
            for bbox in bboxes:
                coords = bbox_to_coords(bbox)
                cv2.rectangle(test_image, (coords[0], coords[1]), (coords[2], coords[3]), color=(255, 0, 0),
                              thickness=3)
                cv2.circle(test_image, (int(bbox[0]), int(bbox[1])), 6, (0, 0, 255), -1)
        elif self.mode == 'multi':
            bboxes = prediction[0][0]
            classes = prediction[1][0]
            for index, bbox in enumerate(bboxes):
                cls = classes[index]
                coords = bbox_to_coords(bbox)
                color = (200, 222, 55) if cls[1] > cls[0] else (32, 230, 226)
                cv2.rectangle(test_image, (coords[0], coords[1]), (coords[2], coords[3]), color=color, thickness=3)
                cv2.circle(test_image, (int(bbox[0]), int(bbox[1])), 6, (0, 0, 255), -1)

        self.running_fps = 1 / fps_elapse * self.fps_a + self.running_fps * (1 - self.fps_a)
        cv2.putText(test_image, str(int(self.running_fps)) + " FPS", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        # test_image = cv2.resize(test_image, (640, 480))
        cv2.imshow('test', test_image)
        cv2.waitKey(1)

        print("Cam Delay: {:3.1f} ms, Forward Prop: {:3.1f} ms".format(
            cam_elapse * 1000, network_elapse * 1000))

    def stop(self):
        cv2.destroyAllWindows()
        self.lock.acquire()

        self.shm_control.buf[0] = 0
        self.shm_image.close()
        self.shm_image.unlink()
        self.shm_proc.close()
        self.shm_proc.unlink()
        self.shm_control.close()
        self.shm_control.unlink()

        self.lock.release()
        self.frame_grabber.join()


if __name__ == "__main__":
    # model = keras.models.load_model('../example.h5', compile=False, custom_objects={'NMSLayerV2': NMSLayerV2, 'ROIPooling': ROIPooling})
    # model.summary()
    # rpn_out = model.get_layer('concat').output
    # bboxes = NMSLayerV2(anchors=anchors, batch_size=1, cls_thresh=0.5, max_iou=0.1)(rpn_out)
    #
    # maps = model.get_layer('block5_conv3').output
    # pooled_maps = ROIPooling(batch_size=1, training=False)([maps, bboxes])
    # cls = LoopedDense(multi_model=model, num_classes=2)(pooled_maps)
    # model = keras.Model(inputs=model.inputs, outputs=[bboxes, cls])
    # model.summary()
    #
    # with open('data_sets/multi_images.pickle', 'rb') as f:
    #     images = pickle.load(f)
    # with open('data_sets/multi_cls.pickle', 'rb') as f:
    #     y_set = pickle.load(f)
    #
    # # for set in range(0, 200):
    # #     test_image = images[set]
    # cap = cv2.VideoCapture(0)
    # while 1:
    #     _, test_image = cap.read()
    #     test_image = test_image[16:-16, 96:-96]
    #     image = zscore(test_image)
    #
    #     model_input = np.expand_dims(image, axis=0)
    #     prediction = model.predict(model_input, batch_size=1)
    #
    #     for i in range(4):
    #         bbox = prediction[0][0, i]
    #         coords = bbox_to_coords(bbox)
    #         cls_pred = prediction[1][0, i]
    #         color = (200, 222, 55) if cls_pred[1] > cls_pred[0] else (32, 230, 226)
    #         cv2.rectangle(test_image, (coords[0], coords[1]), (coords[2], coords[3]), color=color, thickness=3)
    #
    #     print(prediction[1][0, 0])
    #     # print(y_set[set])
    #
    #
    #     cv2.imshow('test', test_image)
    #     cv2.waitKey(10)
    # cv2.destroyAllWindows


    # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
    #                                   allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1})
    # config.gpu_options.allow_growth = True
    # # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # sess = tf.compat.v1.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)
    #
    # model = keras.models.load_model('../example_rpn.h5', compile=False)
    # # model = keras.models.load_model('../ball_rpn1.h5', compile=False)
    #
    # model.summary()
    # with open('data_sets/test/multi_images.pickle', 'rb') as f:
    #     images = pickle.load(f)
    # with open('data_sets/test/multi_y_set.pickle', 'rb') as f:
    #     y_set = pickle.load(f)
    # with open('data_sets/test/multi_cls.pickle', 'rb') as f:
    #     cls_set = pickle.load(f)
    # indices = np.random.choice(range(images.shape[0]), (500,), replace=False)
    # images = images[indices]
    # y_set = y_set[indices]
    # cls_set = cls_set[indices]
    # print(images.shape)
    # print(y_set.shape)
    #
    # for set in range(0, len(images)):
    #     cls = cls_set[set]
    #     print(cls)
    #     test_image = images[set]
    #     image = zscore(test_image)
    #     model_input = np.expand_dims(image, axis=0)
    #     prediction = model.predict(model_input, batch_size=1)
    #
    #     prediction[:, 0, :, -1] = 0
    #     prediction[:, -1, :, -1] = 0
    #     prediction[:, :, 0, -1] = 0
    #     prediction[:, :, -1, -1] = 0
    #
    #     cls_scr = prediction[:, :, :, -1:]
    #     cls_true = np.expand_dims(y_set[set, :, :, -1], axis=0)
    #
    #     index = np.argwhere(cls_scr == np.amax(cls_scr))[0]
    #     # index = np.argwhere(cls_true == np.amax(cls_true))[0]
    #
    #     a_sel_pred = prediction[index[0], index[1], index[2], 36:-1]
    #     # a_sel_pred = y_set[set, index[1], index[2], 36:-1]
    #     anchor_i = np.argmax(a_sel_pred)
    #
    #     cv2.circle(test_image, (index[2] * 16 + 8, index[1] * 16 + 8), 7, (0, 255, 0), -1)
    #     try:
    #         bbox = apply_deltas(anchor_to_bbox(anchors[anchor_i], index[2] * 16 + 8, index[1] * 16 + 8),
    #                             prediction[index[0], index[1], index[2], anchor_i * 4:(anchor_i+1) * 4])
    #     except:
    #         continue
    #
    #     coords = bbox_to_coords(bbox)
    #     if np.amax(coords) > 1000:
    #         coords = [0, 0, 1, 1]
    #     cv2.rectangle(test_image, (coords[0], coords[1]), (coords[2], coords[3]), color=(0, 255, 0), thickness=7)
    #
    #     pred_a_bbox = anchor_to_bbox(anchors[anchor_i], index[2] * 16 + 8, index[1] * 16 + 8)
    #     coords = bbox_to_coords(pred_a_bbox)
    #     cv2.rectangle(test_image, (coords[0], coords[1]), (coords[2], coords[3]), color=(255, 255, 255), thickness=7)
    #
    #     print(anchor_i)
    #
    #     a_sel_true = y_set[set, index[1], index[2], 36:-1]
    #     anchor_i = np.argmax(a_sel_true)
    #     print(anchor_i)
    #     g_bbox = apply_deltas(anchor_to_bbox(anchors[anchor_i], index[2] * 16 + 8, index[1] * 16 + 8),
    #                         y_set[set][index[1], index[2], anchor_i * 4:(anchor_i+1) * 4])
    #     coords = bbox_to_coords(g_bbox)
    #     cv2.rectangle(test_image, (coords[0], coords[1]), (coords[2], coords[3]), color=(255, 0, 0), thickness=1)
    #
    #     a_bbox = anchor_to_bbox(anchors[anchor_i], index[2] * 16 + 8, index[1] * 16 + 8)
    #     coords = bbox_to_coords(a_bbox)
    #     print(U.calc_iou(g_bbox, a_bbox))
    #     print(U.calc_iou(g_bbox, pred_a_bbox))
    #     cv2.rectangle(test_image, (coords[0], coords[1]), (coords[2], coords[3]), color=(255, 255, 255), thickness=1)
    #
    #     cv2.imshow('test', test_image)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows

    obj_detect = ObjectDetection()
    while not keyboard.is_pressed('q'):
        obj_detect.run()
    obj_detect.stop()

