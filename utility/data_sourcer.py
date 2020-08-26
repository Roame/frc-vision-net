from config import *
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as et
import tarfile
import os
import numpy as np
import cv2
from tqdm import tqdm
from .util import *
import multiprocessing
import pickle
from scipy.stats import zscore


class DataSourcer:
    def __init__(self):
        self.params = Parameters()

    @staticmethod
    def get_mapping(url):
        # returns dictionary of the form filename:url
        response = requests.get(url)
        str_soup = str(BeautifulSoup(response.content, 'html.parser'))
        split_mappings = str_soup.split('\r\n')
        map_dict = {entry[0]: entry[1] for entry in [pair.split() for pair in split_mappings if pair != '']}
        return map_dict

    @staticmethod
    def get_file_data(data_path):
        # returns dictionary of the form filename:coords
        print('Extracting bbox info...')
        raw_data = tarfile.open(data_path, 'r:gz')
        raw_data.extractall('utility/extracted_data')
        raw_data.close()
        data_dict = {}
        path = os.path.join('utility/extracted_data/Annotation', data_path[-16:-7])
        for xml in os.listdir(path):
            xml_path = os.path.join(path, xml)
            root = et.parse(xml_path).getroot()
            filename = root.find('filename').text
            x1 = int(root.find('object/bndbox/xmin').text)
            y1 = int(root.find('object/bndbox/ymin').text)
            x2 = int(root.find('object/bndbox/xmax').text)
            y2 = int(root.find('object/bndbox/ymax').text)
            data_dict[filename] = [x1, y1, x2, y2]
        return data_dict

    @staticmethod
    def compile_data(image_size, mapping, coords_data, max_count):
        print('Compiling data...')
        images, bboxes = [], []

        keys = list(coords_data.keys())
        target_num_imgs = min(max_count, len(keys))
        for i in tqdm(range(target_num_imgs)):
            filename = keys[i]
            try:
                response = requests.get(mapping[filename])
                image = np.asarray(bytearray(response.content), np.uint8)
                image = cv2.imdecode(image, 1)
                if image is None:
                    continue
                coords = coords_data[filename]
                bbox = coords_to_bbox(coords)
                x_scaling = image_size[0]/image.shape[1]
                y_scaling = image_size[1]/image.shape[0]

                bbox[0] = int(bbox[0]*x_scaling)
                bbox[1] = int(bbox[1]*y_scaling)
                bbox[2] = int(bbox[2]*x_scaling)
                bbox[3] = int(bbox[3]*y_scaling)

                image = cv2.resize(image, image_size)
                images.append(image)
                bboxes.append(bbox)
            except:
                continue
        images = np.stack(images)
        bboxes = np.array(bboxes)
        return images, bboxes

    def create_bbox_set(self, images, bboxes):
        print('Creating data set...')
        stride = self.params.STRIDE
        anchors = self.params.get_anchors()
        pixels_x = images[0].shape[1]
        pixels_y = images[0].shape[0]
        y_set = np.zeros((images.shape[0], int(pixels_y/stride), int(pixels_x/stride), len(anchors)*5+1))

        for image_index in tqdm(range(len(images))):
            t_bbox = bboxes[image_index]
            highest_iou = 0
            iou_index = None
            pos_count = 0
            for y in range(int(stride/2), int(pixels_y+stride/2), stride):
                for x in range(int(stride/2), int(pixels_x+stride/2), stride):
                    anchor_regs = np.zeros((len(anchors)*4))
                    anchor_count = 0
                    local_highest = 0
                    best_anchor = 0
                    for anchor in anchors:
                        a_bbox = anchor_to_bbox(anchor, x, y)
                        iou = calc_iou(t_bbox, a_bbox)
                        if highest_iou < iou:
                            highest_iou = iou
                            iou_index = [y, x, anchor_count]
                        if local_highest < iou:
                            local_highest = iou
                            best_anchor = anchor_count
                        if iou > 0.7:
                            pos_count += 1
                            y_set[image_index, int(y/stride-0.5), int(x/stride-0.5), -1] = 1
                        elif iou < 0.3 and y_set[image_index, int(y / stride - 0.5), int(x / stride - 0.5), -1] != 1:
                            y_set[image_index, int(y / stride - 0.5), int(x / stride - 0.5), -1] = -1
                        delta_x = (t_bbox[0] - a_bbox[0])/a_bbox[2]
                        delta_y = (t_bbox[1] - a_bbox[1])/a_bbox[3]
                        delta_width = math.log(t_bbox[2]/a_bbox[2])
                        delta_height = math.log(t_bbox[3]/a_bbox[3])
                        anchor_regs[(anchor_count * 4):anchor_count*4 + 4] = np.array([delta_x, delta_y, delta_width,
                                                                                                        delta_height])
                        anchor_count += 1
                    y_set[image_index, int(y/stride-0.5), int(x/stride-0.5), :len(anchors)*4] = anchor_regs
                    y_set[image_index, int(y/stride-0.5), int(x/stride-0.5), best_anchor-10] = 1
            if pos_count == 0 and iou_index is not None:
                y_set[image_index, int(iou_index[0]/stride-0.5), int(iou_index[1]/stride-0.5), -1] = 1
                y_set[image_index, int(iou_index[0] / stride - 0.5), int(iou_index[1] / stride - 0.5), iou_index[2]-10] = 1
        x_set = []
        for image in images:
            x_set.append(zscore(image))
        x_set = np.stack(x_set)
        return x_set, y_set

    def create_bbox_set_from_ids(self, ids, data_points_per_id):
        first_run = True
        images, bboxes = None, None
        for id in ids:
            url = 'http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid='+id
            data_path = os.path.join('utility/tar_files', id+'.tar.gz').replace(os.sep, '/')
            mapping_dict = self.get_mapping(url)
            data_dict = self.get_file_data(data_path)
            temp_images, temp_bboxes = self.compile_data(image_size=(self.params.IMAGE_WIDTH, self.params.IMAGE_HEIGHT),
                                                         mapping=mapping_dict, coords_data=data_dict,
                                                         max_count=data_points_per_id)
            if first_run:
                images = temp_images
                bboxes = temp_bboxes
                first_run = False
            else:
                images = np.concatenate((images, temp_images))
                bboxes = np.concatenate((bboxes, temp_bboxes))

        anchors = smart_calc_anchors(bboxes)
        self.params.ANCHOR_RATIOS = anchors[0]
        self.params.ANCHOR_SCALES = anchors[1]
        x_set, y_set = self.create_bbox_set(images, bboxes)
        print(anchors)
        return x_set, y_set, images, bboxes

    def create_cls_set_from_ids(self, id_dict, num_per_class):
        keys = list(id_dict.keys())
        flipped_dict = {value: key for key, value in id_dict.items()}
        one_hot = get_one_hot(keys)
        x_set, y_set = [], []
        q = multiprocessing.Queue()
        processes = []
        rets = []
        for key in keys:
            p = multiprocessing.Process(target=self.source_images, args=(id_dict[key], num_per_class, self.params.IMAGE_WIDTH, self.params.IMAGE_HEIGHT, q))
            processes.append(p)
            p.start()
        for p in processes:
            ret = q.get()
            rets.append(ret)
        for p in processes:
            p.join()
        first_run = True
        for ret in rets:
            x_set = ret[1] if first_run else np.concatenate((x_set, ret[1]))
            first_run = False
            id = ret[0]
            encoding = one_hot[flipped_dict[id]]
            for i in range(ret[1].shape[0]):
                y_set.append(encoding)
        y_set = np.array(y_set)
        return x_set, y_set

    def source_images(self, id, count, image_width, image_height, queue):
        response = requests.get('http://image-net.org/api/text/imagenet.synset.geturls?wnid='+id)
        str_soup = str(BeautifulSoup(response.content, 'html.parser'))
        split_urls = str_soup.split('\r\n')
        count = min(len(split_urls), count)
        images = []
        for i in range(count):
            print(id, i/count)
            try:
                im_response = requests.get(split_urls[i])
                image = np.asarray(bytearray(im_response.content), np.uint8)
                image = cv2.imdecode(image, 1)
                image = cv2.resize(image, (image_width, image_height))
                images.append(image)
            except:
                continue
        queue.put((id, np.stack(images)))

    @staticmethod
    def bbox_set_from_path(file_path, parameters):
        global base_img, img
        img_directories = [os.path.join(file_path, n) for n in os.listdir(file_path)]

        bboxes = []
        images = []

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 600, 600)
        cv2.setMouseCallback("image", DataSourcer.draw_function)
        for path in img_directories:
            base_img = cv2.imread(path, 1)
            base_img = cv2.resize(base_img, (parameters.IMAGE_WIDTH, parameters.IMAGE_HEIGHT))
            img = np.copy(base_img)

            while 1:
                cv2.imshow("image", img)
                key = cv2.waitKey(20)
                if key == ord('n') or key == ord('q') or key == ord('s'):
                    break
            if key == ord('q'):
                break
            if key == ord('s'):
                continue
            x = int(abs((x1 + x2) / 2.))
            y = int(abs((y1 + y2) / 2.))
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            images.append(base_img)
            bboxes.append([x, y, width, height])
        cv2.destroyAllWindows()
        images = np.stack(images)
        bboxes = np.array(bboxes)

        return images, bboxes

    @staticmethod
    def draw_function(event, x, y, flags, param):
        global x1, y1, x2, y2, img
        source = np.copy(base_img)
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            x2, y2, = x, y
        img = cv2.circle(source, (x1, y1), 5, (0, 255, 0), -1)
        img = cv2.circle(img, (x2, y2), 5, (255, 0, 0), -1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)

    @staticmethod
    def rotate_expansion(images, bboxes):
        index = 0
        temp_x_set = []
        temp_y_set = []
        for image in tqdm(images):
            rotated_img = image
            for i in range(3):
                rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE)
                new_x = (bboxes[index][0] - 224) * math.cos((math.pi / 2) * (i + 1)) - (
                        bboxes[index][1] - 224) * math.sin((math.pi / 2) * (i + 1)) + 224
                new_y = (bboxes[index][0] - 224) * math.sin((math.pi / 2) * (i + 1)) + (
                        bboxes[index][1] - 224) * math.cos((math.pi / 2) * (i + 1)) + 224
                if i == 0 or i == 2:
                    new_width = bboxes[index][3]
                    new_height = bboxes[index][2]
                else:
                    new_width = bboxes[index][2]
                    new_height = bboxes[index][3]

                temp_x_set.append(rotated_img)
                temp_y_set.append([new_x, new_y, new_width, new_height])
            index += 1
        temp_x_set = np.array(temp_x_set)
        temp_y_set = np.array(temp_y_set)

        images = np.concatenate((images, temp_x_set))
        bboxes = np.concatenate((bboxes, temp_y_set))
        return images, bboxes

    @staticmethod
    def split_write(file_location, name, tensor, split=3):
        if split is None:
            with open(file_location + '/' + name + '.pickle', 'wb+') as f:
                pickle.dump(tensor, f)
        else:
            length = tensor.shape[0]
            for i in range(split):
                with open(file_location + "/" + name + str(i) + '.pickle', 'wb+') as f:
                    pickle.dump(tensor[int(i * length / split): int((i + 1) * length / split)], f)
        print(name + " successfully saved!")

    @staticmethod
    def split_read(file_location, name, split=3):
        if split is None:
            with open(file_location + '/' + name + '.pickle', 'rb') as f:
                output = pickle.load(f)
        else:
            first_run = True
            for i in range(split):
                with open(file_location + '/' + name + str(i) + '.pickle', 'rb') as f:
                    output = pickle.load(f) if first_run else np.concatenate((output, pickle.load(f)))
                first_run = False
        return output
