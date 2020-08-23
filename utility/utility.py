import numpy as np
import multiprocessing
from statistics import mean
import math

class Utility:
    @staticmethod
    def calc_iou(bbox1, bbox2):
        coords1 = Utility.bbox_to_coords(bbox1)
        coords2 = Utility.bbox_to_coords(bbox2)

        x_overlap = max(0, min(coords1[2], coords2[2]) - max(coords1[0], coords2[0]))
        y_overlap = max(0, min(coords1[3], coords2[3]) - max(coords1[1], coords2[1]))
        intersection = x_overlap * y_overlap
        union = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection
        return intersection / union

    @staticmethod
    def bbox_to_coords(bbox):
        x1 = int(bbox[0] - bbox[2] / 2)
        x2 = int(bbox[0] + bbox[2] / 2)
        y1 = int(bbox[1] - bbox[3] / 2)
        y2 = int(bbox[1] + bbox[3] / 2)
        return [x1, y1, x2, y2]

    @staticmethod
    def coords_to_bbox(coords):
        x = int((coords[0] + coords[2]) / 2)
        y = int((coords[1] + coords[3]) / 2)
        width = abs(coords[0] - coords[2])
        height = abs(coords[1] - coords[3])
        return [x, y, width, height]

    @staticmethod
    def anchor_to_bbox(anchor, x, y):
        width = anchor[1]
        height = int(anchor[1]/anchor[0])
        return [x, y, width, height]

    @staticmethod
    def apply_deltas(bbox, deltas):
        bbox[0] = (deltas[0] * bbox[2]) + bbox[0]
        bbox[1] = (deltas[1] * bbox[3]) + bbox[1]
        bbox[2] = math.exp(deltas[2]) * bbox[2]
        bbox[3] = math.exp(deltas[3]) * bbox[3]
        return bbox

    @staticmethod
    def smart_calc(ground_bboxes):
        ratios = []
        scales = []
        for bbox in ground_bboxes:
            ratios.append(bbox[2]/bbox[3])
            scales.append(bbox[2])
        ratios = sorted(ratios)
        scales = sorted(scales)
        anchor_ratios = []
        anchor_scales = []
        for i in range(3):
            ratio = mean(ratios[int(i*len(ratios)/3):int((i+1)*len(ratios)/3)])
            scale = mean(scales[int(i*len(ratios)/3):int((i+1)*len(ratios)/3)])
            anchor_ratios.append(ratio)
            anchor_scales.append(scale)
        return [anchor_ratios, anchor_scales]


    @staticmethod
    def calc_anchors(parameters, ground_bboxes, ratio_range, scale_range, ratio_step, scale_step):
        ratio_list = np.arange(ratio_range[0], ratio_range[1], ratio_step).tolist()
        scale_list = np.arange(scale_range[0], scale_range[1], scale_step).tolist()
        all_ratios = [[a, b, c] for a in ratio_list for b in ratio_list[ratio_list.index(a)+1:] for c in ratio_list[ratio_list.index(b)+1:]]
        all_scales = [[a, b, c] for a in scale_list for b in scale_list[scale_list.index(a)+1:] for c in scale_list[scale_list.index(b)+1:]]
        all_combinations = [[[ratio, scale] for ratio in ratios for scale in scales] for ratios in all_ratios for scales in all_scales]
        split = min(4, len(all_combinations))
        processes = []
        q = multiprocessing.Queue()
        rets = []

        for i in range(split):
            anchors = all_combinations[int(i*len(all_combinations)/split):int((i+1)*len(all_combinations)/split)]
            p = multiprocessing.Process(target=Utility.anchor_helper, args=(parameters, anchors, ground_bboxes, q))
            processes.append(p)
            p.start()
        for p in processes:
            rets.append(q.get())
        for p in processes:
            p.join()
        anchors = None
        running_score = 0
        for ret in rets:
            if running_score < ret[1]:
                running_score = ret
                anchors = ret[0]

        return anchors

    @staticmethod
    def anchor_helper(parameters, test_anchors_list, ground_bboxes, queue):
        highest_score = 0
        best_anchors = None
        count = 0
        for anchors in test_anchors_list:
            print(count/len(test_anchors_list))
            running_score = 0
            for t_bbox in ground_bboxes:
                def test_anchors(t_bbox, anchors):
                    for y in range(int(parameters.STRIDE / 2), parameters.IMAGE_HEIGHT + int(parameters.STRIDE / 2),
                                   parameters.STRIDE):
                        for x in range(int(parameters.STRIDE / 2), parameters.IMAGE_WIDTH + int(parameters.STRIDE / 2),
                                       parameters.STRIDE):
                            for anchor in anchors:
                                a_bbox = Utility.anchor_to_bbox(anchor, x, y)
                                iou = Utility.calc_iou(t_bbox, a_bbox)
                                if iou > 0.7:
                                    return 1
                    return 0

                running_score += test_anchors(t_bbox, anchors)
            if running_score > highest_score:
                highest_score = running_score
                best_anchors = anchors
            count += 1
        queue.put(([[best_anchors[i*3][0] for i in range(3)], [best_anchors[i][1] for i in range(3)]], highest_score))

    @staticmethod
    def get_one_hot(cls_list):
        out_dict = {}
        for j in range(len(cls_list)):
            zeros = [0 for i in range(len(cls_list))]
            zeros[j] = 1
            out_dict[cls_list[j]] = zeros
        return out_dict




