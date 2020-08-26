from statistics import mean
import math


def calc_iou(bbox1, bbox2):
    coords1 = bbox_to_coords(bbox1)
    coords2 = bbox_to_coords(bbox2)

    x_overlap = max(0, min(coords1[2], coords2[2]) - max(coords1[0], coords2[0]))
    y_overlap = max(0, min(coords1[3], coords2[3]) - max(coords1[1], coords2[1]))
    intersection = x_overlap * y_overlap
    union = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection
    return intersection / union


def bbox_to_coords(bbox):
    x1 = int(bbox[0] - bbox[2] / 2)
    x2 = int(bbox[0] + bbox[2] / 2)
    y1 = int(bbox[1] - bbox[3] / 2)
    y2 = int(bbox[1] + bbox[3] / 2)
    return [x1, y1, x2, y2]


def coords_to_bbox(coords):
    x = int((coords[0] + coords[2]) / 2)
    y = int((coords[1] + coords[3]) / 2)
    width = abs(coords[0] - coords[2])
    height = abs(coords[1] - coords[3])
    return [x, y, width, height]


def anchor_to_bbox(anchor, x, y):
    width = anchor[1]
    height = int(anchor[1]/anchor[0])
    return [x, y, width, height]


def apply_deltas(bbox, deltas):
    bbox[0] = (deltas[0] * bbox[2]) + bbox[0]
    bbox[1] = (deltas[1] * bbox[3]) + bbox[1]
    bbox[2] = math.exp(deltas[2]) * bbox[2]
    bbox[3] = math.exp(deltas[3]) * bbox[3]
    return bbox


def smart_calc_anchors(ground_bboxes):
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


def get_one_hot(cls_list):
    # Generates dictionary of form {"cls": one-hot array, ...}
    out_dict = {}
    for j in range(len(cls_list)):
        zeros = [0 for i in range(len(cls_list))]
        zeros[j] = 1
        out_dict[cls_list[j]] = zeros
    return out_dict




