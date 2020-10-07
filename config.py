import tensorflow as tf
# from keras import
from tensorflow import keras
import math
import numpy as np
import cv2
import pickle
import os
import time


class _Data:
    def __init__(self):
        self.IMAGE_WIDTH = 448
        self.IMAGE_HEIGHT = 448
        self.ANCHOR_RATIOS = [0.5, 1.0, 2.0]
        self.ANCHOR_SCALES = [75, 150, 300]
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.VALIDATION_SPLIT = 0.1
        self.STRIDE = 16
        self.NUM_CLASSES = 2

    def num_anchors(self):
        return len(self.get_anchors())

    def get_anchors(self):
        return [[r, s] for r in self.ANCHOR_RATIOS for s in self.ANCHOR_SCALES]


class Parameters(_Data):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if Parameters.__instance is None:
            Parameters.__instance = _Data()
        return Parameters.__instance
