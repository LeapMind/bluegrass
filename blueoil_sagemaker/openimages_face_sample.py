# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from easydict import EasyDict
import tensorflow as tf

from blueoil.common import Tasks
from blueoil.networks.object_detection.lm_fyolo import LMFYoloQuantize
from blueoil.datasets.open_images_v4 import OpenImagesV4BoundingBoxBase
from blueoil.data_augmentor import (
    Brightness,
    Color,
    FlipLeftRight,
    Hue,
    SSDRandomCrop,
)
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    ResizeWithGtBoxes,
    DivideBy255,
    PerImageStandardization,
)
from blueoil.post_processor import (
    FormatYoloV2,
    ExcludeLowScoreBox,
    NMS,
)
from blueoil.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LMFYoloQuantize

DATASET_CLASS = type('DATASET_CLASS', (OpenImagesV4BoundingBoxBase,), {'extend_dir': '/opt/ml/input/data/dataset/openimages_face/', 'validation_extend_dir': '/opt/ml/input/data/dataset/openimages_face/'})

IMAGE_SIZE = [224, 224]
BATCH_SIZE = 16
DATA_FORMAT = "NHWC"
TASK = Tasks.OBJECT_DETECTION
CLASSES = ['Humanface']

MAX_EPOCHS = 400
SAVE_CHECKPOINT_STEPS = 1000
KEEP_CHECKPOINT_MAX = 1
TEST_STEPS = 1000
SUMMARISE_STEPS = 10000


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    PerImageStandardization()
])
anchors = [
    (1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)
]
score_threshold = 0.05
nms_iou_threshold = 0.5
nms_max_output_size = 100
POST_PROCESSOR = Sequence([
    FormatYoloV2(
        image_size=IMAGE_SIZE,
        classes=CLASSES,
        anchors=anchors,
        data_format=DATA_FORMAT,
    ),
    ExcludeLowScoreBox(threshold=score_threshold),
    NMS(iou_threshold=nms_iou_threshold, max_output_size=nms_max_output_size, classes=CLASSES,),
])

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.AdamOptimizer

NETWORK.OPTIMIZER_KWARGS = {}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {'values': [0.001, 0.0001, 1e-05, 1e-06], 'boundaries': [59648, 119297, 178945]}

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ANCHORS = anchors
NETWORK.OBJECT_SCALE = 5.0
NETWORK.NO_OBJECT_SCALE = 1.0
NETWORK.CLASS_SCALE = 1.0
NETWORK.COORDINATE_SCALE = 1.0
NETWORK.LOSS_IOU_THRESHOLD = 0.6
NETWORK.WEIGHT_DECAY_RATE = 0.0005
NETWORK.SCORE_THRESHOLD = score_threshold
NETWORK.NMS_IOU_THRESHOLD = nms_iou_threshold
NETWORK.NMS_MAX_OUTPUT_SIZE = nms_max_output_size
NETWORK.LOSS_WARMUP_STEPS = int(8000 / BATCH_SIZE)

# quantize
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2.0
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}
NETWORK.QUANTIZE_FIRST_CONVOLUTION = False
NETWORK.QUANTIZE_LAST_CONVOLUTION = False

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Brightness(value=(0.75, 1.25), ),
    Color(value=(0.75, 1.25), ),
    FlipLeftRight(probability=0.5, ),
    Hue(value=(-10, 10), ),
    SSDRandomCrop(min_crop_ratio=0.3, ),
])
DATASET.ENABLE_PREFETCH = True
