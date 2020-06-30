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
from blueoil.networks.classification.lmnet_v1 import LmnetV1Quantize
from blueoil.datasets.image_folder import ImageFolderBase
from blueoil.data_augmentor import (
    Brightness,
    Color,
    FlipLeftRight,
    Hue,
)
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    Resize,
    DivideBy255,
    PerImageStandardization
)
from blueoil.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LmnetV1Quantize

DATASET_CLASS = type('DATASET_CLASS', (ImageFolderBase,), {'extend_dir': '/opt/ml/input/data/dataset/cifar/train', 'validation_extend_dir': '/opt/ml/input/data/dataset/cifar/test'})

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 64
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

MAX_EPOCHS = 100
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
    Resize(size=IMAGE_SIZE),
    PerImageStandardization()
])
POST_PROCESSOR = None

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {'momentum': 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {'values': [0.001, 0.0001, 1e-05, 1e-06], 'boundaries': [25781, 51562, 77343]}

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005

# quantize
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

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
])
DATASET.ENABLE_PREFETCH = True
