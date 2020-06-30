# -*- coding: utf-8 -*-
# Copyright (c) 2020 LeapMind Inc. All Rights Reserved.
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

import logging
import os
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _run(model, config_file, port, threshold, output_dir):
    sys.path.append(os.path.join(output_dir, "python"))
    from motion_jpeg_server_custom import run
    run(model, config_file, port=port, threshold=threshold)


def run_server():
    base_dir = os.getenv("AWS_GG_RESOURCE_PREFIX") + "/blueoil/output"
    output_dir = os.path.join(base_dir, "output")
    model_dir = os.path.join(output_dir, "models")
    config_file = os.path.join(model_dir, "meta.yaml")
    model = os.path.join(model_dir, "lib/libdlk_fpga.so")
    port = 8080
    threshold = float(os.getenv("BOX_SCORE_THRESHOLD", default=0.5))
    logger.info("Motion JPEG Server Start!")
    _run(model, config_file, port, threshold, output_dir)


run_server()


def run_server_handler(event, context):
    return
